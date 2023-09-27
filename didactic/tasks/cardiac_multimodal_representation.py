import functools
import itertools
import math
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import autogluon.multimodal.models.ft_transformer
import hydra
import rtdl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from rtdl import FeatureTokenizer
from rtdl.modules import _TokenInitialization
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
from torchmetrics.functional import accuracy, mean_absolute_error
from vital.data.augmentation.base import random_masking
from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, filter_image_attributes
from vital.data.cardinal.utils.attributes import CLINICAL_CAT_ATTR_LABELS
from vital.models.classification.mlp import MLP
from vital.tasks.generic import SharedStepsTask
from vital.utils.decorators import auto_move_data

CardiacAttribute = ClinicalAttribute | Tuple[ViewEnum, ImageAttribute]


class CardiacSequenceAttributesTokenizer(nn.Module):
    """Tokenizer that pre-processes attributes extracted from cardiac sequences for a transformer model."""

    def __init__(self, resample_dim: int, embed_dim: int = None, num_attrs: int = None):
        """Initializes class instance.

        Args:
            resample_dim: Target size for a simple interpolation resampling of the attributes. Mutually exclusive
                parameter with `cardiac_sequence_attrs_model`.
            embed_dim: Size of the embedding in which to project the resampled attributes. If not specified, no
                projection is learned and the embedding is directly the resampled attributes. Only used when
                `resample_dim` is provided.
            num_attrs: Number of attributes to tokenize. Only required when `embed_dim` is not None to initialize the
                weights and bias parameters of the learnable embeddings.
        """
        if embed_dim is not None and num_attrs is None:
            raise ValueError(
                "When opting for the resample+project method of tokenizing image attributes, you must indicate the "
                "expected attributes to initialize the weights and biases for the projection."
            )

        super().__init__()

        self.resample_dim = resample_dim

        self.weight = None
        if embed_dim:
            initialization_ = _TokenInitialization.from_str("uniform")
            self.weight = nn.Parameter(Tensor(num_attrs, resample_dim, embed_dim))
            self.bias = nn.Parameter(Tensor(num_attrs, embed_dim))
            for parameter in [self.weight, self.bias]:
                initialization_.apply(parameter, embed_dim)

    @torch.inference_mode()
    def forward(self, attrs: Dict[Any, Tensor] | Sequence[Tensor]) -> Tensor:
        """Embeds image attributes by resampling them, and optionally projecting them to the target embedding.

        Args:
            attrs: (K: S, V: (N, ?)) or S * (N, ?): Attributes to tokenize, where the dimensionality of each attribute
                can vary.

        Returns:
            (N, S, E), Tokenized version of the attributes.
        """
        if not isinstance(attrs, dict):
            attrs = {idx: attr for idx, attr in enumerate(attrs)}

        # Resample attributes to make sure all of them are of `resample_dim`
        for attr_id, attr in attrs.items():
            if attr.shape[-1] != self.resample_dim:
                # Temporarily reshape attribute batch tensor to be 3D to be able to use torch's interpolation
                # (N, ?) -> (N, `resample_dim`)
                attrs[attr_id] = F.interpolate(attr.unsqueeze(1), size=self.resample_dim, mode="linear").squeeze(dim=1)

        # Now that all attributes are of the same shape, merge them into one single tensor
        x = torch.stack(list(attrs.values()), dim=1)  # (N, S, L)

        if self.weight is not None:
            # Broadcast along all but the last two dimensions, which perform the matrix multiply
            # (N, S, 1, L) @ (S, L, E) -> (N, S, E)
            x = (x[..., None, :] @ self.weight).squeeze(dim=-2)
            x = x + self.bias[None]

        return x


class CardiacMultimodalRepresentationTask(SharedStepsTask):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        clinical_attrs: Sequence[ClinicalAttribute | str],
        img_attrs: Sequence[ImageAttribute],
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Dict[ClinicalAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig = None,
        contrastive_loss: Callable[[Tensor, Tensor], Tensor] | DictConfig = None,
        contrastive_loss_weight: float = 0,
        mask_loss: Callable[[Tensor, Tensor], Tensor] | DictConfig = None,
        mask_loss_weight: float = 0,
        constraint: Callable[[Tensor, Tensor], Tensor] | DictConfig = None,
        constraint_weight: float = 0,
        clinical_tokenizer: Optional[FeatureTokenizer | DictConfig] = None,
        img_tokenizer: Optional[nn.Module | DictConfig] = None,
        latent_token: bool = True,
        sequential_pooling: bool = False,
        mtr_p: float | Tuple[float, float] = 0,
        mt_by_attr: bool = False,
        attrs_dropout: float | Tuple[float, float] = 0,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            contrastive_loss: Self-supervised criterion to use as contrastive loss between pairs of (N, E) collections
                of feature vectors, in a contrastive learning step that follows the SCARF pretraining.
                (see ref: https://arxiv.org/abs/2106.15147)
            contrastive_loss_weight: Factor by which to weight the `contrastive_loss` in the overall loss.
            mask_loss: Criterion to use to compare the (N, S) attention mask on the input tokens to a prediction of the
                mask based on the features extracted by the encoder. This technique was proposed as a self-supervised
                pretraining method for tabular data in the following paper: https://arxiv.org/abs/2207.03208.
            mask_loss_weight: Factor by which to weight the `mask_loss` in the overall loss.
            constraint: Self-supervised criterion to use to enforce the encodings to respect arbitrary constraints.
            constraint_weight: When `constraint` is used as an auxiliary loss term, weight to use on the constraint loss
                term.
            clinical_attrs: Clinical attributes to provide to the model.
            img_attrs: Image attributes to provide to the model.
            views: Views from which to include image attributes.
            clinical_tokenizer: Tokenizer that can process clinical, i.e. patient records, data.
            img_tokenizer: Tokenizer that can process imaging data.
            latent_token: Whether to add a latent token (i.e. CLASS token) to use as the encoder's output token.
            sequential_pooling: Whether to perform sequential pooling on the encoder's output tokens. Otherwise, the
                full sequence of tokens is concatenated before being fed to the prediction head.
            mtr_p: Probability to replace tokens by the learned MASK token, following the Mask Token Replacement (MTR)
                data augmentation method.
                If a float, the value will be used as masking rate during training (disabled during inference).
                If a tuple, specify a masking rate to use during training and inference, respectively.
            mt_by_attr: Whether to use one MASK token per attribute (`True`), or one universal MASK token for all
                attributes (`False`).
            attrs_dropout: Probability of randomly masking tokens, effectively dropping them, to simulate missing data.
                If a float, the value will be used as dropout rate during training (disabled during inference).
                If a tuple, specify a dropout rate to use during training and inference, respectively.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        # Ensure string tags are converted to their appropriate enum types
        # And to it before call to the parent's `init` so that the converted values are saved in `hparams`
        clinical_attrs = tuple(ClinicalAttribute[e] for e in clinical_attrs)
        views = tuple(ViewEnum[e] for e in views)
        img_attrs = tuple(ImageAttribute[e] for e in img_attrs)

        # If dropout/masking are not single numbers, make sure they are tuples (and not another container type)
        if not isinstance(mtr_p, (int, float)):
            mtr_p = tuple(mtr_p)
        if not isinstance(attrs_dropout, (int, float)):
            attrs_dropout = tuple(attrs_dropout)

        if (contrastive_loss is None and mask_loss is None) and predict_losses is None:
            raise ValueError(
                "You should provide at least one of  `contrastive_loss`, `mask_loss` or `predict_losses`. Providing "
                "only `contrastive_loss`/`mask_loss` will run a self-supervised (pre)training phase. Providing only "
                "`predict_losses` will run a fully-supervised training phase. Finally, providing both at the same time "
                "will train the model in fully-supervised mode, with the self-supervised loss as an auxiliary term."
            )

        if latent_token and sequential_pooling:
            raise ValueError(
                "`latent_token` and `sequential_pooling` are mutually exclusive options, meant to reduce the "
                "dimensionality of the encoder's output from a sequence of tokens to only one token."
            )

        if not clinical_tokenizer and clinical_attrs:
            raise ValueError(
                f"You have requested the following attributes derived from clinical data: "
                f"{[str(attr) for attr in clinical_attrs]}, but have not configured a tokenizer for clinical-based "
                f"attributes. Either provide this tokenizer (through the `clinical_tokenizer` parameter) or remove any "
                f"clinical-based attributes (by setting the `clinical_attrs` to be an empty list)."
            )
        if not img_tokenizer and img_attrs:
            raise ValueError(
                f"You have requested the following attributes derived from imaging data: "
                f"{[str(attr) for attr in img_attrs]}, but have not configured a tokenizer for image-based "
                f"attributes. Either provide this tokenizer (through the `img_tokenizer` parameter) or remove any "
                f"image-based attributes (by setting the `img_attrs` to be an empty list)."
            )
        if not (clinical_attrs or img_attrs):
            raise ValueError(
                "You configured neither clinical attributes nor image attributes as input variables to the model, but "
                "the model requires at least one input. Set non-empty values for either or both `clinical_attrs` and "
                "`img_attrs`."
            )

        super().__init__(*args, **kwargs)

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams.lr = None

        # Add shortcut to token labels to avoid downstream applications having to determine them from hyperparameters
        self.token_tags = clinical_attrs + tuple(
            "/".join([view, attr]) for view, attr in itertools.product(views, img_attrs)
        )
        if latent_token:
            self.token_tags = self.token_tags + ("LAT",)

        # Categorise the clinical attributes (tabular data) in terms of their type (numerical vs categorical)
        self.clinical_num_attrs = [
            attr for attr in self.hparams.clinical_attrs if attr in ClinicalAttribute.numerical_attrs()
        ]
        self.clinical_cat_attrs = [
            attr for attr in self.hparams.clinical_attrs if attr in ClinicalAttribute.categorical_attrs()
        ]
        self.clinical_cat_attrs_cardinalities = [
            len(CLINICAL_CAT_ATTR_LABELS[cat_attr]) for cat_attr in self.clinical_cat_attrs
        ]

        # Extract train/test dropout/masking probabilities from their configs
        if isinstance(self.hparams.attrs_dropout, tuple):
            self.train_attrs_dropout, self.test_attrs_dropout = self.hparams.attrs_dropout
        else:
            self.train_attrs_dropout = self.hparams.attrs_dropout
            self.test_attrs_dropout = 0
        if isinstance(self.hparams.mtr_p, tuple):
            self.train_mtr_p, self.test_mtr_p = self.hparams.mtr_p
        else:
            self.train_mtr_p = self.hparams.mtr_p
            self.test_mtr_p = 0

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = {}

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
                ClinicalAttribute[attr]: hydra.utils.instantiate(attr_loss)
                if isinstance(attr_loss, DictConfig)
                else attr_loss
                for attr, attr_loss in predict_losses.items()
            }
        self.hparams.target_clinical_attrs = tuple(
            self.predict_losses
        )  # Hyperparameter to easily access target attributes
        for attr in self.predict_losses:
            if attr in ClinicalAttribute.numerical_attrs():
                self.metrics[attr] = {"mae": functools.partial(mean_absolute_error)}
            elif attr in ClinicalAttribute.binary_attrs():
                self.metrics[attr] = {"acc": functools.partial(accuracy, task="binary")}
            else:  # attr in ClinicalAttribute.categorical_attrs()
                self.metrics[attr] = {
                    "acc": functools.partial(
                        accuracy, task="multiclass", num_classes=len(CLINICAL_CAT_ATTR_LABELS[attr])
                    )
                }

        # Self-supervised losses and metrics
        self.contrastive_loss = None
        if contrastive_loss:
            self.contrastive_loss = (
                hydra.utils.instantiate(contrastive_loss)
                if isinstance(contrastive_loss, DictConfig)
                else contrastive_loss
            )
        self.mask_loss = None
        if mask_loss:
            self.mask_loss = hydra.utils.instantiate(mask_loss) if isinstance(mask_loss, DictConfig) else mask_loss

        # Latent space consistency loss
        self.constraint = None
        if constraint:
            self.constraint = hydra.utils.instantiate(constraint) if isinstance(constraint, DictConfig) else constraint

        # Compute shapes relevant for defining the models' architectures
        self.sequence_length = (
            len(self.hparams.clinical_attrs)
            + (len(self.hparams.img_attrs) * len(self.hparams.views))
            + self.hparams.latent_token
        )

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.contrastive_head, self.mask_head, self.prediction_heads = self.configure_model()

        # Configure tokenizers and extract relevant info about the models' architectures
        if isinstance(self.encoder, nn.TransformerEncoder):  # Native PyTorch `TransformerEncoder`
            self.nhead = self.encoder.layers[0].self_attn.num_heads
        elif isinstance(self.encoder, autogluon.multimodal.models.ft_transformer.FT_Transformer):  # XTab FT-Transformer
            self.nhead = self.encoder.blocks[0]["attention"].n_heads
        else:
            raise NotImplementedError(
                "To instantiate the cardiac multimodal representation task, it is necessary to determine the number of "
                f"attention heads. However, this is not implemented for the requested encoder configuration: "
                f"'{self.encoder.__class__.__name__}'. Either change the configuration, or implement the introspection "
                f"of the number of attention heads for your configuration above this warning."
            )

        if clinical_attrs:
            if isinstance(clinical_tokenizer, DictConfig):
                clinical_tokenizer = hydra.utils.instantiate(
                    clinical_tokenizer,
                    n_num_features=len(self.clinical_num_attrs),
                    cat_cardinalities=self.clinical_cat_attrs_cardinalities,
                )
        else:
            # Set tokenizer to `None` if it's not going to be used
            clinical_tokenizer = None
        self.clinical_tokenizer = clinical_tokenizer

        if img_attrs:
            if isinstance(img_tokenizer, DictConfig):
                img_tokenizer = hydra.utils.instantiate(img_tokenizer)
        else:
            # Set tokenizer to `None` if it's not going to be used
            img_tokenizer = None
        self.img_tokenizer = img_tokenizer

        # Initialize modules/parameters dependent on the encoder's configuration
        # Initialize learnable positional embedding parameters
        self.positional_embedding = Parameter(torch.empty(1, self.sequence_length, self.hparams.embed_dim))
        init.trunc_normal_(self.positional_embedding, std=0.2)

        if self.hparams.latent_token:
            # Initialize parameters of the latent token
            self.latent_token = rtdl.CLSToken(self.hparams.embed_dim, "uniform")

        if self.hparams.mtr_p:

            def _init_mask_token() -> Parameter:
                # Initialize parameters of MTR's mask token
                # Since MTR is built on top of FTT and copies implementation details (including the initialization)
                # we also default to re-using their initialization
                mask_token = Parameter(torch.empty(self.hparams.embed_dim))
                d_sqrt_inv = 1 / math.sqrt(self.hparams.embed_dim)
                init.uniform_(mask_token, a=-d_sqrt_inv, b=d_sqrt_inv)
                return mask_token

            if self.hparams.mt_by_attr:
                # Init one MASK token for each attribute
                attr_tags = self.token_tags
                if self.hparams.latent_token:
                    attr_tags = attr_tags[:-1]
                self.mask_token = nn.ParameterDict({attr: _init_mask_token() for attr in attr_tags})
            else:
                # Init a single universal MASK token
                self.mask_token = _init_mask_token()

        if self.hparams.sequential_pooling:
            # Initialize parameters used in sequential pooling
            self.attention_pool = nn.Linear(self.hparams.embed_dim, 1)

    @property
    def example_input_array(
        self,
    ) -> Tuple[Dict[ClinicalAttribute, Tensor], Dict[Tuple[ViewEnum, ImageAttribute], Tensor]]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        clinical_attrs = {attr: torch.randn(2) for attr in self.clinical_num_attrs}
        # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        clinical_attrs.update({attr: torch.randint(2, (2,)) for attr in self.clinical_cat_attrs})
        img_attrs = {
            (view, attr): torch.randn(2, self.hparams.data_params.in_shape[CardinalTag.image_attrs][1])
            for view, attr in itertools.product(self.hparams.views, self.hparams.img_attrs)
        }
        return clinical_attrs, img_attrs

    def configure_model(
        self,
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.Module], Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams.model.get("encoder"))

        # Determine the number of features at the output of the encoder
        if self.hparams.latent_token or self.hparams.sequential_pooling:
            num_features = self.hparams.embed_dim
        else:
            num_features = self.sequence_length * self.hparams.embed_dim

        # Build the projection head as an MLP with a single hidden layer and constant width, as proposed in
        # https://arxiv.org/abs/2106.15147
        contrastive_head = None
        if self.contrastive_loss:
            contrastive_head = MLP((num_features,), (num_features,), hidden=(num_features,), dropout=0)
        mask_head = None
        if self.mask_loss:
            mask_head = MLP(
                (num_features,), (self.sequence_length - self.hparams.latent_token,), hidden=(num_features,), dropout=0
            )

        # Build the prediction heads (one by clinical attribute to predict) following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = {}
            for target_clinical_attr in self.predict_losses:
                if target_clinical_attr in ClinicalAttribute.categorical_attrs():
                    if target_clinical_attr in ClinicalAttribute.binary_attrs():
                        # Binary classification target
                        output_size = 1
                    else:
                        # Multi-class classification target
                        output_size = len(CLINICAL_CAT_ATTR_LABELS[target_clinical_attr])
                else:
                    # Regression target
                    output_size = 1
                prediction_heads[target_clinical_attr] = nn.Sequential(
                    nn.LayerNorm(num_features), nn.ReLU(), nn.Linear(num_features, output_size)
                )
        prediction_heads = nn.ModuleDict(prediction_heads)

        return encoder, contrastive_head, mask_head, prediction_heads

    def configure_optimizers(self) -> Dict[Literal["optimizer", "lr_scheduler"], Any]:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. image tokenizer)."""
        return super().configure_optimizers(params=filter(lambda p: p.requires_grad, self.parameters()))

    def setup(self, stage: str) -> None:  # noqa: D102
        # Call `setup` on the constraint (if it is defined)
        # Workaround since constraints can't be implemented as callbacks whose `setup` would be called automatically
        # (because they're "essential" to the module, i.e. they are part of the training loop), but they are mostly
        # independent of the rest of the module and can be extracted and called in a plug-and-play manner
        if self.constraint:
            self.constraint.setup(self.trainer, self, stage=stage)

    @auto_move_data
    def tokenize(
        self, clinical_attrs: Dict[ClinicalAttribute, Tensor], img_attrs: Dict[Tuple[ViewEnum, ImageAttribute], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Tokenizes the input clinical and image attributes, providing a mask of non-missing attributes.

        Args:
            clinical_attrs: (K: S, V: N), Sequence of batches of clinical attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            img_attrs: (K: S, V: (N, ?)), Sequence of batches of image attributes, where the dimensionality of each
                attribute can vary.

        Returns:
            Batch of i) (N, S, E) tokens for each attribute, and ii) (N, S) mask of non-missing attributes.
        """
        # Initialize lists for cumulating (optional) tensors for each modality, that will be concatenated into tensors
        tokens, notna_mask = [], []

        # Tokenize the attributes
        if clinical_attrs:
            clinical_num_attrs, clinical_cat_attrs = None, None
            if self.clinical_num_attrs:
                # Group the numerical attributes from the `clinical_attrs` input in a single tensor
                clinical_num_attrs = torch.hstack(
                    [clinical_attrs[attr].unsqueeze(1) for attr in self.clinical_num_attrs]
                )  # (N, S_num)
            if self.clinical_cat_attrs:
                # Group the categorical attributes from the `clinical_attrs` input in a single tensor
                clinical_cat_attrs = torch.hstack(
                    [clinical_attrs[attr].unsqueeze(1) for attr in self.clinical_cat_attrs]
                )  # (N, S_cat)
            # Use "sanitized" version of the inputs, where invalid values are replaced by null/default values, for the
            # tokenization process. Since the embeddings of the missing tokens will be ignored later on using the attention
            # mask anyway, it doesn't matter that the embeddings returned are not "accurate"; it only matters that the
            # tokenization doesn't crash or returns NaNs
            # 1) Convert missing numerical attributes (NaNs) to numbers to avoid propagating NaNs
            # 2) Clip categorical labels to convert indicators of missing data (-1) into valid indices (0)
            clinical_attrs_tokens = self.clinical_tokenizer(
                x_num=torch.nan_to_num(clinical_num_attrs) if clinical_num_attrs is not None else None,
                x_cat=clinical_cat_attrs.clip(0) if clinical_cat_attrs is not None else None,
            )  # (N, S_clinical, E)
            tokens.append(clinical_attrs_tokens)

            # Identify missing data in clinical attributes
            if self.clinical_num_attrs:
                notna_mask.append(~(clinical_num_attrs.isnan()))
            if self.clinical_cat_attrs:
                notna_mask.append(clinical_cat_attrs != MISSING_CAT_ATTR)

        if img_attrs:
            img_attrs_tokens = self.img_tokenizer(img_attrs)  # S * (N, ?) -> (N, S_img, E)
            tokens.append(img_attrs_tokens)

            # Indicate that, when image tokens are requested, they are always available
            image_notna_mask = torch.full(img_attrs_tokens.shape[:2], True, device=img_attrs_tokens.device)
            notna_mask.append(image_notna_mask)

        # Cast to float to make sure tokens are not represented using double
        tokens = torch.cat(tokens, dim=1).float()  # (N, S_clinical + S_img, E)
        # Cast to bool to make sure attention mask is represented by bool
        notna_mask = torch.cat(notna_mask, dim=1).bool()  # (N, S_clinical + S_img)

        return tokens, notna_mask

    @auto_move_data
    def encode(self, tokens: Tensor, avail_mask: Tensor, apply_augments: bool = True) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output tokens for the embedding.

        Args:
            tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Mask indicating available (i.e. non-missing) tokens. Missing tokens will not be attended
                to by the encoder.
            apply_augments: Whether to perform augments on the tokens (e.g. dropout, masking). Normally augments will
                be performed differently (if not outright disabled) when not in training, but this parameter allows to
                disable them even during training. This is useful to compute "uncorrupted" views of the data for
                contrastive learning.

        Returns: (N, E) or (N, S * E), Embeddings of the input sequences. The shape of the embeddings depends on the
            selection/pooling applied on the output tokens.
        """
        # Cast attention map to float to be able to perform matmul (and the underlying addmul operations), since Pytorch
        # doesn't support addmul for int types (see this issue: https://github.com/pytorch/pytorch/issues/44428)
        avail_mask = avail_mask.float()
        # Default to attend to all non-missing tokens
        attn_mask = torch.ones_like(avail_mask)

        dropout = self.train_attrs_dropout if self.training else self.test_attrs_dropout
        if dropout and apply_augments:
            # Draw independent Bernoulli samples for each item/attribute pair in the batch, representing whether
            # to keep (1) or drop (0) attributes for each item
            dropout_dist = torch.full_like(avail_mask, 1 - dropout)
            keep_mask = torch.bernoulli(dropout_dist)

            # Repeat the sampling in case all attributes are dropped, missing or masked for an item
            while not (keep_mask * avail_mask).any(dim=1).all(dim=0):
                keep_mask = torch.bernoulli(dropout_dist)

            attn_mask *= keep_mask

        mtr_p = self.train_mtr_p if self.training else self.test_mtr_p
        if mtr_p and apply_augments:
            # Mask Token Replacement (MTR) data augmentation
            mask_token = self.mask_token
            if isinstance(mask_token, ParameterDict):
                mask_token = torch.stack(list(mask_token.values()))
            tokens, _ = random_masking(tokens, mask_token, mtr_p)

        if self.hparams.latent_token:
            # Add the latent token to the end of each item in the batch
            tokens = self.latent_token(tokens)

            # Pad attention mask to account for latent token only after dropout, so that latent token is always kept
            attn_mask = F.pad(attn_mask, (0, 1), value=1)

        # Build attention mask that avoids attending to missing tokens
        attn_mask = torch.stack(
            [item_attn_mask[None].T @ item_attn_mask[None] for item_attn_mask in attn_mask]
        )  # (N, S, S)
        # Cast attention mask back to bool and flip (because Pytorch's MHA expects true/non-zero values to mark where
        # NOT to attend)
        attn_mask = ~(attn_mask.bool())
        # Repeat the mask to have it be identical for each head of the multi-head attention
        # (to respect Pytorch's expected attention mask format)
        attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)  # (N * nhead, S, S)

        # Add positional embedding to the tokens + forward pass through the transformer encoder
        kwargs = {}
        if isinstance(self.encoder, nn.TransformerEncoder):
            kwargs["mask"] = attn_mask
        out_tokens = self.encoder(tokens + self.positional_embedding, **kwargs)

        if self.hparams.sequential_pooling:
            # Perform sequential pooling of the transformers' output tokens
            attn_vector = F.softmax(self.attention_pool(out_tokens), dim=1)  # (N, S, 1)
            broadcast_attn_vector = attn_vector.transpose(2, 1)  # (N, S, 1) -> (N, 1, S)
            out_features = (broadcast_attn_vector @ out_tokens).squeeze(1)  # (N, S, E) -> (N, E)
        elif self.hparams.latent_token:
            # Only keep the latent token (i.e. the last token) from the tokens outputted by the encoder
            out_features = out_tokens[:, -1, :]  # (N, S, E) -> (N, E)
        else:
            out_features = out_tokens.flatten(start_dim=1)  # (N, S, E) -> (N, S * E)

        return out_features

    @auto_move_data
    def forward(
        self, clinical_attrs: Dict[ClinicalAttribute, Tensor], img_attrs: Dict[Tuple[ViewEnum, ImageAttribute], Tensor]
    ) -> Tuple[Tensor, Optional[Dict[ClinicalAttribute, Tensor]]]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            clinical_attrs: (K: S, V: N) Sequence of batches of clinical attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            img_attrs: (K: S, V: (N, ?)), Sequence of batches of image attributes, where the dimensionality of each
                attribute can vary.

        Returns:
            Batch of i) features extracted by the encoder (N, E), and ii) predictions for each target in `losses`, if
            the model includes a prediction head.
        """
        in_tokens, avail_mask = self.tokenize(clinical_attrs, img_attrs)  # (N, S, E), (N, S)
        out_features = self.encode(in_tokens, avail_mask)  # (N, S, E) -> (N, E) / (N, S * E)

        # If the model includes prediction heads to predict attributes from the features, forward pass through them
        y_hat = None
        if self.prediction_heads:
            y_hat = {
                attr: prediction_head(out_features).squeeze(dim=1)
                for attr, prediction_head in self.prediction_heads.items()
            }

        return out_features, y_hat

    def _shared_step(self, batch: PatientData, batch_idx: int) -> Dict[str, Tensor]:
        # Extract clinical and image attributes from the batch
        clinical_attrs = {attr: attr_data for attr, attr_data in batch.items() if attr in self.hparams.clinical_attrs}
        img_attrs = filter_image_attributes(batch, views=self.hparams.views, attributes=self.hparams.img_attrs)

        in_tokens, avail_mask = self.tokenize(clinical_attrs, img_attrs)  # (N, S, E), (N, S)
        out_features = self.encode(in_tokens, avail_mask)  # (N, S, E) -> (N, E) / (N, S * E)

        metrics = {}
        losses = []
        if self.predict_losses:  # run fully-supervised prediction step
            metrics.update(self._prediction_shared_step(batch, batch_idx, in_tokens, avail_mask, out_features))
            losses.append(metrics["s_loss"])
        if self.contrastive_loss:  # run self-supervised contrastive step
            metrics.update(self._contrastive_shared_step(batch, batch_idx, in_tokens, avail_mask, out_features))
            losses.append(self.hparams.contrastive_loss_weight * metrics["cont_loss"])
        if self.mask_loss:  # run self-supervised mask prediction step
            metrics.update(self._mask_prediction_shared_step(batch, batch_idx, in_tokens, avail_mask, out_features))
            losses.append(self.hparams.mask_loss_weight * metrics["mask_loss"])
        if self.constraint:  # run self-supervised constraint step
            metrics.update(self._constraint_shared_step(batch, batch_idx, in_tokens, avail_mask, out_features))
            losses.append(self.hparams.constraint_weight * metrics["cstr_loss"])

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)
        return metrics

    def _prediction_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        # Forward pass through each target's prediction head
        predictions = {
            attr: prediction_head(out_features).squeeze(dim=1)
            for attr, prediction_head in self.prediction_heads.items()
        }

        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}
        for attr, loss in self.predict_losses.items():
            target, y_hat = batch[attr], predictions[attr]

            if attr in ClinicalAttribute.categorical_attrs():
                notna_mask = target != MISSING_CAT_ATTR
            else:  # attr in ClinicalAttribute.numerical_attrs():
                notna_mask = ~target.isnan()

            losses[f"{loss.__class__.__name__.lower().replace('loss', '')}/{attr}"] = loss(
                y_hat[notna_mask],
                # For BCE losses (e.g. `BCELoss`, BCEWithLogitsLoss`, etc.), the targets have to be floats,
                # so convert them from long to float
                target[notna_mask] if attr not in ClinicalAttribute.binary_attrs() else target[notna_mask].float(),
            )

            for metric_tag, metric in self.metrics[attr].items():
                metrics[f"{metric_tag}/{attr}"] = metric(y_hat[notna_mask], target[notna_mask])

        # Reduce loss across the multiple targets
        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def _contrastive_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        corrupted_out_features = out_features  # Features from a view corrupted by augmentations
        anchor_out_features = self.encode(in_tokens, avail_mask, apply_augments=False)

        # Compute the contrastive loss/metrics
        metrics = {
            "cont_loss": self.contrastive_loss(
                self.contrastive_head(anchor_out_features), self.contrastive_head(corrupted_out_features)
            )
        }

        return metrics

    def _mask_prediction_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        mask_prediction = self.mask_head(out_features)

        # Compute the loss on the prediction of the mask
        metrics = {"mask_loss": self.mask_loss(mask_prediction, avail_mask.float())}

        return metrics

    def _constraint_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        # Compute the latent consistency loss/metrics
        metrics = {"cstr_loss": self.constraint(batch["id"], out_features)}

        return metrics

    @torch.inference_mode()
    def predict_step(  # noqa: D102
        self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[Tensor, Optional[Dict[ClinicalAttribute, Tensor]]]:
        # Extract clinical and image attributes from the patient and add batch dimension
        clinical_attrs = {
            attr: attr_data[None, ...] for attr, attr_data in batch.items() if attr in self.hparams.clinical_attrs
        }
        img_attrs = {
            attr: attr_data[None, ...]
            for attr, attr_data in filter_image_attributes(
                batch, views=self.hparams.views, attributes=self.hparams.img_attrs
            ).items()
        }

        # Forward pass through the model
        out_features, predictions = self(clinical_attrs, img_attrs)

        # Remove unnecessary batch dimension from the output
        out_features = out_features.squeeze(dim=0)
        if predictions:
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}

        return out_features, predictions

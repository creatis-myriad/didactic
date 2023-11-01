import functools
import itertools
import math
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import autogluon.multimodal.models.ft_transformer
import hydra
import rtdl
import torch
from omegaconf import DictConfig
from rtdl import FeatureTokenizer
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
from torchmetrics.functional import accuracy, mean_absolute_error
from vital.data.augmentation.base import mask_tokens, random_masking
from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, filter_image_attributes
from vital.data.cardinal.utils.attributes import CLINICAL_CAT_ATTR_LABELS
from vital.models.classification.mlp import MLP
from vital.tasks.generic import SharedStepsTask
from vital.utils.decorators import auto_move_data

from didactic.models.layers import PositionalEncoding, SequentialPooling, UnimodalLogitsHead

CardiacAttribute = ClinicalAttribute | Tuple[ViewEnum, ImageAttribute]


class CardiacMultimodalRepresentationTask(SharedStepsTask):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        clinical_attrs: Sequence[ClinicalAttribute | str],
        img_attrs: Sequence[ImageAttribute],
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Dict[ClinicalAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig = None,
        ordinal_mode: bool = True,
        unimodal_head_kwargs: Dict[str, Any] | DictConfig = None,
        contrastive_loss: Callable[[Tensor, Tensor], Tensor] | DictConfig = None,
        contrastive_loss_weight: float = 0,
        clinical_tokenizer: Optional[FeatureTokenizer | DictConfig] = None,
        img_tokenizer: Optional[nn.Module | DictConfig] = None,
        latent_token: bool = True,
        sequential_pooling: bool = False,
        mtr_p: float | Tuple[float, float] = 0,
        mt_by_attr: bool = False,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            ordinal_mode: Whether to consider applicable targets as ordinal variables, which means:
                - Applying a constraint to enforce an unimodal softmax output from the prediction heads;
                - Predicting a new output for each ordinal target, namely the parameter of the unimodal softmax.
            unimodal_head_kwargs: Keyword arguments to forward to the initialization of the unimodal prediction heads.
            contrastive_loss: Self-supervised criterion to use as contrastive loss between pairs of (N, E) collections
                of feature vectors, in a contrastive learning step that follows the SCARF pretraining.
                (see ref: https://arxiv.org/abs/2106.15147)
            contrastive_loss_weight: Factor by which to weight the `contrastive_loss` in the overall loss.
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

        # If kwargs are null, set them to empty dict
        if unimodal_head_kwargs is None:
            unimodal_head_kwargs = {}

        if contrastive_loss is None and predict_losses is None:
            raise ValueError(
                "You should provide at least one of  `contrastive_loss` or `predict_losses`. Providing only "
                "`contrastive_loss` will run a self-supervised (pre)training phase. Providing only `predict_losses` "
                "will run a fully-supervised training phase. Finally, providing both at the same time will train the "
                "model in fully-supervised mode, with the self-supervised loss as an auxiliary term."
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

        # Extract train/test masking probabilities from their configs
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

        # Compute shapes relevant for defining the models' architectures
        self.sequence_length = (
            len(self.hparams.clinical_attrs)
            + (len(self.hparams.img_attrs) * len(self.hparams.views))
            + self.hparams.latent_token
        )

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.contrastive_head, self.prediction_heads = self.configure_model()

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
        self.positional_encoding = PositionalEncoding(self.sequence_length, self.hparams.embed_dim)

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
            self.sequential_pooling = SequentialPooling(self.hparams.embed_dim)

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
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.ModuleDict]]:
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

        # Build the prediction heads (one by clinical attribute to predict) following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = nn.ModuleDict()
            for target_clinical_attr in self.predict_losses:
                if (
                    target_clinical_attr in ClinicalAttribute.categorical_attrs()
                    and target_clinical_attr not in ClinicalAttribute.binary_attrs()
                ):
                    # Multi-class classification target
                    output_size = len(CLINICAL_CAT_ATTR_LABELS[target_clinical_attr])
                else:
                    # Binary classification or regression target
                    output_size = 1

                if self.hparams.ordinal_mode and target_clinical_attr in ClinicalAttribute.ordinal_attrs():
                    # For ordinal targets, use a custom prediction head to constraint the distribution of logits
                    prediction_heads[target_clinical_attr] = UnimodalLogitsHead(
                        num_features, output_size, **self.hparams.unimodal_head_kwargs
                    )
                else:
                    prediction_heads[target_clinical_attr] = nn.Sequential(
                        nn.LayerNorm(num_features), nn.ReLU(), nn.Linear(num_features, output_size)
                    )

        return encoder, contrastive_head, prediction_heads

    def configure_optimizers(self) -> Dict[Literal["optimizer", "lr_scheduler"], Any]:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. image tokenizer)."""
        return super().configure_optimizers(params=filter(lambda p: p.requires_grad, self.parameters()))

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
            # tokenization process. This is done to avoid propagating NaNs to available/valid values.
            # If the embeddings cannot be ignored later on (e.g. by using an attention mask during inference), they
            # should be replaced w/ a more distinct value to indicate that they are missing (e.g. a specific token),
            # instead of their current null/default values.
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
    def encode(self, tokens: Tensor, avail_mask: Tensor, disable_augments: bool = False) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output tokens for the embedding.

        Args:
            tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) tokens. Missing tokens can thus be
                treated distinctly from others (e.g. replaced w/ a specific mask).
            disable_augments: Whether to perform augments on the tokens (e.g. masking). Normally augments will
                be performed differently (if not outright disabled) when not in training, but this parameter allows to
                disable them even during training. This is useful to compute "uncorrupted" views of the data for
                contrastive learning.

        Returns: (N, E) or (N, S * E), Embeddings of the input sequences. The shape of the embeddings depends on the
            selection/pooling applied on the output tokens.
        """
        mask_token = self.mask_token
        if isinstance(mask_token, ParameterDict):
            mask_token = torch.stack(list(mask_token.values()))

        if mask_token is not None:
            # If a mask token is configured, substitute the missing tokens with the mask token to distinguish them from
            # the other tokens
            tokens = mask_tokens(tokens, mask_token, ~avail_mask)

        mtr_p = self.train_mtr_p if self.training else self.test_mtr_p
        if mtr_p and disable_augments:
            # Mask Token Replacement (MTR) data augmentation
            # Replace random non-missing tokens with the mask token to perturb the input
            tokens, _ = random_masking(tokens, mask_token, mtr_p)

        if self.hparams.latent_token:
            # Add the latent token to the end of each item in the batch
            tokens = self.latent_token(tokens)

        # Forward pass through the transformer encoder
        out_tokens = self.encoder(self.positional_encoding(tokens))

        if self.hparams.sequential_pooling:
            # Perform sequential pooling of the transformers' output tokens
            out_features = self.sequential_pooling(out_tokens)  # (N, S, E) -> (N, E)
        elif self.hparams.latent_token:
            # Only keep the latent token (i.e. the last token) from the tokens outputted by the encoder
            out_features = out_tokens[:, -1, :]  # (N, S, E) -> (N, E)
        else:
            out_features = out_tokens.flatten(start_dim=1)  # (N, S, E) -> (N, S * E)

        return out_features

    @auto_move_data
    def forward(
        self,
        clinical_attrs: Dict[ClinicalAttribute, Tensor],
        img_attrs: Dict[Tuple[ViewEnum, ImageAttribute], Tensor],
        task: Literal["encode", "predict", "unimodal_param", "unimodal_tau"] = "encode",
    ) -> Tensor | Dict[ClinicalAttribute, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            clinical_attrs: (K: S, V: N) Sequence of batches of clinical attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            img_attrs: (K: S, V: (N, ?)), Sequence of batches of image attributes, where the dimensionality of each
                attribute can vary.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (N, E) | (N, S * E), Batch of features extracted by the encoder.
            if `task` == 'unimodal_param`:
                ? * (M), Parameter of the unimodal logits distribution for ordinal targets.
            if `task` == 'unimodal_tau`:
                ? * (M), Temperature used to control the sharpness of the unimodal logits distribution for ordinal
                         targets.
            if `task` == 'predict' (and the model includes prediction heads):
                ? * (N), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        if task in ["unimodal_param", "unimodal_tau"] and not self.hparams.ordinal_mode:
            raise ValueError(
                "You requested to obtain some parameters of the unimodal softmax for ordinal attributes, but the model "
                "is not configured to predict unimodal ordinal targets. Either set `ordinal_mode` to `True` or change "
                "the requested inference task."
            )

        in_tokens, avail_mask = self.tokenize(clinical_attrs, img_attrs)  # (N, S, E), (N, S)
        out_features = self.encode(in_tokens, avail_mask)  # (N, S, E) -> (N, E) | (N, S * E)

        # Early return if requested task requires no prediction heads
        if task == "encode":
            return out_features

        # Forward pass through each target's prediction head
        predictions = {attr: prediction_head(out_features) for attr, prediction_head in self.prediction_heads.items()}

        # Based on the requested task, extract and format the appropriate output of the prediction heads
        match task:
            case "predict":
                if self.hparams.ordinal_mode:
                    predictions = {
                        attr: pred[0] if attr in ClinicalAttribute.ordinal_attrs() else pred
                        for attr, pred in predictions.items()
                    }
            case "unimodal_param":
                predictions = {attr: pred[1] for attr, pred in predictions.items()}
            case "unimodal_tau":
                predictions = {attr: pred[2] for attr, pred in predictions.items()}
            case _:
                raise ValueError(f"Unknown task '{task}'.")

        # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
        predictions = {attr: prediction.squeeze(dim=1) for attr, prediction in predictions.items()}
        return predictions

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

        # Compute the sum of the (weighted) losses
        metrics["loss"] = sum(losses)
        return metrics

    def _prediction_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        # Forward pass through each target's prediction head
        predictions = {}
        for attr, prediction_head in self.prediction_heads.items():
            pred = prediction_head(out_features)
            if self.hparams.ordinal_mode and attr in ClinicalAttribute.ordinal_attrs():
                # For ordinal targets, extract the logits from the multiple outputs of unimodal logits head
                pred = pred[0]
            predictions[attr] = pred.squeeze(dim=1)

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
        anchor_out_features = self.encode(in_tokens, avail_mask, disable_augments=True)

        # Compute the contrastive loss/metrics
        metrics = {
            "cont_loss": self.contrastive_loss(
                self.contrastive_head(anchor_out_features), self.contrastive_head(corrupted_out_features)
            )
        }

        return metrics

    @torch.inference_mode()
    def predict_step(  # noqa: D102
        self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0
    ) -> Tuple[
        Tensor,
        Optional[Dict[ClinicalAttribute, Tensor]],
        Optional[Dict[ClinicalAttribute, Tensor]],
        Optional[Dict[ClinicalAttribute, Tensor]],
    ]:
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

        # Encoder's output
        out_features = self(clinical_attrs, img_attrs)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(clinical_attrs, img_attrs, task="predict")

        # If the model enforces unimodal constraint on ordinal targets, output the unimodal parametrization
        unimodal_params, unimodal_taus = None, None
        if self.hparams.ordinal_mode:
            unimodal_params = self(clinical_attrs, img_attrs, task="unimodal_param")
            unimodal_taus = self(clinical_attrs, img_attrs, task="unimodal_tau")

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)
        if predictions is not None:
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
        if self.hparams.ordinal_mode:
            unimodal_params = {attr: unimodal_param.squeeze(dim=0) for attr, unimodal_param in unimodal_params.items()}
            unimodal_taus = {attr: unimodal_tau.squeeze(dim=0) for attr, unimodal_tau in unimodal_taus.items()}

        return out_features, predictions, unimodal_params, unimodal_taus

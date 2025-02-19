import itertools
import logging
import math
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple

import hydra
import torch
import vital
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict, init
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC
from vital.data.augmentation.base import mask_tokens, random_masking
from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import MISSING_CAT_ATTR, PatientData, filter_time_series_attributes
from vital.data.cardinal.utils.attributes import TABULAR_CAT_ATTR_LABELS
from vital.models.attention.layers import CLSToken, PositionalEncoding, SequencePooling
from vital.tasks.generic import SharedStepsTask
from vital.utils.decorators import auto_move_data

from didactic.models.tabular import TabularEmbedding
from didactic.models.time_series import TimeSeriesEmbedding

logger = logging.getLogger(__name__)
CardiacAttribute = TabularAttribute | Tuple[ViewEnum, TimeSeriesAttribute]


class CardiacMultimodalRepresentationTask(SharedStepsTask):
    """Multi-modal transformer to learn a representation from cardiac imaging and patient records data."""

    def __init__(
        self,
        embed_dim: int,
        tabular_attrs: Sequence[TabularAttribute | str],
        time_series_attrs: Sequence[TimeSeriesAttribute],
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        predict_losses: Dict[TabularAttribute | str, Callable[[Tensor, Tensor], Tensor]] | DictConfig = None,
        ordinal_mode: bool = True,
        contrastive_loss: Callable[[Tensor, Tensor], Tensor] | DictConfig = None,
        contrastive_loss_weight: float = 0,
        tabular_tokenizer: Optional[TabularEmbedding | DictConfig] = None,
        time_series_tokenizer: Optional[TimeSeriesEmbedding | DictConfig] = None,
        cls_token: bool = True,
        sequence_pooling: bool = False,
        mtr_p: float | Tuple[float, float] = 0,
        mt_by_attr: bool = False,
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            embed_dim: Size of the tokens/embedding for all the modalities.
            tabular_attrs: Tabular attributes to provide to the model.
            time_series_attrs: Time-series attributes to provide to the model.
            views: Views from which to include time-series attributes.
            predict_losses: Supervised criteria to measure the error between the predicted attributes and their real
                value.
            ordinal_mode: Whether to consider applicable targets as ordinal variables, which means:
                - Applying a constraint to enforce an unimodal softmax output from the prediction heads;
                - Predicting a new continuum value for each ordinal target, namely the param. of the unimodal softmax.
            contrastive_loss: Self-supervised criterion to use as contrastive loss between pairs of (N, E) collections
                of feature vectors, in a contrastive learning step that follows the SCARF pretraining.
                (see ref: https://arxiv.org/abs/2106.15147)
            contrastive_loss_weight: Factor by which to weight the `contrastive_loss` in the overall loss.
            tabular_tokenizer: Tokenizer that can process tabular, i.e. patient records, data.
            time_series_tokenizer: Tokenizer that can process time-series data.
            cross_attention_module: Module to use for cross-attention between the tabular and time-series tokens.
            cls_token: If `True`, adds a CLS token to use as the encoder's output token. Mutually exclusive with
                `sequence_pooling`.
            sequence_pooling: If `True`, the output token is obtained by sequence pooling over the encoder's output
                tokens. Mutually exclusive with `cls_token`.
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
        # And do it before call to the parent's `init` so that the converted values are saved in `hparams`
        tabular_attrs = tuple(TabularAttribute[e] for e in tabular_attrs)
        views = tuple(ViewEnum[e] for e in views)
        time_series_attrs = tuple(TimeSeriesAttribute[e] for e in time_series_attrs)

        # If dropout/masking are not single numbers, make sure they are tuples (and not another container type)
        if not isinstance(mtr_p, (int, float)):
            mtr_p = tuple(mtr_p)

        if contrastive_loss is None and predict_losses is None:
            raise ValueError(
                "You should provide at least one of  `contrastive_loss` or `predict_losses`. Providing only "
                "`contrastive_loss` will run a self-supervised (pre)training phase. Providing only `predict_losses` "
                "will run a fully-supervised training phase. Finally, providing both at the same time will train the "
                "model in fully-supervised mode, with the self-supervised loss as an auxiliary term."
            )

        if not tabular_tokenizer and tabular_attrs:
            raise ValueError(
                f"You have requested the following tabular attributes: "
                f"{[str(attr) for attr in tabular_attrs]}, but have not configured a tokenizer for tabular attributes. "
                f"Either provide this tokenizer (through the `tabular_tokenizer` parameter) or remove any tabular "
                f"attributes (by setting the `tabular_attrs` to be an empty list)."
            )
        if time_series_attrs:
            if not time_series_tokenizer:
                raise ValueError(
                    f"You have requested the following time-series attributes: "
                    f"{[str(attr) for attr in time_series_attrs]}, but have not configured a tokenizer for time-series "
                    f"attributes. Either provide this tokenizer (through the `time_series_tokenizer` parameter) or "
                    f"remove any time-series attributes (by setting the `time_series_attrs` to be an empty list)."
                )
            if getattr(time_series_tokenizer, "model") is None:
                logger.warning(
                    f"You have requested the following time-series attributes: "
                    f"{[str(attr) for attr in time_series_attrs]}, but have not configured a model for the tokenizer "
                    f"for time-series attributes. The tokenizer's model is optional, but highly recommended, so this "
                    f"is likely an oversight. You can provide this model through the `time_series_tokenizer.model` "
                    f"parameter."
                )
        if not (tabular_attrs or time_series_attrs):
            raise ValueError(
                "You configured neither tabular attributes nor time-series attributes as input variables to the model, "
                "but the model requires at least one input. Set non-empty values for either or both `tabular_attrs` "
                "and `time_series_attrs`."
            )
        if cls_token and sequence_pooling:
            raise ValueError(
                "You have enabled both the CLS token and sequence pooling. These are two mutually exclusive options to "
                "define how to extract the encoder's output token."
            )

        super().__init__(*args, **kwargs)

        if self.hparams.model.encoder.get("n_bidirectional_blocks", None) and not (tabular_attrs and time_series_attrs):
            raise ValueError(
                "You have configured a multimodal cross-attention module, but either the tabular or the time-series "
                "tabular or the time-series attributes are missing. Make sure to provide both tabular and time-series "
                "attributes when configuring a cross-attention module."
            )

        # TOFIX: Hack to log time-series tokenizer model's hparams when it's a config for a `torch.nn.Sequential` object
        # In that case, we have to use a `ListConfig` for the reserved `_args_` key. However, the automatic
        # serialization of `_args_` fails (w/ a '`DictConfig' not JSON serializable' error). Therefore, we fix it by
        # manually unpacking and logging the first and only element in the `_args_` `ListConfig`
        if isinstance(time_series_tokenizer, DictConfig):
            if time_series_tokenizer.get("model", {}).get("_target_") == "torch.nn.Sequential":
                self.save_hyperparameters(
                    {"time_series_tokenizer/model/_args_/0": time_series_tokenizer.model._args_[0]}
                )

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams.lr = None

        # Add shortcut to token labels to avoid downstream applications having to determine them from hyperparameters
        self.token_tags = (
            tuple("/".join([view, attr]) for view, attr in itertools.product(views, time_series_attrs)) + tabular_attrs
        )
        if cls_token:
            self.token_tags = self.token_tags + ("CLS",)

        # Categorise the tabular attributes in terms of their type (numerical vs categorical)
        self.tabular_num_attrs = [
            attr for attr in self.hparams.tabular_attrs if attr in TabularAttribute.numerical_attrs()
        ]
        self.tabular_cat_attrs = [
            attr for attr in self.hparams.tabular_attrs if attr in TabularAttribute.categorical_attrs()
        ]
        self.tabular_cat_attrs_cardinalities = [
            len(TABULAR_CAT_ATTR_LABELS[cat_attr]) for cat_attr in self.tabular_cat_attrs
        ]

        # Extract train/test masking probabilities from their configs
        if isinstance(self.hparams.mtr_p, tuple):
            self.train_mtr_p, self.test_mtr_p = self.hparams.mtr_p
        else:
            self.train_mtr_p = self.hparams.mtr_p
            self.test_mtr_p = 0

        # Configure losses/metrics to compute at each train/val/test step
        self.metrics = nn.ModuleDict()

        # Supervised losses and metrics
        self.predict_losses = {}
        if predict_losses:
            self.predict_losses = {
                TabularAttribute[attr]: hydra.utils.instantiate(attr_loss)
                if isinstance(attr_loss, DictConfig)
                else attr_loss
                for attr, attr_loss in predict_losses.items()
            }
        self.hparams.target_tabular_attrs = tuple(
            self.predict_losses
        )  # Hyperparameter to easily access target attributes
        for attr in self.predict_losses:
            if attr in TabularAttribute.numerical_attrs():
                self.metrics[attr] = nn.ModuleDict({"mae": MeanAbsoluteError()})
            elif attr in TabularAttribute.binary_attrs():
                self.metrics[attr] = nn.ModuleDict(
                    {
                        "acc": BinaryAccuracy(),
                        "auroc": BinaryAUROC(),
                    }
                )
            else:  # attr in TabularAttribute.categorical_attrs()
                num_classes = len(TABULAR_CAT_ATTR_LABELS[attr])
                self.metrics[attr] = nn.ModuleDict(
                    {
                        "acc": MulticlassAccuracy(num_classes=num_classes, average="none"),
                        "auroc": MulticlassAUROC(num_classes=num_classes, average="none"),
                    }
                )
        # Switch on ordinal mode if i) it's enabled, and ii) there are ordinal targets to predict
        self.hparams.ordinal_mode = self.hparams.ordinal_mode and any(
            attr in TabularAttribute.ordinal_attrs() for attr in self.predict_losses
        )

        # Self-supervised losses and metrics
        self.contrastive_loss = None
        if contrastive_loss:
            self.contrastive_loss = (
                hydra.utils.instantiate(contrastive_loss)
                if isinstance(contrastive_loss, DictConfig)
                else contrastive_loss
            )

        # Compute shapes relevant for defining the models' architectures
        self.n_tabular_attrs = len(self.hparams.tabular_attrs)
        self.n_time_series_attrs = len(self.hparams.time_series_attrs) * len(self.hparams.views)
        self.sequence_length = self.n_time_series_attrs + self.n_tabular_attrs + self.hparams.cls_token

        # Initialize transformer encoder and self-supervised + prediction heads
        self.encoder, self.contrastive_head, self.prediction_heads = self.configure_model()

        # Configure tokenizers and extract relevant info about the models' architectures
        self.multimodal_encoder = False
        if isinstance(self.encoder, nn.TransformerEncoder):  # Native PyTorch `TransformerEncoder`
            self.nhead = self.encoder.layers[0].self_attn.num_heads
        elif isinstance(self.encoder, vital.models.attention.transformer.Transformer):  # vital submodule `Transformer`
            self.nhead = self.hparams.model.encoder.attention_n_heads
            self.multimodal_encoder = bool(self.hparams.model.encoder.n_bidirectional_blocks)
        else:
            raise NotImplementedError(
                "To instantiate the cardiac multimodal representation task, it is necessary to determine the number of "
                f"attention heads. However, this is not implemented for the requested encoder configuration: "
                f"'{self.encoder.__class__.__name__}'. Either change the configuration, or implement the introspection "
                f"of the number of attention heads for your configuration above this warning."
            )

        if tabular_attrs:
            if isinstance(tabular_tokenizer, DictConfig):
                tabular_tokenizer = hydra.utils.instantiate(
                    tabular_tokenizer,
                    n_num_features=len(self.tabular_num_attrs),
                    cat_cardinalities=self.tabular_cat_attrs_cardinalities,
                )
        else:
            # Set tokenizer to `None` if it's not going to be used
            tabular_tokenizer = None
        self.tabular_tokenizer = tabular_tokenizer

        if time_series_attrs:
            if isinstance(time_series_tokenizer, DictConfig):
                time_series_tokenizer = hydra.utils.instantiate(time_series_tokenizer)
        else:
            # Set tokenizer to `None` if it's not going to be used
            time_series_tokenizer = None
        self.time_series_tokenizer = time_series_tokenizer

        # Initialize modules/parameters dependent on the encoder's configuration

        # Initialize learnable positional embedding parameters
        self.positional_encoding = PositionalEncoding(self.sequence_length, self.hparams.embed_dim)

        # Initialize parameters of method for reducing the dimensionality of the encoder's output to only one token
        if self.hparams.cls_token:
            self.cls_token = CLSToken(self.hparams.embed_dim)
        elif self.hparams.sequence_pooling:
            self.sequence_pooling = SequencePooling(self.hparams.embed_dim)

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
                if self.hparams.cls_token:
                    attr_tags = attr_tags[:-1]
                self.mask_token = nn.ParameterDict({attr: _init_mask_token() for attr in attr_tags})
            else:
                # Init a single universal MASK token
                self.mask_token = _init_mask_token()

    @property
    def example_input_array(
        self,
    ) -> Tuple[Dict[TabularAttribute, Tensor], Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]]:
        """Redefine example input array based on the cardiac attributes provided to the model."""
        # 2 is the size of the batch in the example
        tab_attrs = {attr: torch.randn(2) for attr in self.tabular_num_attrs}
        # Only generate 0/1 labels, to avoid generating labels bigger than the number of classes, which would lead to
        # an index out of range error when looking up the embedding of the class in the categorical feature tokenizer
        tab_attrs.update({attr: torch.randint(2, (2,)) for attr in self.tabular_cat_attrs})
        time_series_attrs = {
            (view, attr): torch.randn(2, self.hparams.data_params.in_shape[CardinalTag.time_series_attrs][1])
            for view, attr in itertools.product(self.hparams.views, self.hparams.time_series_attrs)
        }
        return tab_attrs, time_series_attrs

    def configure_model(
        self,
    ) -> Tuple[nn.Module, Optional[nn.Module], Optional[nn.ModuleDict]]:
        """Build the model, which must return a transformer encoder, and self-supervised or prediction heads."""
        # Build the transformer encoder
        encoder = hydra.utils.instantiate(self.hparams.model.encoder)

        # Build the projection head for contrastive learning, if contrastive learning is enabled
        contrastive_head = None
        if self.contrastive_loss:
            contrastive_head = hydra.utils.instantiate(self.hparams.model.contrastive_head)

        # Build the prediction heads (one by tabular attribute to predict) following the architecture proposed in
        # https://arxiv.org/pdf/2106.11959
        prediction_heads = None
        if self.predict_losses:
            prediction_heads = nn.ModuleDict()
            for target_tab_attr in self.predict_losses:
                if (
                    target_tab_attr in TabularAttribute.categorical_attrs()
                    and target_tab_attr not in TabularAttribute.binary_attrs()
                ):
                    # Multi-class classification target
                    output_size = len(TABULAR_CAT_ATTR_LABELS[target_tab_attr])
                else:
                    # Binary classification or regression target
                    output_size = 1

                if self.hparams.ordinal_mode and target_tab_attr in TabularAttribute.ordinal_attrs():
                    # For ordinal targets, use a separate prediction head config
                    prediction_heads[target_tab_attr] = hydra.utils.instantiate(
                        self.hparams.model.ordinal_head, num_logits=output_size
                    )
                else:
                    prediction_heads[target_tab_attr] = hydra.utils.instantiate(
                        self.hparams.model.prediction_head, out_features=output_size
                    )

        return encoder, contrastive_head, prediction_heads

    def configure_optimizers(self) -> Dict[Literal["optimizer", "lr_scheduler"], Any]:
        """Configure optimizer to ignore parameters that should remain frozen (e.g. tokenizers)."""
        return super().configure_optimizers(params=filter(lambda p: p.requires_grad, self.parameters()))

    @auto_move_data
    def tokenize(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Tokenizes the input tabular and time-series attributes, providing a mask of non-missing attributes.

        Args:
            tabular_attrs: (K: S, V: N), Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.

        Returns:
            Batch of i) (N, S, E) tokens for each attribute, and ii) (N, S) mask of non-missing attributes.
        """
        # Initialize lists for cumulating (optional) tensors for each modality, that will be concatenated into tensors
        tokens, notna_mask = [], []

        # Tokenize the attributes
        if time_series_attrs:
            time_series_attrs_tokens = self.time_series_tokenizer(time_series_attrs)  # S * (N, ?) -> (N, S_ts, E)
            tokens.append(time_series_attrs_tokens)

            # Indicate that, when time-series tokens are requested, they are always available
            time_series_notna_mask = torch.full(
                time_series_attrs_tokens.shape[:2], True, device=time_series_attrs_tokens.device
            )
            notna_mask.append(time_series_notna_mask)

        if tabular_attrs:
            num_attrs, cat_attrs = None, None
            if self.tabular_num_attrs:
                # Group the numerical attributes from the `tabular_attrs` input in a single tensor
                num_attrs = torch.hstack(
                    [tabular_attrs[attr].unsqueeze(1) for attr in self.tabular_num_attrs]
                )  # (N, S_num)
            if self.tabular_cat_attrs:
                # Group the categorical attributes from the `tabular_attrs` input in a single tensor
                cat_attrs = torch.hstack(
                    [tabular_attrs[attr].unsqueeze(1) for attr in self.tabular_cat_attrs]
                )  # (N, S_cat)
            # Use "sanitized" version of the inputs, where invalid values are replaced by null/default values, for the
            # tokenization process. This is done to avoid propagating NaNs to available/valid values.
            # If the embeddings cannot be ignored later on (e.g. by using an attention mask during inference), they
            # should be replaced w/ a more distinct value to indicate that they are missing (e.g. a specific token),
            # instead of their current null/default values.
            # 1) Convert missing numerical attributes (NaNs) to numbers to avoid propagating NaNs
            # 2) Clip categorical labels to convert indicators of missing data (-1) into valid indices (0)
            tab_attrs_tokens = self.tabular_tokenizer(
                x_num=torch.nan_to_num(num_attrs) if num_attrs is not None else None,
                x_cat=cat_attrs.clip(0) if cat_attrs is not None else None,
            )  # (N, S_tab, E)
            tokens.append(tab_attrs_tokens)

            # Identify missing data in tabular attributes
            if self.tabular_num_attrs:
                notna_mask.append(~(num_attrs.isnan()))
            if self.tabular_cat_attrs:
                notna_mask.append(cat_attrs != MISSING_CAT_ATTR)

        # Cast to float to make sure tokens are not represented using double
        tokens = torch.cat(tokens, dim=1).float()  # (N, S_ts + S_tab, E)
        # Cast to bool to make sure attention mask is represented by bool
        notna_mask = torch.cat(notna_mask, dim=1).bool()  # (N, S_ts + S_tab)

        return tokens, notna_mask

    def preprocess_tokens(self, tokens: Tensor, avail_mask: Tensor, enable_augments: bool = False) -> Tensor:
        """Preprocesses the input tokens, optionally masking missing data and random tokens to cause perturbations.

        Args:
            tokens: (N, S, E) Tokens to preprocess.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) tokens.
            enable_augments: Whether to perform augments on the tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns:
            Tokens with missing data masked and/or random tokens replaced by the mask token.
        """
        mask_token = self.mask_token
        if isinstance(mask_token, ParameterDict):
            mask_token = torch.stack(list(mask_token.values()))

        if mask_token is not None:
            # If a mask token is configured, substitute the missing tokens with the mask token to distinguish them from
            # the other tokens
            tokens = mask_tokens(tokens, mask_token, ~avail_mask)

        mtr_p = self.train_mtr_p if self.training else self.test_mtr_p
        if mtr_p and enable_augments:
            # Mask Token Replacement (MTR) data augmentation
            # Replace random non-missing tokens with the mask token to perturb the input
            tokens, _ = random_masking(tokens, mask_token, mtr_p)

        return tokens

    @auto_move_data
    def encode(self, tokens: Tensor, avail_mask: Tensor, enable_augments: bool = False) -> Tensor:
        """Embeds input sequences using the encoder model, optionally selecting/pooling output tokens for the embedding.

        Args:
            tokens: (N, S, E), Tokens to feed to the encoder.
            avail_mask: (N, S), Boolean mask indicating available (i.e. non-missing) tokens. Missing tokens can thus be
                treated distinctly from others (e.g. replaced w/ a specific mask).
            enable_augments: Whether to perform augments on the tokens (e.g. masking) to obtain a "corrupted" view for
                contrastive learning. Augments are already configured differently for training/testing (to avoid
                stochastic test-time predictions), so this parameter is simply useful to easily toggle augments on/off
                to obtain contrasting views.

        Returns: (N, E), Embeddings of the input sequences.
        """
        tokens = self.preprocess_tokens(tokens, avail_mask, enable_augments=enable_augments)

        if self.hparams.cls_token:
            # Add the CLS token to the end of each item in the batch
            tokens = self.cls_token(tokens)

        # Add positional encoding to the tokens
        tokens = self.positional_encoding(tokens)

        if self.multimodal_encoder:
            # Split the sequence of tokens into tabular and time-series tokens
            ts_tokens, tab_cls_tokens = tokens[:, : self.n_time_series_attrs], tokens[:, self.n_time_series_attrs :]

            # Forward pass through the transformer encoder (starting with the cross-attention module)
            out_tokens = self.encoder(ts_tokens, tab_cls_tokens)

        else:
            # Forward pass through the transformer encoder
            out_tokens = self.encoder(tokens)

        if self.hparams.cls_token:
            # Only keep the CLS token (i.e. the last token) from the tokens outputted by the encoder
            out_features = out_tokens[:, -1, :]  # (N, S, E) -> (N, E)
        elif self.hparams.sequence_pooling:
            # Perform sequence pooling of the transformers' output tokens
            out_features = self.sequence_pooling(out_tokens)  # (N, S, E) -> (N, E)
        else:
            # Perform average pooling of the transformers' output tokens
            out_features = out_tokens.mean(dim=1)  # (N, S, E) -> (N, E)

        return out_features

    @auto_move_data
    def forward(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        task: Literal["encode", "predict", "continuum_param", "continuum_tau"] = "encode",
    ) -> Tensor | Dict[TabularAttribute, Tensor]:
        """Performs a forward pass through i) the tokenizer, ii) the transformer encoder and iii) the prediction head.

        Args:
            tabular_attrs: (K: S, V: N) Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if `task` == 'encode':
                (N, E), Batch of features extracted by the encoder.
            if `task` == 'continuum_param`:
                ? * (M), Parameter of the unimodal logits distribution for ordinal targets.
            if `task` == 'continuum_tau`:
                ? * (M), Temperature used to control the sharpness of the unimodal logits distribution for ordinal
                         targets.
            if `task` == 'predict' (and the model includes prediction heads):
                ? * (N), Prediction for each target in `losses`.
        """
        if task != "encode" and not self.prediction_heads:
            raise ValueError(
                "You requested to perform a prediction task, but the model does not include any prediction heads."
            )
        if task in ["continuum_param", "continuum_tau"] and not self.hparams.ordinal_mode:
            raise ValueError(
                "You requested to obtain some parameters for ordinal attributes, but the model is not configured to "
                "predict ordinal targets. Either set `ordinal_mode` to `True` or change the requested inference task."
            )

        in_tokens, avail_mask = self.tokenize(tabular_attrs, time_series_attrs)  # (N, S, E), (N, S)
        out_features = self.encode(in_tokens, avail_mask)  # (N, S, E) -> (N, E)

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
                        attr: pred[0] if attr in TabularAttribute.ordinal_attrs() else pred
                        for attr, pred in predictions.items()
                    }
            case "continuum_param":
                predictions = {attr: pred[1] for attr, pred in predictions.items()}
            case "continuum_tau":
                predictions = {attr: pred[2] for attr, pred in predictions.items()}
            case _:
                raise ValueError(f"Unknown task '{task}'.")

        # Squeeze out the singleton dimension from the predictions' features (only relevant for scalar predictions)
        predictions = {attr: prediction.squeeze(dim=1) for attr, prediction in predictions.items()}
        return predictions

    def _shared_step(self, batch: PatientData, batch_idx: int) -> Dict[str, Tensor]:
        # Extract tabular and time-series attributes from the batch
        tabular_attrs = {attr: attr_data for attr, attr_data in batch.items() if attr in self.hparams.tabular_attrs}
        time_series_attrs = filter_time_series_attributes(
            batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
        )

        in_tokens, avail_mask = self.tokenize(tabular_attrs, time_series_attrs)  # (N, S, E), (N, S)
        out_features = self.encode(in_tokens, avail_mask)  # (N, S, E) -> (N, E)

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
            if self.hparams.ordinal_mode and attr in TabularAttribute.ordinal_attrs():
                # For ordinal targets, extract the logits from the multiple outputs of classification head
                pred = pred[0]
            predictions[attr] = pred.squeeze(dim=1)

        # Compute the loss/metrics for each target attribute, ignoring items for which targets are missing
        losses, metrics = {}, {}
        for attr, loss in self.predict_losses.items():
            target, y_hat = batch[attr], predictions[attr]

            if attr in TabularAttribute.categorical_attrs():
                notna_mask = target != MISSING_CAT_ATTR
            else:  # attr in TabularAttribute.numerical_attrs():
                notna_mask = ~target.isnan()

            losses[f"{loss.__class__.__name__.lower().replace('loss', '')}/{attr}"] = loss(
                y_hat[notna_mask],
                # For BCE losses (e.g. `BCELoss`, BCEWithLogitsLoss`, etc.), the targets have to be floats,
                # so convert them from long to float
                target[notna_mask] if attr not in TabularAttribute.binary_attrs() else target[notna_mask].float(),
            )

            for metric_tag, metric in self.metrics[attr].items():
                metric_res = metric(y_hat[notna_mask], target[notna_mask])

                # For multiclass categorical attributes, metrics are not averaged by default, so log them for each
                # class separately and then average them manually
                if attr in TabularAttribute.categorical_attrs():
                    for class_label, metric_res_for_class in zip(TABULAR_CAT_ATTR_LABELS[attr], metric_res):
                        metrics[f"{metric_tag}/{attr}/{class_label}"] = metric_res_for_class
                    metric_res = metric_res.mean()

                metrics[f"{metric_tag}/{attr}"] = metric_res

        # Reduce loss across the multiple targets
        losses["s_loss"] = torch.stack(list(losses.values())).mean()
        metrics.update(losses)

        return metrics

    def _contrastive_shared_step(
        self, batch: PatientData, batch_idx: int, in_tokens: Tensor, avail_mask: Tensor, out_features: Tensor
    ) -> Dict[str, Tensor]:
        # Extract features from the original view + from a view corrupted by augmentations
        anchor_out_features = out_features
        corrupted_out_features = self.encode(in_tokens, avail_mask, enable_augments=True)

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
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
        Optional[Dict[TabularAttribute, Tensor]],
    ]:
        # Extract tabular and time-series attributes from the patient and add batch dimension
        tabular_attrs = {
            attr: attr_data[None, ...] for attr, attr_data in batch.items() if attr in self.hparams.tabular_attrs
        }
        time_series_attrs = {
            attr: attr_data[None, ...]
            for attr, attr_data in filter_time_series_attributes(
                batch, views=self.hparams.views, attrs=self.hparams.time_series_attrs
            ).items()
        }

        # Encoder's output
        out_features = self(tabular_attrs, time_series_attrs)

        # If the model has targets to predict, output the predictions
        predictions = None
        if self.prediction_heads:
            predictions = self(tabular_attrs, time_series_attrs, task="predict")

        # If the model enforces constraint on ordinal targets, output the continuum parametrization
        continuum_params, continuum_taus = None, None
        if self.hparams.ordinal_mode:
            continuum_params = self(tabular_attrs, time_series_attrs, task="continuum_param")
            continuum_taus = self(tabular_attrs, time_series_attrs, task="continuum_tau")

        # Remove unnecessary batch dimension from the different outputs
        # (only do this once all downstream inferences have been performed)
        out_features = out_features.squeeze(dim=0)
        if predictions is not None:
            predictions = {attr: prediction.squeeze(dim=0) for attr, prediction in predictions.items()}
        if self.hparams.ordinal_mode:
            continuum_params = {
                attr: continuum_param.squeeze(dim=0) for attr, continuum_param in continuum_params.items()
            }
            continuum_taus = {attr: continuum_tau.squeeze(dim=0) for attr, continuum_tau in continuum_taus.items()}

        return out_features, predictions, continuum_params, continuum_taus

import itertools
import logging
from enum import auto, unique
from pathlib import Path
from typing import Dict, Literal, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig
from strenum import SnakeCaseStrEnum
from torch import Tensor, nn
from torch.nn import functional as F
from vital.data.cardinal.config import CardinalTag, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import PatientData, filter_time_series_attributes
from vital.tasks.generic import SharedStepsTask
from vital.utils.decorators import auto_move_data
from vital.utils.norm import minmax_scaling, scale
from vital.utils.saving import load_from_checkpoint

logger = logging.getLogger(__name__)


@unique
class _AttributeNormalization(SnakeCaseStrEnum):
    """Names of the available strategies for normalizing time-series attributes values."""

    data = auto()
    """Normalize the attributes' values w.r.t. statistics computed on the whole training dataset."""
    metrics = auto()
    """Normalize the metrics computed on the time-series attributes w.r.t. statistics computed on each mini-batch."""


@unique
class _AttributeStatistic(SnakeCaseStrEnum):
    """Statistics about the time-series attributes that are computed on the dataset and stored inside the model."""

    min = auto()
    max = auto()


class CardiacSequenceAttributesAutoencoder(SharedStepsTask):
    """Autoencoder pipeline specialized for cardiac sequences time-series attributes."""

    def __init__(
        self,
        views: Sequence[ViewEnum] = tuple(ViewEnum),
        attrs: Sequence[TimeSeriesAttribute] = tuple(TimeSeriesAttribute),
        normalization: _AttributeNormalization = _AttributeNormalization.data,
        reconstruction_loss: nn.Module | DictConfig = nn.L1Loss(),
        *args,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            views: Views to train the model on.
            attrs: Time-series attributes to train the model on.
            normalization: Strategy to use to normalize time-series attributes values.
            reconstruction_loss: Criterion to measure the reconstruction error on the time-series attribute curves, or
                Hydra config object describing how to instantiate such criterion.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)

        # Add shortcut to lr to work with Lightning's learning rate finder
        self.hparams.lr = None

        model = self.configure_model()
        self.encoder = model.encoder
        self.decoder = model.decoder

        if isinstance(reconstruction_loss, DictConfig):
            reconstruction_loss = hydra.utils.instantiate(reconstruction_loss)
        self.reconstruction_loss = reconstruction_loss
        self._reconstruction_loss_name = self.reconstruction_loss.__class__.__name__.lower().replace("loss", "")

        # Register buffers for time-series attributes (needs to be in `__init__`)
        attrs_stats_defaults = {
            _AttributeStatistic.min: torch.finfo().max,
            _AttributeStatistic.max: torch.finfo().min,
        }
        for view_enum, attr, (stat, default_val) in itertools.product(
            self.hparams.views, self.hparams.attrs, attrs_stats_defaults.items()
        ):
            self.register_buffer("_".join((view_enum, attr, stat)), torch.tensor(default_val))

    @property
    def example_input_array(self) -> Tensor:
        """Redefine example input array based only on the time-series attributes modality."""
        attrs_shape = self.hparams.data_params.in_shape[CardinalTag.time_series_attrs]
        return torch.randn((2, 1, attrs_shape[1]))

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the model's latent space."""
        return self.hparams.model.latent_dim

    @property
    def in_shape(self) -> Tuple[int, int]:
        """Dimensionality of one test-time input sample expected by the model."""
        return self.example_input_array.shape[1:]

    @property
    def reconstruction_loss_scale(self) -> int:
        """Power by which the reconstruction loss scales its values."""
        match self.reconstruction_loss:
            case nn.L1Loss():
                reconstruction_loss_scale = 1
            case nn.MSELoss():
                reconstruction_loss_scale = 2
            case _:
                raise NotImplementedError(
                    f"Could not determine the scale of reconstruction loss '{self.reconstruction_loss}'. Please add a "
                    f"case in '{self.__class__.__name__}'s constructor to instruct which scale to use for this loss."
                )
        return reconstruction_loss_scale

    def _get_attr_bounds(self, attr: Tuple[ViewEnum, TimeSeriesAttribute]) -> Tuple[Tensor, Tensor]:
        """Access the stored min/max bounds related to a time-series attribute.

        Args:
            attr: Key identifying the attribute for which to look up the bounds.

        Returns:
            Min/max bounds for the requested time-series attribute.
        """
        return self._get_attr_stat(attr, _AttributeStatistic.min), self._get_attr_stat(attr, _AttributeStatistic.max)

    def _get_attr_stat(self, attr: Tuple[ViewEnum, TimeSeriesAttribute], stat: _AttributeStatistic) -> Tensor:
        """Access a statistic related to a time-series attribute, saved as a torch buffer inside the model.

        Args:
            attr: Key identifying the attribute for which to look up the statistic.
            stat: Statistic to look up.

        Returns:
            Statistic for the requested time-series attribute.
        """
        return getattr(self, "_".join((*attr, stat)))

    def _set_attr_stat(
        self, attr: Tuple[ViewEnum, TimeSeriesAttribute], stat: _AttributeStatistic, val: Tensor
    ) -> None:
        """Sets the value of a statistic related to a time-series attribute, saved as a torch buffer inside the model.

        Args:
            attr: Key identifying the attribute for which to set the statistic.
            stat: Statistic to set.
            val: Value to set for the statistic.
        """
        setattr(self, "_".join((*attr, stat)), val)

    def configure_model(self) -> nn.Module:
        """Configure the network architecture used by the system."""
        attrs_shape = self.hparams.data_params.in_shape[CardinalTag.time_series_attrs]
        model = hydra.utils.instantiate(self.hparams.model, input_shape=(1, attrs_shape[-1]))
        return model

    def on_fit_start(self) -> None:
        """Computes global statistics for the time-series attributes on the training subset.

        These stats will be used during training and inference to normalize attributes values or metrics
        """
        attrs_stats_update_fn = {
            _AttributeStatistic.min: lambda cur_stat, attr_batch: min(cur_stat, attr_data.min()),
            _AttributeStatistic.max: lambda cur_stat, attr_batch: max(cur_stat, attr_data.max()),
        }
        train_dl = self.trainer.datamodule.train_dataloader()
        for batch in train_dl:
            for (stat, update_fn), (attr, attr_data) in itertools.product(
                attrs_stats_update_fn.items(),
                filter_time_series_attributes(batch, views=self.hparams.views, attrs=self.hparams.attrs).items(),
            ):
                self._set_attr_stat(attr, stat, update_fn(self._get_attr_stat(attr, stat), attr_data))

    @classmethod
    def preprocess_attr_data(
        cls, data: Tensor, bounds: Tuple[float | Tensor, float | Tensor] = None, length: int = None
    ) -> Tensor:
        """Pre-processes input data, to prepare it for inference using the autoencoder.

        Args:
            data: (N, C, L), Input data to process to prepare for inference.
            bounds: If applicable, minimum/maximum bounds to use to perform min-max scaling on the data.
            length: If applicable, length expected by the autoencoder, to reach by linearly interpolating the data.

        Returns:
            (N, C, M), Pre-processed input ready to be fed to the autoencoder.
        """
        if length is not None and length != data.shape[-1]:
            # Interpolate data to reach the requested number of data points
            data = F.interpolate(data, size=length, mode="linear")
        if bounds is not None:
            # Make sure the data is scaled between 0 and 1
            data = minmax_scaling(data, bounds=bounds)
        return data

    @classmethod
    def postprocess_attr_prediction(
        cls, prediction: Tensor, bounds: Tuple[float | Tensor, float | Tensor] = None, length: int = None
    ) -> Tensor:
        """Post-processes reconstructed attributes, undoing pre-processing to make predictions comparable with inputs.

        Args:
            prediction: (N, C, M), Reconstructed input to post-process before comparing to the input.
            bounds: If applicable, minimum/maximum to use to invert the input's min-max scaling on the prediction.
            length: If applicable, target output length to reach by linearly interpolating the prediction.

        Returns:
            (N, C, L), Post-processed prediction ready to be compared to the input.
        """
        if bounds is not None:
            # Undo the scaling of the input data on the reconstructed signal
            prediction = scale(prediction, bounds)
        if length is not None and length != prediction.shape[-1]:
            # Interpolate data to reach the requested number of data points
            prediction = F.interpolate(prediction, size=length, mode="linear")
        return prediction

    @auto_move_data
    def forward(
        self,
        x: Tensor,
        task: Literal["encode", "decode", "reconstruct"] = "reconstruct",
        attr: Tuple[ViewEnum, TimeSeriesAttribute] = None,
        out_shape: Tuple[int, ...] = None,
    ) -> Tensor:
        """Performs test-time inference on the input.

        Args:
            x: - if ``task`` == 'decode': (N, ?), Encoding in the latent space.
               - else: (N, [C,] L), Input, with an optional dimension for the channels.
            task: Flag indicating which type of inference task to perform.
            attr: Key identifying the attribute for which to perform inference.
            out_shape: Target ([C], L) of the reconstructed output. When `task!=decode`, this is normally inferred from
                the data, and providing it in this case will lead to the output shape not matching the input.

        Returns:
            if ``task`` == 'encode':
                (N, ``Z``), Encoding of the input in the latent space.
            else:
                (N, [C,] L), Reconstructed input, with the same channels as in the input.
        """
        attr_bounds = None
        if self.hparams.normalization == _AttributeNormalization.data:
            if attr:
                attr_bounds = self._get_attr_bounds(attr)
            else:
                logger.warning(
                    "You're using a model trained to normalize each type of attribute as a pre-processing step. "
                    "However, you're missing the `attr` parameter to tell the model the type of attribute passed as "
                    "input. Because of this, the model does not automatically normalize its input, which could lead to "
                    "faulty inference. It might be because you already normalize the data yourself, but it is most "
                    "likely an error."
                )

        if out_shape is None:
            # If the user doesn't request a specific output shape
            if task == "decode":
                # If we have no input data whose output shape to match, use the default shape used during training
                out_shape = (self.in_shape[-1],)
            else:
                # Match the shape of the input data
                out_shape = x.shape[1:]
        out_channels = out_shape[0] if len(out_shape) > 1 else 0
        out_length = out_shape[-1]

        if task in ["encode", "reconstruct"]:
            # Add channel dimension if it is not in the input data
            if x.ndim == 2:
                x = x.unsqueeze(1)
            x = self.preprocess_attr_data(x, bounds=attr_bounds, length=self.in_shape[-1])
            x = self.encoder(x)
        if task in ["decode", "reconstruct"]:
            x = self.decoder(x)
            x = self.postprocess_attr_prediction(x, bounds=attr_bounds, length=out_length)
            # Eliminate channel dimension if it was not in the original data
            if not out_channels:
                x = x.squeeze(dim=1)
        return x

    def _shared_step(self, batch: PatientData, batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        attrs = filter_time_series_attributes(batch, views=self.hparams.views, attrs=self.hparams.attrs)

        if self.hparams.normalization == _AttributeNormalization.data:
            attrs = self._normalize_attrs(attrs)

        # Forward on time-series attributes
        attrs_x_hat, attrs_z = {}, {}
        for attr_key, attr_data in attrs.items():
            attrs_z[attr_key] = self.encoder(attr_data.unsqueeze(1))
            attrs_x_hat[attr_key] = self.decoder(attrs_z[attr_key]).squeeze(dim=1)

        # Compute input reconstruction metrics
        attrs_rec_losses = {
            attr_key: self.reconstruction_loss(attrs_x_hat[attr_key], attr) for attr_key, attr in attrs.items()
        }
        metrics = {
            "/".join((self._reconstruction_loss_name, *attr_key)): self.reconstruction_loss(attrs_x_hat[attr_key], attr)
            for attr_key, attr in attrs.items()
        }

        if self.hparams.normalization == _AttributeNormalization.metrics:
            # To aggregate reconstruction losses, weight the losses w.r.t. the power of the error and the domains values
            attrs_rec_losses = self._normalize_attrs_reconstruction_metrics(attrs_rec_losses)
        attrs_rec_loss = torch.mean(torch.stack(list(attrs_rec_losses.values())))
        metrics[self._reconstruction_loss_name] = attrs_rec_loss

        # Compute loss from metrics
        metrics["loss"] = attrs_rec_loss

        return metrics

    def _normalize_attrs(
        self, attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]
    ) -> Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]:
        """Normalizes attributes with different range of values.

        Args:
            attrs: Batch of attributes values to normalize.

        Returns:
            Attributes values normalized w.r.t their respective range of values.
        """
        return {
            attr_key: self.preprocess_attr_data(attr_data, bounds=self._get_attr_bounds(attr_key))
            for attr_key, attr_data in attrs.items()
        }

    def _normalize_attrs_reconstruction_metrics(
        self, attrs_metrics: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]
    ) -> Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]:
        """Normalizes reconstruction metrics computed on attributes with different range of values.

        Args:
            attrs_metrics: Reconstruction metrics computed on the attributes.

        Returns:
            Attributes' reconstruction losses normalized w.r.t their respective range of values.
        """
        return {
            attr_key: attr_rec_loss
            / (
                (
                    self._get_attr_stat(attr_key, _AttributeStatistic.max)
                    - self._get_attr_stat(attr_key, _AttributeStatistic.min)
                )
                ** self.reconstruction_loss_scale
            )
            for attr_key, attr_rec_loss in attrs_metrics.items()
        }

    @torch.inference_mode()
    def predict_step(  # noqa: D102
        self, batch: PatientData, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tuple[Tensor, Tensor]]:
        # Reconstruct the time-series attributes
        attrs = filter_time_series_attributes(batch, views=self.hparams.views, attrs=self.hparams.attrs)
        prediction = {}
        for attr_key, attr_data in attrs.items():
            if not (is_batch := attr_data.ndim == 2):
                attr_data = attr_data.unsqueeze(0)

            forward_kwargs = {}
            if self.hparams.normalization == _AttributeNormalization.data:
                forward_kwargs["attr"] = attr_key

            attr_encoding = self(attr_data, task="encode", **forward_kwargs)
            attr_reconstruction = self(attr_encoding, task="decode", out_shape=attr_data.shape[-1:], **forward_kwargs)

            if not is_batch:
                attr_encoding = attr_encoding.squeeze(dim=0)
                attr_reconstruction = attr_reconstruction.squeeze(dim=0)

            prediction[attr_key] = attr_reconstruction, attr_encoding
        return prediction


class CardiacSequenceAttributesAutoencoderTokenizer(nn.Module):
    """Tokenizer that embeds attributes extracted from cardiac sequences by encoding them using an autoencoder."""

    def __init__(self, cardiac_sequence_attrs_model: str | Path | CardiacSequenceAttributesAutoencoder = None):
        """Initializes class instance.

        Args:
            cardiac_sequence_attrs_model: Pretrained time-series attributes autoencoder model used to compress the
                attributes into tokens. Mutually exclusive parameter with `embed_dim`.
        """
        super().__init__()

        # If the time-series attributes encoder is a checkpoint rather than an instantiated network, load the model from
        # the checkpoint
        if isinstance(cardiac_sequence_attrs_model, (str, Path)):
            cardiac_sequence_attrs_model = load_from_checkpoint(cardiac_sequence_attrs_model)
        # Make sure the weights of the backend model used in the tokenizer are frozen
        # Also, the backend model needs to be saved as a class member even if it's not necessary so that the
        # tokenizer as a whole can behave as expected of a module
        # (e.g. moving it across devices is applied recursively to the backend model, etc.)
        self.autoencoder = cardiac_sequence_attrs_model.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor]) -> Tensor:
        """Encodes time-series attributes using the autoencoder.

        Args:
            attrs: (K: S, V: (N, ?)): Attributes to tokenize, where the dimensionality of each attribute can vary.

        Returns:
            (N, S, E), Tokenized version of the attributes.
        """
        return torch.stack([self.autoencoder(x, task="encode", attr=attr) for attr, x in attrs.items()], dim=1)

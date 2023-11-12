from typing import Sequence

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import BaseFinetuning
from torch import nn
from torch.nn import ParameterDict
from torch.optim import Optimizer

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask


class TransformerEncoderFreeze(Callback):
    """Freezes layers/params that are NOT to be finetuned.

    Notes:
        - Implemented as a base callback that calls some `BaseFinetuning` methods rather than a child of the
          `BaseFinetuning` callback because of an issue when coupled with the `LearningRateFinder` callback
          (see this issue: https://github.com/Lightning-AI/lightning/issues/14674)

    """

    def __init__(self, finetune_layers: Sequence[int] = None):
        """Initializes class instance.

        Args:
            finetune_layers: Enumerate the indices of transformer encoder layers to finetune (only used when pretrained
                weights are provided). If `None`, defaults to finetuning all the layers.
        """
        super().__init__()
        self.finetune_layers = finetune_layers

    # TODO: Remove this override of `setup` when able to revert to inheriting from `BaseFinetuning`
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:  # noqa: D102
        self.freeze_before_training(pl_module)

    def freeze_before_training(self, pl_module: CardiacMultimodalRepresentationTask) -> None:
        """Freezes layers/params that are NOT to be finetuned at the beginning of the training."""
        if not isinstance(pl_module.encoder, nn.TransformerEncoder):
            raise NotImplementedError(
                "Finetuning specific layers is only supported for the torch-native implementation of the transformer "
                "encoder. If you're using a different transformer implementation, you should set the "
                "`finetune_layers` flag to `None`."
            )

        # If no specific layers were set to be finetuned, default to finetuning all the layers
        if self.finetune_layers is None:
            self.finetune_layers = tuple(range(len(pl_module.encoder.layers)))

        # Identifies layers to freeze based on the config and a dynamic inspection of the model's architecture
        # NOTE: Layers that are not to be finetuned are to be frozen
        num_layers = len(pl_module.encoder.layers)
        layers_to_freeze = [
            layer_idx
            for layer_idx in range(num_layers)
            if not (
                layer_idx in self.finetune_layers  # check positive layer index
                or (layer_idx - num_layers) in self.finetune_layers  # check negative layer index
            )
        ]

        # Flag specific layers/params to freeze depending on the layers to finetune
        modules_to_freeze = []
        params_to_freeze = []

        # Freeze the encoder layers that are not to be finetuned
        for layer_idx in layers_to_freeze:
            modules_to_freeze.append(pl_module.encoder.layers[layer_idx])

        if 0 in layers_to_freeze:
            # If encoder's first layer is frozen, then it is also necessary to freeze everything upstream of
            # the encoder (e.g. CLS token, tokenizers, positional embedding, etc.) to make sure the encoder's
            # inputs remain the same
            params_to_freeze.append(pl_module.positional_encoding)

            # Check if tokenizers are models before marking them to be frozen, so that we'll not try to freeze
            # other possible types of tokenizers that are not `nn.Module`s (e.g. functions)
            if isinstance(pl_module.tabular_tokenizer, nn.Module):
                modules_to_freeze.append(pl_module.tabular_tokenizer)
            if isinstance(pl_module.time_series_tokenizer, nn.Module):
                modules_to_freeze.append(pl_module.time_series_tokenizer)

            # Add optional models/parameters if they're used in the model
            if pl_module.hparams.cls_token:
                modules_to_freeze.append(pl_module.cls_token)
            if pl_module.hparams.sequence_pooling:
                modules_to_freeze.append(pl_module.sequence_pooling)

        if layers_to_freeze == list(range(num_layers)):
            # If all layers of the encoder are frozen
            # 1) Make sure the whole encoder is frozen, i.e. include optional layers apart from the main
            #    `TransformerEncoderLayer`s
            modules_to_freeze.append(pl_module.encoder)

            # 2) Freeze auxiliary tokens used in the training process
            if pl_module.hparams.mtr_p:
                if isinstance(pl_module.mask_token, ParameterDict):
                    params_to_freeze.extend(pl_module.mask_token.values())
                else:
                    params_to_freeze.append(pl_module.mask_token)

        # Freeze the relevant modules and parameters
        # TODO: Call parent `self.freeze` when able to revert to inheriting from `BaseFinetuning`
        BaseFinetuning.freeze(modules_to_freeze)
        for param in params_to_freeze:
            param.requires_grad = False

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        """Leave this function empty because we're not dynamically unfreezing layers."""

from typing import Any, Dict, Literal, Optional

import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from vital.utils.loggers import log_figure
from vital.utils.plot import plot_heatmap

from didactic.models.explain import attention_rollout, register_attn_weights_hook
from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask


class AttentionWeightsLogger(Callback):
    """Logs attention weights from the `MultiHeadAttention` layers of a model at given training steps."""

    def __init__(
        self,
        submodule: str = None,
        reduction: Literal["first", "mean"] = "first",
        compute_attention_rollout: bool = True,
        attention_rollout_kwargs: Dict[str, Any] = None,
        log_every_n_steps: int = 50,
        rescale_above_n_tokens: int = 10,
    ):
        """Initializes class instance.

        Args:
            submodule: Name of the module (e.g. 'encoder', 'classifier.', etc.) inside which to search for matching
                layers. If none is provided, the Lightning module will be inspected starting from its root.
            reduction: Reduction to apply along the batch axis to pass from (N, S, S) attention weights to
                (S, S) attention weights, which can be logged as a heatmap. Available reduction methods:
                - ``'first'``: takes the first item in the batch (default)
                - ``'mean'``: takes the mean across the batch dimension
            compute_attention_rollout: Whether to also compute and log a global attention map using attention rollout.
            attention_rollout_kwargs: If `compute_attention_rollout` is True, parameters to forward to
                `didactic.models.explain.attention_rollout`.
            log_every_n_steps: Frequency at which to log the attention weights computed during the forward pass.
            rescale_above_n_tokens: For token sequences longer than this threshold, the size of the heatmap is
                scaled so that the tick labels and annotations become visibly smaller, instead of overlapping and
                becoming unreadable.
        """
        self.submodule_name = submodule
        self.reduction = reduction
        self.compute_attention_rollout = compute_attention_rollout
        self._attention_rollout_kwargs = attention_rollout_kwargs if attention_rollout_kwargs else {}
        self.log_every_n_steps = log_every_n_steps
        self.rescale_above_n_tokens = rescale_above_n_tokens
        self.train_step = 0

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Sets up hook to compute and store the attention weights during the forward pass when in training mode.

        Also extracts metadata about the tokens that will be useful to label attention weights' plots.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            stage: Current stage (e.g. fit, test, etc.) of the experiment.
        """
        # Extract the requested submodule from the root module
        module = pl_module
        if self.submodule_name:
            for submodule_name in self.submodule_name.split("."):
                module = getattr(module, submodule_name)

        # Set up the hooks inside the model to record the attention maps produced for each batch
        self._attn_weights = {}
        self._hook_handles = register_attn_weights_hook(module, self._attn_weights, reduction=self.reduction)

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Removes the hooks setup to capture the attention weights during the forward pass.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            stage: Current stage (e.g. fit, test, etc.) of the experiment.
        """
        for layer_hook in self._hook_handles.values():
            layer_hook.remove()

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: CardiacMultimodalRepresentationTask) -> None:
        """Computes attention weights based on the model's current weights, and logs the attention weights as heatmaps.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
        """
        if (self.train_step % self.log_every_n_steps) == 0:
            # Log attention weights for each layer w.r.t. input tokens
            for layer_name, attn_weights in self._attn_weights.items():
                attention_heads_fused = attn_weights.mean(dim=0).cpu().numpy()
                plot_heatmap(
                    pd.DataFrame(attention_heads_fused, index=pl_module.token_tags, columns=pl_module.token_tags),
                    rescale_above_n_elems=self.rescale_above_n_tokens,
                )
                log_figure(trainer.logger, figure_name=f"{layer_name}_attn_weight", step=self.train_step)
                plt.close()  # Close the figure to avoid contamination between plots

            # Compute and log attention rollout (using the attention weights at each layer)
            if self.compute_attention_rollout:
                attn_rollout_mask = (
                    attention_rollout(list(self._attn_weights.values()), **self._attention_rollout_kwargs).cpu().numpy()
                )

                if pl_module.hparams.cls_token:
                    # If we have the attention vector of the CLS token w.r.t. other tokens,
                    # reshape it into a matrix to be able to display it as a 2D heatmap
                    attn_rollout_df = pd.DataFrame(
                        attn_rollout_mask.reshape((1, -1)),
                        index=pl_module.token_tags[-1:],
                        columns=pl_module.token_tags[:-1],
                    )

                else:
                    # If we have the self-attention matrix, display it directly as a heatmap
                    attn_rollout_df = pd.DataFrame(
                        attn_rollout_mask, index=pl_module.token_tags, columns=pl_module.token_tags
                    )

                plot_heatmap(attn_rollout_df, rescale_above_n_elems=self.rescale_above_n_tokens)
                log_figure(trainer.logger, figure_name="attention_rollout", step=self.train_step)
                plt.close()  # Close the figure to avoid contamination between plots

        self.train_step += 1

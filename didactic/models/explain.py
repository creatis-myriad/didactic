import functools
import itertools
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torchmetrics.utilities.data import to_onehot
from vital.data.cardinal.config import TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum

from didactic.tasks.cardiac_multimodal_representation import CardiacMultimodalRepresentationTask


class SelfAttentionGenerator:
    """Computes attention w.r.t. input tokens for transformer models, using various attention attribution techniques."""

    def __init__(self, model: CardiacMultimodalRepresentationTask):
        """Initializes class instance.

        Args:
            model: Model for which to generate attention scores.
        """
        self.model = model.eval()

    def generate_relevancy(
        self,
        tabular_attrs: Dict[TabularAttribute, Tensor],
        time_series_attrs: Dict[Tuple[ViewEnum, TimeSeriesAttribute], Tensor],
        target_labels: Tensor,
        target_attr: TabularAttribute,
    ) -> Tensor:
        """Compute the relevancy formulation w.r.t. a target class, as proposed by Chefer et al.

        References:
            - Paper by Chefer et al. proposing the relevancy formulation: https://arxiv.org/abs/2103.15679

        Args:
            tabular_attrs: (K: S, V: N) Sequence of batches of tabular attributes. To indicate an item is missing an
                attribute, the flags `MISSING_NUM_ATTR`/`MISSING_CAT_ATTR` can be used for numerical and categorical
                attributes, respectively.
            time_series_attrs: (K: S, V: (N, ?)), Sequence of batches of time-series attributes, where the
                dimensionality of each attribute can vary.
            target_labels: (N), Target labels for the batch of samples.
            target_attr: Target attribute for which to generate the attention score.

        Returns:
            (N, S-1), Attention scores w.r.t. the target class for each token in the sequence, excluding the CLS token.
        """
        batch_size = len(target_labels)

        # Extract the model's predicted probabilities on the target class
        output = self.model(tabular_attrs, time_series_attrs, task="predict")[target_attr]
        target_labels = to_onehot(target_labels, num_classes=output.shape[-1])
        target_labels = target_labels.to(output.device, dtype=float).requires_grad_(True)
        base_output = torch.sum(output * target_labels)

        # Compute the gradients of the target class probabilities w.r.t. the model's parameters
        self.model.zero_grad()
        base_output.backward(retain_graph=True)

        # Initialize the relevancy score as the identity matrix
        blocks = self.model.encoder.blocks
        num_tokens = blocks[0].attention.attn.shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.attn.device)
        R = R.repeat(batch_size, 1, 1)

        # Update the relevancy score along the attention layers in the model
        for blk in blocks:
            grad = blk.attention.attn_grad  # (N * n_heads, S, S)
            cam = blk.attention.attn  # (N * n_heads, S, S)
            cam = grad * cam  # (N * n_heads, S, S)
            cam = cam.clamp(min=0)  # Remove negative attributions
            # Split the tensor to separate the batch and heads dimensions -> (N, n_heads, S, S)
            # then average across the heads dimension -> (N, S, S)
            cam = torch.stack(cam.split(batch_size, dim=0), dim=0).mean(dim=0)
            R += torch.matmul(cam, R)

        # Normalize the relevancy score per row, following equations 8 and 9 from the paper
        identity_rel = torch.eye(R.shape[-1], device=R.device)
        R_hat = R - identity_rel
        R = (R_hat / R_hat.sum(dim=-1, keepdim=True)) + identity_rel

        # Take the attention of the CLS token w.r.t. other tokens (2nd dim indexing),
        # dropping its attention on itself at the same time (3rd dim indexing)
        cls_per_token_score = R[:, -1, :-1].detach()
        return cls_per_token_score


def _patch_attn(attn_module: nn.MultiheadAttention) -> nn.MultiheadAttention:
    """Patches an attention layer to always compute attention weights, however it is called.

    References:
        - Inspired by this gist: https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91

    Args:
        attn_module: Attention layer to patch.

    Returns:
        Reference to the input attention layer, which has been patched.
    """
    attn_module_forward = attn_module.forward

    def force_attn_weights_wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return attn_module_forward(*args, **kwargs)

    attn_module.forward = force_attn_weights_wrap

    return attn_module


def register_attn_weights_hook(
    model: nn.Module,
    out: Dict[str, Tensor],
    average_attn_weights: bool = False,
    reduction: Optional[Literal["mean", "first"]] = None,
) -> Dict[str, torch.utils.hooks.RemovableHandle]:
    """Sets-up hooks to collect and save all attention maps produced by a model.

    While PyTorch's `MultiheadAttention` layer provides an API to get its attention maps, higher-level modules (e.g.
    `TransformerEncoderLayer`, `TransformerEncoder`, etc.) provide no means of accessing the attention maps. In these
    cases, this function becomes useful to hook into the model and record the attention maps.

    Args:
        model: Attention-based model for which to track attention maps.
        out: The dictionary in which to save the collected attention maps.
        average_attn_weights: If attention layers have multiple heads, indicates that the returned attention maps should
            be averaged across heads.
        reduction: Reduction to apply along the batch axis to pass from (N, S, S) attention weights to (S, S) attention
            weights, which can be logged as a heatmap. Available reduction methods:
                - ``None``: no reduction applied across the batch dimension (default)
                - ``'mean'``: takes the mean across the batch dimension
                - ``'first'``: takes the first item in the batch

    Returns:
        Handles on hooks registered for each layer, to be able to remove them later.
    """

    def _save_attn_weights_hook(attn_module: nn.Module, input: Tensor, output: Tensor, layer_name: str) -> None:
        attn_weights = output[1].detach()
        is_input_batched = attn_weights.ndim == 4

        if average_attn_weights:
            heads_dim = 1 if is_input_batched else 0
            attn_weights = attn_weights.mean(dim=heads_dim)

        if is_input_batched and reduction:
            # Reduce the batch dimension of the attention weights, i.e. from shape (N, [H,] S, S) to ([H,] S, S)
            match reduction:
                case "mean":
                    attn_weights = attn_weights.mean(dim=0)
                case "first":
                    attn_weights = attn_weights[0]
                case _:
                    raise ValueError(
                        f"Unexpected value for 'reduction': {reduction}. Use one of: [None, 'mean', 'first']."
                    )

        out[layer_name] = attn_weights

    attn_layers = {
        f"attn_layer_{layer_idx}": _patch_attn(layer)
        for layer_idx, layer in enumerate(
            module for module in model.modules() if isinstance(module, nn.MultiheadAttention)
        )
    }
    handles = {
        layer_name: attn_layer.register_forward_hook(functools.partial(_save_attn_weights_hook, layer_name=layer_name))
        for layer_name, attn_layer in attn_layers.items()
    }
    return handles


def attention_rollout(
    attentions: Sequence[Tensor],
    gradients: Sequence[Tensor] = None,
    head_reduction: Literal["mean", "max", "min"] = "max",
    includes_cls_token: bool = False,
) -> Tensor:
    """Computes global attention maps for transformer models using the (gradient) attention rollout technique.

    References:
        - Inspired by the following Attention Rollout and Gradient Attention Rollout implementations of explainability
          methods for Vision Transformers:
          - Attention Rollout: https://github.com/jacobgil/vit-explain/blob/15a81d355a5aa6128ea4e71bbd56c28888d0f33b/vit_rollout.py#L9-L42
          - Gradient Attention Rollout: https://github.com/jacobgil/vit-explain/blob/15a81d355a5aa6128ea4e71bbd56c28888d0f33b/vit_grad_rollout.py#L9-L36

    Args:
        attentions: Mx([N,] H, S, S), Attentions matrices of the successive attention layers in the model.
        gradients: If provided, will be used to compute "Gradient Attention Rollout" by weighing attention. Their shape
            should match that of `attentions`.
        head_reduction: Reduction to apply across each layer's multiple attention heads.
        includes_cls_token: Whether the attention model used a class token (which should not be discarded, regardless
            of its attention value and of the discard ratio).

    Returns:
        ([N,] S) if `includes_cls_token`, ([N,] S, S) otherwise. Attention map values, computed using attention rollout.
    """
    # Add a batch dimension to the attentions, if they are not batched
    if is_item := attentions[0].ndim == 3:
        attentions = [layer_attention.unsqueeze(dim=0) for layer_attention in attentions]

    attention_mask = torch.eye(attentions[0].shape[-1], device=attentions[0].device)

    if gradients is None:
        gradients = []
    gradients = [
        layer_gradient.reshape(layer_attention.shape) for layer_gradient, layer_attention in zip(gradients, attentions)
    ]

    with torch.inference_mode():
        for attention, grad in itertools.zip_longest(attentions, gradients):
            if grad is not None:
                attention *= grad
            match head_reduction:
                case "mean":
                    attention_heads_fused = attention.mean(dim=1)
                case "max":
                    attention_heads_fused = attention.max(dim=1)[0]
                case "min":
                    attention_heads_fused = attention.min(dim=1)[0]
                case _:
                    raise ValueError(
                        f"Unexpected value for 'head_reduction': {head_reduction}. Use one of: ['mean', 'max', 'min']."
                    )

            # To account for residual connections, add identity to the attention and re-normalize the weights
            identity_matrix = torch.eye(attention_heads_fused.shape[-1], device=attentions[0].device)
            cur_layer_attention = attention_heads_fused + identity_matrix
            cur_layer_attention = cur_layer_attention / cur_layer_attention.sum(dim=-1)

            # Recursively multiply the attention matrices
            attention_mask = torch.matmul(cur_layer_attention, attention_mask)  # (N, S, S)

    if includes_cls_token:
        # Look only at the attention between the class token and the other tokens
        attention_mask = attention_mask[:, -1, :-1]  # (N, S, S) -> (N, S-1)

    # Normalize the computed attention mask so that the total attention given by a token (i.e. each row) sums to 1
    attention_mask = attention_mask / attention_mask.sum(dim=-1)

    if is_item:
        attention_mask = attention_mask.squeeze(dim=0)
    return attention_mask


def k_number(attention: Tensor, row_reduction: Literal["median", "mean", "min", "max"] = "median") -> float:
    """Computes an attention map's k-number, an indicator of the spread of its attention.

    References:
        - Paper proposing k-number to discriminate attention maps' importance: https://arxiv.org/abs/2104.00926

    Args:
        attention: (S, S), Attention map for which to compute the k-number.
        row_reduction: Reduction to apply across the rows to obtain only one k-number for the whole attention map.

    Returns:
        The attention map's k-number.
    """
    sorted_attn, _ = torch.sort(attention, 1, descending=True)
    cum_attn = sorted_attn.cumsum(dim=1)
    first_past_90_idx = (cum_attn < 0.9).sum(dim=1) + 1  # Add 1 to include the token to reach > 0.9 in the count
    k_numbers_by_row = first_past_90_idx / len(attention)  # Normalize the k-number w.r.t. the number of tokens

    match row_reduction:
        case "median":
            k_number = k_numbers_by_row.median()
        case "mean":
            k_number = k_numbers_by_row.mean()
        case "max":
            k_number = k_numbers_by_row.max()[0]
        case "min":
            k_number = k_numbers_by_row.min()[0]
        case _:
            raise ValueError(
                f"Unexpected value for 'row_reduction': {row_reduction}. Use one of: ['median', 'mean', 'max', 'min']."
            )

    return k_number.item()

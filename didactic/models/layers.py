import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""

    def __init__(self, sequence_len: int, d_model: int):
        """Initializes layers parameters.

        Args:
            sequence_len: The number of tokens in the input sequence.
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        self.positional_encoding = Parameter(torch.empty(sequence_len, d_model))
        init.trunc_normal_(self.positional_encoding, std=0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that adds positional encoding to the input tensor.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, S, `d_model`), Tensor with added positional encoding.
        """
        return x + self.positional_encoding[None, ...]


class SequentialPooling(nn.Module):
    """Sequential pooling layer."""

    def __init__(self, d_model: int):
        """Initializes layer submodules.

        Args:
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        # Initialize the learnable parameters of the sequential pooling
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that performs a (learnable) weighted averaging of the different tokens.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, `d_model`), Output tensor.
        """
        attn_vector = F.softmax(self.attention_pool(x), dim=1)  # (N, S, 1)
        broadcast_attn_vector = attn_vector.transpose(2, 1)  # (N, S, 1) -> (N, 1, S)
        pooled_x = (broadcast_attn_vector @ x).squeeze(1)  # (N, 1, S) @ (N, S, E) -> (N, E)
        return pooled_x

from typing import Literal, Tuple

import torch
from scipy.special import binom, factorial
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


class UnimodalLogitsHead(nn.Module):
    """Layer to output (enforced) unimodal logits from an input feature vector.

    This is a re-implementation of a 2017 ICML paper by Beckham and Pal, which proposes to use either a Poisson or
    binomial distribution to output unimodal logits (because they are constrained as such by the distribution) from a
    scalar value.

    References:
        - ICML 2017 paper: https://proceedings.mlr.press/v70/beckham17a.html
    """

    def __init__(
        self,
        in_features: int,
        num_logits: int,
        distribution: Literal["poisson", "binomial"] = "binomial",
        tau: float = 1.0,
        tau_mode: Literal["fixed", "learn", "learn_sigm", "learn_fn"] = "learn_fn",
        eps: float = 1e-6,
    ):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            num_logits: Number of logits to output.
            distribution: Distribution whose probability mass function (PMF) is used to enforce an unimodal distribution
                of the logits.
            tau: Temperature parameter to control the sharpness of the distribution.
                - If `tau_mode` is 'fixed', this is the fixed value of tau.
                - If `tau_mode` is 'learn' or 'learn_sigm', this is the initial value of tau.
                - If `tau_mode` is 'learn_fn', this argument is ignored.
            tau_mode: Method to use to set or learn the temperature parameter:
                - 'fixed': Use a fixed value of tau.
                - 'learn': Learn tau.
                - 'learn_sigm': Learn tau through a sigmoid function.
                - 'learn_fn': Learn tau through a function of the input, i.e. a tau that varies for each input.
                  The function is 1 / (1 + g(L(x))), where g is the softplus function. and L is a linear layer.
            eps: Epsilon value to use in probabilities' log to avoid numerical instability.
        """
        super().__init__()
        self.num_logits = num_logits
        self.distribution = distribution
        self.tau_mode = tau_mode
        self.eps = eps

        self.register_buffer("logits_idx", torch.arange(self.num_logits))
        match self.distribution:
            case "poisson":
                self.register_buffer("logits_factorial", torch.from_numpy(factorial(self.logits_idx)))
            case "binomial":
                self.register_buffer("binom_coef", binom(self.num_logits - 1, self.logits_idx))
            case _:
                raise ValueError(f"Unsupported distribution '{distribution}'.")

        self.param_head = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

        match self.tau_mode:
            case "fixed":
                self.tau = tau
            case "learn" | "learn_sigm":
                self.tau = nn.Parameter(torch.tensor(float(tau)))
            case "learn_fn":
                self.tau_head = nn.Sequential(nn.Linear(in_features, 1), nn.Softplus())
            case _:
                raise ValueError(f"Unsupported tau mode '{tau_mode}'.")

    def __repr__(self):
        """Overrides the default repr to display the important parameters of the layer."""
        vars = {"in_features": self.param_head[0].in_features}
        vars.update({var: getattr(self, var) for var in ["num_logits", "distribution", "tau_mode"]})
        vars_str = [f"{var}={val}" for var, val in vars.items()]
        return f"{self.__class__.__name__}({', '.join(vars_str)})"

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predicts unnormalized, unimodal logits from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `num_logits`), Output tensor of unimodal logits. The logits are unnormalized, but the temperature
              to control the sharpness of the distribution as already been applied.
            - (N, 1), Predicted parameter of the distribution, in the range [0, 1].
            - (N, 1), Temperature parameter tau, in the range [0, inf) or [0, 1], depending on `tau_mode`.
        """
        # Forward through the linear layer to get a scalar param in [0,1] for the distribution
        f_x = param = self.param_head(x)

        # Compute the probability mass function (PMF) of the distribution
        # For technical reasons, use the log instead of the direct value
        match self.distribution:
            case "poisson":
                f_x = (self.num_logits + 1) * f_x  # Rescale f(x) to [0, num_logits+1]
                log_f = (self.logits_idx * torch.log(f_x + self.eps)) - f_x - torch.log(self.logits_factorial)
            case "binomial":
                log_f = (
                    torch.log(self.binom_coef)
                    + (self.logits_idx * torch.log(f_x + self.eps))
                    + ((self.num_logits - 1 - self.logits_idx) * torch.log(1 - f_x + self.eps))
                )

        # Compute the temperature parameter tau
        # In cases where tau is a scalar, manually broadcast it to a tensor w/ one value for each item in the batch
        # This is done to keep a consistent API for the different tau modes, with tau having a different value for each
        # item in the batch when `tau_mode` is 'learn_fn'
        match self.tau_mode:
            case "fixed":
                tau = torch.full_like(param, self.tau)  # Manual broadcast
            case "learn":
                tau = self.tau.expand_as(param)  # Manual broadcast
            case "learn_sigm":
                tau = torch.sigmoid(self.tau).expand_as(param)  # Sigmoid + manual broadcast
            case "learn_fn":
                tau = 1 / (1 + self.tau_head(x))
            case _:
                raise ValueError(f"Unsupported 'tau_mode': '{self.tau_mode}'.")

        return log_f / tau, param, tau

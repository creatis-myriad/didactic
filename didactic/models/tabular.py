from typing import List, Optional

import torch
from rtdl_revisiting_models import CategoricalEmbeddings, LinearEmbeddings
from torch import Tensor, nn


def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


class TabularEmbedding(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from v0.0.2 of the `rtdl_revisiting_models` package, instead of
          the original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        .. testcode::

            n_objects = 4
            n_num_features = 3
            n_cat_features = 2
            d_token = 7
            x_num = torch.randn(n_objects, n_num_features)
            x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
            # [2, 3] reflects cardinalities
            tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
            tokens = tokenizer(x_num, x_cat)
            assert tokens.shape == (n_objects, n_num_features + n_cat_features, d_token)
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int,
    ) -> None:
        """Initializes class instance.

        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there are no numerical features.
            cat_cardinalities: the number of unique values for each feature. Pass an empty list if there are no
                categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert (
            n_num_features or cat_cardinalities
        ), "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = CategoricalEmbeddings(cat_cardinalities, d_token) if cat_cardinalities else None

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """Perform the forward pass.

        Args:
            x_num: continuous features. Must be presented if :code:`n_num_features > 0` was passed to the constructor.
            x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed to the
                constructor.

        Returns:
            tokens

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """

        assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)

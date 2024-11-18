import numpy as np
import torch.nn as nn

from typing import Optional

from math import (
    pi,
    sqrt,
)

from torch import (
    cat as torch_cat,
    Tensor,
    LongTensor,
    tanh as torch_tanh,
    pow as torch_pow
)

from tools.utils import fix_random_seed

from ..layers import TokenEmbedding


__all__ = (
    'AdvancedItemEncoder',
)

class GELU(nn.Module):

    def forward(self, x: Tensor):
        return 0.5 * x * (1 + torch_tanh(sqrt(2 / pi) * (x + 0.044715 * torch_pow(x, 3))))

class AdvancedItemEncoder(nn.Module):
    #modification : added dataset name argument
    def __init__(self,
                 num_items: int,
                 ifeatures: np.ndarray,
                 ifeature_dim: int,
                 icontext_dim: int,
                 hidden_dim: int = 64,
                 num_known_item: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 name: str = None
                 ):
        super().__init__()

        # data params
        self.num_items = num_items
        self.ifeature_dim = ifeature_dim
        self.icontext_dim = icontext_dim

        # main params
        self.hidden_dim = hidden_dim
        self.num_known_item = num_known_item
        ###
        self.name = name

        # optional params
        self.random_seed = random_seed

        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        # V+1 ~: unknown token (convert to V+1 after mask)
        if num_known_item is None:
            self.vocab_size = num_items + 1
        else:
            self.vocab_size = num_known_item + 2

        # ifeature cache
        self.ifeature_cache = nn.Embedding.from_pretrained(Tensor(ifeatures), freeze=True)

        # embedding layers
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=hidden_dim
        )

        # main layers
        self.ac_encoder = nn.Linear(ifeature_dim + icontext_dim, hidden_dim * 4)
        self.item_encoder = nn.Linear(hidden_dim * 4, hidden_dim)

        # Gelu activation
        self.gelu = GELU()

    def forward(self,
                tokens: LongTensor,  # (b x L|C)
                icontexts: Tensor,  # (b x L|C x d_Ci)
                ):

        # get ifeatures from cache
        ifeatures = self.ifeature_cache(tokens)

        # get ac vector
        #modifications:
        if self.name == 'vwfs':
            ac = ifeatures
        else:
            ac = torch_cat([ifeatures, icontexts], dim=-1)
        ac_vector = self.ac_encoder(ac)
        vector = self.gelu(ac_vector)
        vector = self.item_encoder(vector)

        return vector

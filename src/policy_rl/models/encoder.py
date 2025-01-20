from typing import Tuple, Union

import torch.nn as nn

from torch import Tensor

from .nn.env_embeddings import env_init_embedding
from .nn.graph.attnnet import GraphAttentionNetwork


class GraphAttentionEncoder(nn.Module):
    """Graph Attention Encoder as in Kool et al. (2019).

    Args:
        env_name: environment name to solve
        num_heads: Number of heads for the attention
        embedding_dim: Dimension of the embeddings
        num_layers: Number of layers for the encoder
        normalization: Normalization to use for the attention
        feed_forward_hidden: Hidden dimension for the feed-forward network
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        sdpa_fn: Scaled dot product function to use for the attention
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        num_layers: int,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        init_embedding: nn.Module = None,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # TODO
        self.init_name = "common"
        self.init_embedding = (
            env_init_embedding(self.init_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        self.net = GraphAttentionNetwork(
            num_heads,
            embedding_dim,
            num_layers,
            normalization,
            feed_forward_hidden,
            sdpa_fn=None,    # TODO？
        )

    def forward(        # TODO
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)        # 

        # Process embedding
        h = self.net(init_h, mask)

        # Return latent representation and initial embedding
        return h, init_h

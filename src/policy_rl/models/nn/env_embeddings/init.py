import torch
import torch.nn as nn


def env_init_embedding(init_name: str, config: dict) -> nn.Module:
    """Get environment initial embedding. The init embedding is used to initialize the
    general embedding of the problem nodes without any solution information.
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "common": InitEmbedding,
    }

    if init_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{init_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[init_name](**config)

class InitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embedding_dim, linear_bias=True):
        super(InitEmbedding, self).__init__()
        input_dim = 2  # x, y
        self.init_embed = nn.Linear(input_dim, embedding_dim, linear_bias)

    def forward(self, td):  # TODO
        out = self.init_embed(td["locs"])
        return out
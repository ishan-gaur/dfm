import torch
from torch import nn
from typing import Callable


class PredictiveModel(nn.Module):
    pass


class LinearProbe(nn.Module):
    """
    Linear probe on top of pre-computed embeddings.

    Tensor Dimension Labels:
        I: batch index
        D: embedding dimension
        O: output dimension
    """

    def __init__(
        self,
        embed_fn: Callable,
        embedding_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.embed_fn = embed_fn
        self.w = nn.Linear(self.embedding_dim, self.output_dim)

    def forward(self, x_SP: torch.LongTensor):
        x_ID = self.embed_fn(x_SP)
        y_IO = self.w(x_ID)
        return y_IO


class OneHotMLP(nn.Module):
    """
    MLP operating on one-hot encoded sequences.

    Uses a frozen identity embedding to convert token indices to one-hot vectors,
    flattens across all positions, then passes through an MLP.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        T: token dimension (one-hot)
        O: output dimension
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        model_dim: int,
        n_layers: int,
        output_dim: int,
        padding_idx: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.padding_idx = padding_idx

        # Frozen one-hot embedding: each token maps to its one-hot vector
        self.embed = nn.Embedding(
            self.vocab_size,
            self.vocab_size,
            self.padding_idx,
            _weight=torch.eye(self.vocab_size),
            _freeze=True,
        )

        layers: list[nn.Module] = [
            nn.Linear(self.sequence_length * self.vocab_size, self.model_dim)
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.model_dim, self.model_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.model_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x_SP: torch.LongTensor):
        x_SPT = self.embed(x_SP)
        x_SPxT = x_SPT.reshape(x_SPT.size(0), -1)
        y_hat_SO = self.layers(x_SPxT)
        return y_hat_SO

import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from abc import ABC, abstractmethod
from contextlib import contextmanager


class PredictiveModel(nn.Module, ABC):
    """
    Note, super important to not forget to set self.input_dim so the seq->ohe conversion works as expected
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_dim

    @abstractmethod
    def forward(
        self, ohe_seq_SPT: torch.FloatTensor
    ):  # note this is a float as we might want to take a gradient on it
        pass

    @abstractmethod
    def with_target(self, target):
        # sets a specification of the target so that the raw forward regression
        # value or class logits turn into a cdf or class probability
        pass

    @abstractmethod
    def target_log_probs_given_seq(ohe_seq_SPT: torch.FloatTensor):
        pass

    def target_log_probs_given_seq(self, seq_SP):
        ohe_seq_SPT = F.one_hot(seq_SP, self.input_dim)
        return self.target_log_probs_given_ohe(ohe_seq_SPT)

    @abstractmethod
    def target_log_probs_given_seq(self, seq_SP):
        # basically applies the model to get the right probability using the specified target
        pass

    @abstractmethod
    def target_log_probs_given_ohe(self, ohe_seq_SPT: torch.FloatTensor):
        pass


class RealValuedPredictiveModel(PredictiveModel, ABC):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.threshold = None

    @contextmanager
    def with_target(self, threshold: Callable[torch.Tensor, bool]):
        pre_context_target = self.threshold
        self.threshold = threshold
        try:
            yield self
        finally:
            self.threshold = pre_context_target

    @abstractmethod
    def compute_log_cdf(
        self, prediction: torch.Tensor
    ) -> float:  # Prediction could be ensemble of values, mean/variance, monte-carlo samples, even the CDF
        pass

    def target_log_probs_given_ohe(self, ohe_seq_SP):
        prediction = self.forward(ohe_seq_SPT)
        logp_y_g_x_S = self.compute_log_cdf(prediction)
        return logp_y_g_x_S


class EnsemblePredictiveModel(RealValuedPredictiveModel):
    pass


class ClassValuedPredictiveModel(PredictiveModel, ABC):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.class = None
        self.temp = 1.0

    @contextmanager
    def with_target(self, class: int):
        pre_context_class = self.class
        self.class = class
        try:
            yield self
        finally:
            self.class = pre_context_target

    # TODO[pi] make this a mixin
    @contextmanager
    def with_temp(self, temp: float) -> TransitionModel:
        """
        The ``with_temp`` method modifies the state of the instantiated
        objecte so that the transition_log_probs function returned by the class is
        computed from the logits using the specified temperature.
        """
        pre_context_temp = self.temp
        self.temp = temp
        try:
            yield self
        finally:
            self.temp = pre_context_temp

    def target_log_probs_given_ohe(self, ohe_seq_SP):
        logits_y_g_x_SC= self.forward(ohe_seq_SPT)
        logp_y_g_x_S = F.log_softmax(logits_y_g_x_SC / temp)[:, self.class]
        return logp_y_g_x_S


# ==========================================================================================
# ==========================================================================================
# The following classes are templates to train your own predictive models with.
# When creating your own predictive models, make sure that they inherit from PredictiveModel
# and wrap around one of the template models below. See the dfm/models/ folder for examples.
# ==========================================================================================
# ==========================================================================================


class PreTrainedEmbeddingModel(nn.Module, ABC):
    EMB_DIM = None
    @abstractmethod
    def forward_ohe(self, ohe_seq_SPT: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor):
        # returns both the log probs and embeddings
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
        embed_model: PreTrainedEmbeddingModel,
        output_dim: int,
    ):
        super().__init__()
        self.embed_model = embed_model
        self.embedding_dim = embed_model.EMB_DIM
        self.output_dim = output_dim
        self.w = nn.Linear(self.embedding_dim, self.output_dim)

    def forward(self, ohe_x_SPT: torch.LongTensor):
        x_ID = self.embed_model(ohe_x_SPT)
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
        x_SPT = F.one_hot(x_SP, num_classes=self.vocab_size)
        x_SPxT = x_SPT.reshape(x_SPT.size(0), -1)
        y_hat_SO = self.layers(x_SPxT)
        return y_hat_SO

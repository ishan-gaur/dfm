import torch
from dfm.generative_modeling import (
    TransitionModel,
    ConditionalTransitionModel,
    MaskedModelLogitFormatter,
    LogitFormatter,
)
from dfm.predictive_modeling import PreTrainedEmbeddingModel
from transformers import PreTrainedTokenizerBase
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from esm.utils.residue_constants import atom_order
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

OUTPUT_DIM = 64
TOKENIZER = EsmSequenceTokenizer()
MASKED_FORMATTER = MaskedModelLogitFormatter(TOKENIZER, "<mask>", OUTPUT_DIM)


class ESMC(TransitionModel):
    """
    Tensor Index Legend
    S: sequence index in batch
    P: position index in sequence
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase = TOKENIZER,
        logit_formatter: LogitFormatter = MASKED_FORMATTER,
        esmc_checkpoint: str = "esmc_300m",
    ):
        # init base model and logit formatter
        # the forward calls and runs formatter
        super().__init__(tokenizer, logit_formatter)
        self.model = ESMC.from_pretrained(esmc_checkpoint)
        self.model.eval()

    def forward(self, seq_SP: torch.LongTensor):
        logits_SPT = self.model(seq_SP).sequence_logits.float()
        assert logits_SPT.shape[2] == OUTPUT_DIM, "OUTPUT_DIM constant is wrong"
        return logits_SPT


class ESM3IF(ConditionalTransitionModel):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase = TOKENIZER,
        logit_formatter: LogitFormatter = MASKED_FORMATTER,
        esm3_checkpoint: str = "esm3-open",
    ):
        # init base model and logit formatter
        # the forward calls and runs formatter
        super().__init__(tokenizer, logit_formatter)
        self.model = ESM3.from_pretrained(esm3_checkpoint)
        self.model.eval()

    def set_condition(self, observations):
        raise NotImplementedError()

    def forward(self, seq_SP: torch.LongTensor):
        raise NotImplementedError()

    # TODO[pi] add embedding

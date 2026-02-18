import torch
from torch.nn import functional as F
from dfm.generative_model import (
    TransitionModel,
    MaskedModelLogitFomatter,
    LogitFormatter,
)
from transformers import PreTrainedTokenizerBase
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

OUTPUT_DIM = 64
TOKENIZER = EsmSequenceTokenizer()
MASKED_FORMATTER = MaskedModelLogitFomatter(TOKENIZER, "<mask>", OUTPUT_DIM)


class ESM(TransitionModel):
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

    def forward(self, seq_SP: torch.LongTensor):
        logits_SPT = self.model(seq_SP).sequence_logits.float()
        assert logits_SPT.shape[2] == OUTPUT_DIM, "OUTPUT_DIM constant is wrong"
        return logits_SPT

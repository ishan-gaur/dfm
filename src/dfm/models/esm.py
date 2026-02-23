import torch
from torch import nn
from dfm.generative_modeling import (
    TransitionModel,
    MaskedModelLogitFormatter,
    LogitFormatter,
)
from dfm.predictive_modeling import PreTrainedEmbeddingModel
from transformers import PreTrainedTokenizerBase
from esm.models.esmc import ESMC as _ESMC
from esm.models.esmc import ESMCOutput
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ESMC(TransitionModel):
    """ESM-C masked language model wrapped as a TransitionModel.

    Tensor Index Legend:
        S: sequence index in batch
        P: position index in sequence
        T: token/vocab dimension
    """

    OUTPUT_DIM = 64

    def __init__(self, esmc_checkpoint: str = "esmc_300m"):
        tokenizer = EsmSequenceTokenizer()
        logit_formatter = MaskedModelLogitFormatter(tokenizer, ESMC.OUTPUT_DIM)
        esmc = _ESMC.from_pretrained(esmc_checkpoint).eval()
        super().__init__(
            model=esmc, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def format_raw_to_logits(
        self, model_output: ESMCOutput, seq_SP: torch.LongTensor
    ) -> torch.FloatTensor:
        logits_SPT = model_output.sequence_logits.float()
        logits_SPT = self.logit_formatter(logits_SPT, seq_SP)
        return logits_SPT


# TODO[pi] implement ESM3IF as a structure-conditioned TransitionModel

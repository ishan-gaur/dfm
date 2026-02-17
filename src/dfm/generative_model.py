"""Generative model utilities.

Currently provides the MPNNTokenizer for converting amino acid sequences
to ProteinMPNN token indices.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Protocol, runtime_checkable, Optional
from transformers import PreTrainedTokenizerBase


# TODO[pi] add a DFM Protocol and Base Class which defines a score method which returns logits for each positions and makes sure to do things like force register token probs to themselves and regular tokens to only be regular tokens, etc. Actually give it a fix register tokens method and allow child classes to specify those token ids and output indices in the init to make it easy. Maybe the model output formatter is a general thing that's defined ... -- like what's the best way to spec the output_dim of ESM models??


@runtime_checkable
class LogitFormatter(Protocol, nn.Module):
    # TODO[pi] let's discuss a protocol or ABC design based on the example class I added below
    pass


class MaskedModelLogitFomatter(LogitFormatter):
    # mask maps to non-special tokens
    # all other tokens map to themselves
    # this class assumes the output logits
    # this class assumes that input and output token positions are the same
    # output dim is taken as a param in cases the output dim is changed for memory fragmentation reasons
    # TODO[pi] let's discuss a good way to make the valid outputs matrix automatically go to the model device when the model is moved. As you can see in esm_cath/model.py:ESM we will instantiate this in the model class.
    # TODO[pi] add a proper explanation of the design of the class and its forward method when we're done
    # TODO I actually only wrote this for EsmSequenceTokenizer--should wrote tests with this an some other random tokenizers, like BERT or LLaDA
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_token: str,
        output_dim: Optional[int] = None,
    ):

        super(
            nn.Module
        ).__init__()  # will this move the matrix below to the right device

        self.tokenizer = tokenizer
        self.output_dim = (
            self.tokenizer.vocab_size if output_dim is None else output_dim
        )
        # TODO[pi] do I need to somehow indicate this matrix is a module parameter so it moves device accordingly, also need to make sure it doesn't take a gradient
        self.valid_outputs_TiTo = torch.ones(
            self.tokenizer.vocab_size, self.output_dim
        ) * float("-inf")
        # All tokens map to themselves
        for idx in self.tokenizer.vocab.values():
            self.valid_outputs_TiTo[idx, idx] = 1.0
        mask_token_idx = self.tokenizer.vocab[mask_token]

        # Except for mask which maps to any non-special token
        # TODO add support for mask which can go to a special token??
        # TODO[pi] Ti is tokens in, To is tokens out, can you add docs for the tensor indices?
        self.valid_outputs_TiTo[mask_token_idx, mask_token_idx] = float("-inf")
        valid_mask_outputs = set(range(output_dim))
        for idx in self.tokenizer.added_tokens_decoder.keys():
            valid_mask_outputs.remove(idx)
        for idx in valid_mask_outputs:
            self.valid_outputs_TiTo[mask_token_idx, idx] = 1.0

    def forward(self, logits_SPT: torch.FloatTensor, seq_SP: torch.LongTensor):
        ohe_seq_SPTi = F.one_hot(seq_SP, self.tokenizer.vocab_size)
        output_mask_SPTo = ohe_seq_SPTi @ self.valid_outputs_TiTo
        return logits_SPT * output_mask_SPTo


class MPNNTokenizer:
    """Tokenizer using ProteinMPNN's amino acid vocabulary.

    Maps single-letter amino acid sequences to/from PMPNN token indices.
    Vocabulary: 20 standard amino acids + UNK (X), indexed 0-20.

    Follows HuggingFace tokenizer conventions:
        - encode(sequence) -> list[int]
        - decode(token_ids) -> str
        - __call__(sequences) -> dict with 'input_ids' tensor
        - vocab_size property
    """

    def __init__(self):
        from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA
        from mpnn.transforms.feature_aggregation.token_encodings import (
            MPNN_TOKEN_ENCODING,
        )

        three_to_idx = MPNN_TOKEN_ENCODING.token_to_idx

        # Build one-letter <-> index mappings
        self._one_to_idx: dict[str, int] = {}
        self._idx_to_one: dict[int, str] = {}
        for three_letter, idx in three_to_idx.items():
            one_letter = DICT_THREE_TO_ONE.get(
                str(three_letter), DICT_THREE_TO_ONE[UNKNOWN_AA]
            )
            self._one_to_idx[one_letter] = int(idx)
            self._idx_to_one[int(idx)] = one_letter

        self.unk_token = "X"
        self.unk_token_id = self._one_to_idx[self.unk_token]
        self._vocab_size = len(three_to_idx)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, sequence: str) -> list[int]:
        """Convert a single-letter AA sequence to token indices."""
        return [self._one_to_idx.get(aa, self.unk_token_id) for aa in sequence]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Convert token indices back to a single-letter AA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self._idx_to_one.get(idx, self.unk_token) for idx in token_ids)

    def __call__(self, sequences: str | list[str]) -> dict[str, torch.Tensor]:
        """Tokenize one or more sequences, returning a dict with 'input_ids' tensor."""
        if isinstance(sequences, str):
            sequences = [sequences]
        input_ids = torch.tensor(
            [self.encode(seq) for seq in sequences], dtype=torch.long
        )
        return {"input_ids": input_ids}

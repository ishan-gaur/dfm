"""Generative model utilities.

Currently provides the MPNNTokenizer for converting amino acid sequences
to ProteinMPNN token indices.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase
from typing import Protocol, runtime_checkable, Optional
from abc import ABC, abstractmethod


# TODO[pi] add a DFM Protocol and Base Class which defines a score method which returns logits for each positions and makes sure to do things like force register token probs to themselves and regular tokens to only be regular tokens, etc. Actually give it a fix register tokens method and allow child classes to specify those token ids and output indices in the init to make it easy. Maybe the model output formatter is a general thing that's defined ... -- like what's the best way to spec the output_dim of ESM models??


class TransitionModel(nn.Module, ABC):
    """Standard interface for wrapping models for DFM sampling and guidance.

    Child classes must implement ``forward`` and pass a tokenizer and logit
    formatter to ``super().__init__``.  For most use-cases
    ``PassThroughLogitFormatter`` or ``MaskedModelLogitFomatter`` will suffice.

    Example (mirrors the ESM wrapper in esm-cath)::

        class ESM(TransitionModel):
            OUTPUT_DIM = 64

            def __init__(self, checkpoint="esmc_300m"):
                tokenizer = EsmSequenceTokenizer()
                format_logits = MaskedModelLogitFomatter(tokenizer, "<mask>", self.OUTPUT_DIM)
                super().__init__(tokenizer=tokenizer, format_logits=format_logits)
                self.model = ESMC.from_pretrained(checkpoint)

            def forward(self, seq_SP):
                logits = self.model(seq_SP).sequence_logits.float()
                logits = self.format_logits(logits, seq_SP)
                return F.log_softmax(logits, dim=-1)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, format_logits: LogitFormatter):
        super().__init__()
        self.tokenizer = tokenizer
        self.format_logits = format_logits

    @abstractmethod
    def forward(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Return logits with ``format_logits`` already applied."""
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_from_string(self, sequences: list[str]):
        # TODO[pi] can I just get the encoded, padded sequences from the tokenizer directly?
        ids_list = [self.tokenizer.encode(s) for s in sequences]
        max_len = max(len(x) for x in ids_list)
        padded = []
        for x in ids_list:
            x_list = x if isinstance(x, list) else x.tolist()
            padded.append(
                x_list + [self.tokenizer.pad_token_id] * (max_len - len(x_list))
            )
        seq_SP = torch.tensor(padded, device=self.device, dtype=torch.long)
        return self.forward(seq_SP)

    def get_transition_log_probs(
        self,
    ):  # TODO[pi] not quite sure about this part and how it will play with the conditional model, maybe we should discuss
        def forward_unconditional(seq_SP: torch.LongTensor):
            logits_SPT = self.forward(seq_SP)
            return self.format_logits(logits_SPT, seq_SP)


@runtime_checkable
class LogitFormatter(Protocol):
    """Constrains model output logits based on input token identities.

    Applied before log_softmax to enforce valid output distributions
    per input token (e.g. special tokens predict themselves, mask tokens
    predict only non-special tokens).

    Implementations intended for use as model submodules should inherit
    from nn.Module and use register_buffer for device tracking. When
    inheriting from both nn.Module and LogitFormatter, nn.Module must
    come first in the MRO (e.g. ``class Foo(nn.Module, LogitFormatter)``)
    so that nn.Module.__call__ (which dispatches to forward) is resolved
    before Protocol.__call__.

    Must return a FloatTensor so that the softmax doesn't have normalization
    issues due to a lack of precision.

    Design note — implementation approaches:
        The reference implementation (MaskedModelLogitFomatter) uses a precomputed
        dense additive mask matrix indexed by input token ids. This is the right
        tradeoff for typical protein/NLP vocabularies (33–30k tokens): the matrix
        is built once at init and reused every forward pass, fully vectorized with
        no branching. Alternative approaches include:

        - **In-place scatter**: no precomputed matrix; loop over positions at forward
          time and write -inf into invalid outputs. Simple but slow.
        - **Boolean mask + masked_fill**: store a boolean matrix (1 bit vs 32 bits
          per entry), index it the same way, then ``logits.masked_fill(~mask, -inf)``.
          Saves memory at the cost of an extra op.
        - **Sparse allowlist**: store a dict mapping each token id to a LongTensor
          of valid output indices. More natural for huge vocabularies where the
          valid set per token is tiny.
        - **Categorical branching**: classify each input token as mask/special/regular
          and apply a different rule per type. No matrix, but introduces branching.
        - **Post-softmax renormalization**: run softmax normally, zero out invalid
          probs, renormalize. Changes the gradient landscape vs. additive masking.
        - **Loss-side only**: don't constrain logits at all; mask the loss instead
          and trust the model learns the constraints. No guarantees at inference.
    """

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor: ...


class PassThroughLogitFormatter(LogitFormatter):
    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        return logits.float()


class MaskedModelLogitFomatter(nn.Module, LogitFormatter):
    """Enforces output constraints for masked language models via additive masking.

    Builds a static mask matrix of shape (Ti, To) that defines which output tokens
    are valid for each input token. In forward, input token ids directly index into
    this matrix to select the per-position mask, which is then added to the raw
    logits before log_softmax.

    Constraints:
        - Special tokens (CLS, EOS, PAD, etc.) can only predict themselves.
        - The mask token can predict any non-special token (but not itself).
        - All other tokens (standard vocabulary) predict only themselves.

    The mask matrix contains 0.0 for valid outputs and -inf for invalid outputs,
    so adding it to logits zeros out invalid positions after softmax.

    output_dim may exceed vocab_size when model designers pad the output space
    for memory alignment (e.g. ESM's 33-token vocab mapped to 64-dim output).
    Extra columns beyond vocab_size are valid mask outputs (not special tokens).

    Tensor index conventions:
        Ti: input token index — rows of the mask matrix, size = vocab_size
        To: output token index — columns of the mask matrix, size = output_dim
        S:  batch (sequence) index
        P:  position index within a sequence
        T:  token/vocab dimension in logits (same axis as To)
    """

    # TODO[pi] move the init to be a "from_hf_tokenizer" method and make the actual init general like we discussed
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        mask_token: str,
        output_dim: Optional[int] = None,
    ):

        nn.Module.__init__(self)

        self.tokenizer = tokenizer
        if output_dim is None:
            self.output_dim = self.tokenizer.vocab_size
        else:
            self.output_dim = output_dim
        assert self.output_dim >= self.tokenizer.vocab_size, (
            "Outputs can't include all tokens!! Output dim is less than the tokenizer vocab size"
        )  # TODO: add support for if output tokens are a subset of the tokenizer--maybe when create the constructor for non HF-tokenizer

        # Construct valid output mask: 0.0 = pass through, -inf = block
        PASS_THROUGH, BLOCK_OUT = 0.0, float("-inf")

        valid_output_mask_TiTo = torch.full(
            (self.tokenizer.vocab_size, self.output_dim), BLOCK_OUT, dtype=torch.float32
        )

        # All tokens map to themselves
        for idx in self.tokenizer.vocab.values():
            valid_output_mask_TiTo[idx, idx] = PASS_THROUGH
        mask_token_idx = self.tokenizer.vocab[mask_token]

        # Except for mask which maps to any non-special token
        valid_output_mask_TiTo[mask_token_idx, mask_token_idx] = BLOCK_OUT
        valid_mask_outputs = set(range(self.output_dim))
        for idx in self.tokenizer.added_tokens_decoder.keys():
            valid_mask_outputs.discard(idx)
        for idx in valid_mask_outputs:
            valid_output_mask_TiTo[mask_token_idx, idx] = PASS_THROUGH

        self.register_buffer("valid_output_mask_TiTo", valid_output_mask_TiTo)

    def forward(self, logits_SPT: torch.Tensor, seq_SP: torch.LongTensor):
        """Apply per-position output constraints to raw logits.

        Indexes the mask matrix by input token ids to select the constraint
        row for each position, then adds it to the logits. Positions with
        special tokens will have -inf at all output indices except their own;
        mask positions will have 0.0 at all non-special outputs.

        Args:
            logits_SPT: Raw model logits, shape (S, P, T).
            seq_SP: Input token ids, shape (S, P).

        Returns:
            Constrained logits as float32, shape (S, P, To).
        """
        output_mask_SPTo = self.valid_output_mask_TiTo[seq_SP]
        return logits_SPT.float() + output_mask_SPTo


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

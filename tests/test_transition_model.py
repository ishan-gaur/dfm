"""Tests for TransitionModel ABC.

Uses a lightweight mock child class that mirrors the ESM pattern from
~/PALM/esm-cath/src/esm_cath/model.py without importing it (avoids circular deps).
"""

import torch
import pytest
from torch import nn
from torch.nn import functional as F

from dfm.generative_model import (
    TransitionModel,
    LogitFormatter,
    PassThroughLogitFormatter,
    MaskedModelLogitFomatter,
)


# ---------------------------------------------------------------------------
# Mock tokenizer — mimics HF PreTrainedTokenizerBase interface used by
# EsmSequenceTokenizer in the real ESM child class
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer satisfying TransitionModel.forward_from_string needs.

    20 standard amino acids (A-Y) at indices 0-19,
    plus <pad>=20, <cls>=21, <eos>=22, <mask>=23.
    """

    _AA = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self):
        self.vocab: dict[str, int] = {aa: i for i, aa in enumerate(self._AA)}
        self.vocab["<pad>"] = 20
        self.vocab["<cls>"] = 21
        self.vocab["<eos>"] = 22
        self.vocab["<mask>"] = 23
        self._idx_to_token = {v: k for k, v in self.vocab.items()}

        self.pad_token_id = 20
        self.cls_token_id = 21
        self.eos_token_id = 22
        self.mask_token_id = 23

        # HF tokenizers expose special tokens via added_tokens_decoder
        self.added_tokens_decoder: dict[int, object] = {
            20: "<pad>",
            21: "<cls>",
            22: "<eos>",
            23: "<mask>",
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, sequence: str) -> list[int]:
        """Encode with CLS/EOS wrapping, like EsmSequenceTokenizer."""
        ids = [self.cls_token_id]
        ids.extend(self.vocab.get(c, self.pad_token_id) for c in sequence)
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._idx_to_token.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Concrete TransitionModel children — mirror the ESM pattern:
#   super().__init__(tokenizer=..., logit_formatter=...)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 24  # 20 AA + 4 special
OUTPUT_DIM = 32  # like ESM's 64-dim padded output


class StubTransitionModel(TransitionModel):
    """Concrete child that mirrors ESM's structure:

    - Wraps a simple linear 'backbone' (so parameters() is non-empty)
    - Passes tokenizer & logit_formatter up to the ABC __init__
    - forward() returns log-softmax logits of shape (S, P, OUTPUT_DIM)
    """

    def __init__(
        self,
        tokenizer: FakeTokenizer | None = None,
        logit_formatter: LogitFormatter | None = None,
        output_dim: int = OUTPUT_DIM,
    ):
        tok = tokenizer or FakeTokenizer()
        fmt = logit_formatter or PassThroughLogitFormatter()
        super().__init__(tokenizer=tok, logit_formatter=fmt)
        self._backbone = nn.Linear(1, 1)  # gives us a parameter for .device
        self._output_dim = output_dim

    def forward(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        S, P = seq_SP.shape
        logits_SPT = torch.randn(S, P, self._output_dim, device=seq_SP.device)
        logits_SPT = self.logit_formatter(logits_SPT, seq_SP)
        return F.log_softmax(logits_SPT, dim=-1)


class RecordingTransitionModel(TransitionModel):
    """Records every forward call's input tensor for assertion."""

    def __init__(self, tokenizer: FakeTokenizer | None = None):
        tok = tokenizer or FakeTokenizer()
        super().__init__(tokenizer=tok, logit_formatter=PassThroughLogitFormatter())
        self._backbone = nn.Linear(1, 1)
        self.forward_calls: list[torch.Tensor] = []

    def forward(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        self.forward_calls.append(seq_SP.clone())
        S, P = seq_SP.shape
        return torch.randn(S, P, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def model(tokenizer):
    return StubTransitionModel(tokenizer=tokenizer)


@pytest.fixture
def recording_model(tokenizer):
    return RecordingTransitionModel(tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestABCContract:
    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError, match="abstract"):
            TransitionModel(
                tokenizer=FakeTokenizer(),
                logit_formatter=PassThroughLogitFormatter(),
            )

    def test_missing_forward_raises(self):
        class NoForward(TransitionModel):
            pass

        with pytest.raises(TypeError, match="abstract"):
            NoForward(
                tokenizer=FakeTokenizer(),
                logit_formatter=PassThroughLogitFormatter(),
            )

    def test_complete_implementation_instantiates(self, model):
        assert isinstance(model, TransitionModel)
        assert isinstance(model, nn.Module)

    def test_tokenizer_set_by_init(self, model, tokenizer):
        assert model.tokenizer is tokenizer

    def test_logit_formatter_set_by_init(self):
        fmt = PassThroughLogitFormatter()
        m = StubTransitionModel(logit_formatter=fmt)
        assert m.logit_formatter is fmt


# ---------------------------------------------------------------------------
# device property tests
# ---------------------------------------------------------------------------


class TestDevice:
    def test_device_returns_cpu_by_default(self, model):
        assert model.device == torch.device("cpu")

    def test_device_matches_parameter_device(self, model):
        assert model.device == next(model.parameters()).device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_follows_to_cuda(self, model):
        model = model.cuda()
        assert model.device.type == "cuda"
        model = model.cpu()
        assert model.device.type == "cpu"

    def test_device_raises_without_parameters(self):
        """If a child class has no nn.Parameters, .device raises StopIteration."""

        class NoParams(TransitionModel):
            def forward(self, seq_SP):
                return torch.randn(1)

        m = NoParams(
            tokenizer=FakeTokenizer(),
            logit_formatter=PassThroughLogitFormatter(),
        )
        with pytest.raises(StopIteration):
            _ = m.device


# ---------------------------------------------------------------------------
# forward tests
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape(self, model):
        seq = torch.tensor([[21, 0, 4, 7, 22]])  # CLS A E H EOS
        out = model(seq)
        assert out.shape == (1, 5, OUTPUT_DIM)

    def test_output_is_log_probs(self, model):
        """log_softmax output should exponentiate to probabilities summing to 1."""
        seq = torch.tensor([[21, 0, 4, 7, 22]])
        log_probs = model(seq)
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_batched_forward(self, model):
        seq = torch.tensor(
            [
                [21, 0, 4, 7, 22, 20],  # CLS A E H EOS PAD
                [21, 1, 2, 3, 8, 22],  # CLS C D F I EOS
            ]
        )
        out = model(seq)
        assert out.shape == (2, 6, OUTPUT_DIM)

    def test_forward_called_with_correct_tensor(self, recording_model):
        seq = torch.tensor([[21, 0, 4, 22]])
        recording_model(seq)
        assert len(recording_model.forward_calls) == 1
        assert torch.equal(recording_model.forward_calls[0], seq)


# ---------------------------------------------------------------------------
# forward_from_string tests
# ---------------------------------------------------------------------------


class TestForwardFromString:
    def test_single_sequence(self, recording_model, tokenizer):
        recording_model.forward_from_string(["ACE"])
        assert len(recording_model.forward_calls) == 1
        tensor = recording_model.forward_calls[0]
        expected = torch.tensor([tokenizer.encode("ACE")], dtype=torch.long)
        assert torch.equal(tensor, expected)

    def test_output_shape(self, model):
        out = model.forward_from_string(["ACE"])
        # encode adds CLS + EOS: "ACE" → [CLS, A, C, E, EOS] = 5 tokens
        assert out.shape == (1, 5, OUTPUT_DIM)

    def test_batch_of_sequences(self, recording_model):
        recording_model.forward_from_string(["AC", "ACDE"])
        tensor = recording_model.forward_calls[0]
        assert tensor.shape[0] == 2  # batch size

    def test_padding_to_max_length(self, recording_model, tokenizer):
        """Shorter sequences should be right-padded to the longest."""
        recording_model.forward_from_string(["AC", "ACDE"])
        tensor = recording_model.forward_calls[0]

        # "AC"   → [CLS, A, C, EOS]       = 4 tokens → padded to 6
        # "ACDE" → [CLS, A, C, D, E, EOS] = 6 tokens
        assert tensor.shape == (2, 6)
        # Last 2 positions of first sequence should be pad tokens
        assert tensor[0, -1].item() == tokenizer.pad_token_id
        assert tensor[0, -2].item() == tokenizer.pad_token_id
        # Second sequence should have no padding
        assert tensor[1, -1].item() == tokenizer.eos_token_id

    def test_equal_length_no_padding(self, recording_model, tokenizer):
        recording_model.forward_from_string(["ACE", "FGH"])
        tensor = recording_model.forward_calls[0]
        # Both encode to length 5 (CLS + 3 AA + EOS), no padding needed
        assert tensor.shape == (2, 5)
        assert (tensor != tokenizer.pad_token_id).all()

    def test_device_propagation(self, recording_model):
        """Tensor passed to forward should be on the model's device."""
        recording_model.forward_from_string(["ACE"])
        tensor = recording_model.forward_calls[0]
        assert tensor.device == recording_model.device

    def test_dtype_is_long(self, recording_model):
        recording_model.forward_from_string(["ACE"])
        tensor = recording_model.forward_calls[0]
        assert tensor.dtype == torch.long

    def test_empty_sequence(self, recording_model, tokenizer):
        """Empty string should still produce CLS + EOS."""
        recording_model.forward_from_string([""])
        tensor = recording_model.forward_calls[0]
        assert tensor.shape == (1, 2)
        assert tensor[0, 0].item() == tokenizer.cls_token_id
        assert tensor[0, 1].item() == tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# logit_formatter interaction tests
# ---------------------------------------------------------------------------


class TestFormatLogitsIntegration:
    """Verify that the ABC's contract around logit_formatter plays correctly
    with MaskedModelLogitFomatter — the formatter used by ESM."""

    @pytest.fixture
    def esm_like_model(self, tokenizer):
        """Model using MaskedModelLogitFomatter, like the real ESM class."""
        formatter = MaskedModelLogitFomatter(tokenizer, "<mask>", output_dim=OUTPUT_DIM)
        return StubTransitionModel(
            tokenizer=tokenizer, logit_formatter=formatter, output_dim=OUTPUT_DIM
        )

    def test_mask_positions_block_special_tokens(self, esm_like_model, tokenizer):
        mask_id = tokenizer.mask_token_id
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model(seq)

        # At the mask position, special tokens should have prob ~0
        mask_pos_probs = log_probs[0, 1].exp()
        for special_id in tokenizer.added_tokens_decoder:
            assert mask_pos_probs[special_id] < 1e-6, (
                f"Special token {special_id} should be blocked at mask position"
            )

    def test_special_positions_predict_themselves(self, esm_like_model, tokenizer):
        mask_id = tokenizer.mask_token_id
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model(seq)

        # CLS position: all probability on CLS
        cls_probs = log_probs[0, 0].exp()
        assert torch.isclose(
            cls_probs[tokenizer.cls_token_id], torch.tensor(1.0), atol=1e-5
        )

        # EOS position: all probability on EOS
        eos_probs = log_probs[0, 2].exp()
        assert torch.isclose(
            eos_probs[tokenizer.eos_token_id], torch.tensor(1.0), atol=1e-5
        )

    def test_regular_token_predicts_itself(self, esm_like_model, tokenizer):
        """A non-mask, non-special token should have all mass on itself."""
        aa_id = tokenizer.vocab["A"]
        seq = torch.tensor([[tokenizer.cls_token_id, aa_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model(seq)
        aa_probs = log_probs[0, 1].exp()
        assert torch.isclose(aa_probs[aa_id], torch.tensor(1.0), atol=1e-5)

    def test_mask_position_allows_all_non_special(self, esm_like_model, tokenizer):
        """Mask position should have nonzero probability for every non-special token."""
        mask_id = tokenizer.mask_token_id
        seq = torch.tensor([[tokenizer.cls_token_id, mask_id, tokenizer.eos_token_id]])
        log_probs = esm_like_model(seq)
        mask_probs = log_probs[0, 1].exp()

        non_special = set(range(tokenizer.vocab_size)) - set(
            tokenizer.added_tokens_decoder
        )
        for idx in non_special:
            assert mask_probs[idx] > 0, (
                f"Non-special token {idx} should have nonzero prob"
            )

    def test_logit_formatter_is_accessible(self, esm_like_model):
        assert isinstance(esm_like_model.logit_formatter, LogitFormatter)
        assert isinstance(esm_like_model.logit_formatter, MaskedModelLogitFomatter)


# ---------------------------------------------------------------------------
# get_transition_log_probs tests
# ---------------------------------------------------------------------------


class TestGetTransitionLogProbsFn:
    """get_transition_log_probs_fn should return a callable that applies
    logit_formatter to the raw forward output. Currently the method body
    is incomplete (no return statement), so this documents intended behavior."""

    def test_returns_callable(self, model):
        """get_transition_log_probs_fn should return a callable."""
        result = model.get_transition_log_probs_fn()
        assert callable(result)


# ---------------------------------------------------------------------------
# nn.Module integration tests
# ---------------------------------------------------------------------------


class TestModuleIntegration:
    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_parameters_iterable(self, model):
        params = list(model.parameters())
        assert len(params) > 0

    def test_state_dict_saveable(self, model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        loaded = torch.load(path, weights_only=True)
        assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_eval_and_train_modes(self, model):
        model.eval()
        assert not model.training
        model.train()
        assert model.training

    def test_formatter_submodule_device_propagation(self, tokenizer):
        """When logit_formatter is an nn.Module (like MaskedModelLogitFomatter),
        registering it as a submodule lets .to(device) propagate buffers."""
        formatter = MaskedModelLogitFomatter(tokenizer, "<mask>", output_dim=OUTPUT_DIM)

        class ModuleFormatterModel(TransitionModel):
            def __init__(self):
                super().__init__(tokenizer=tokenizer, logit_formatter=formatter)
                self._backbone = nn.Linear(1, 1)
                # Register formatter as a named submodule so .to() propagates
                self.add_module("_formatter_module", formatter)

            def forward(self, seq_SP):
                return torch.randn(1)

        m = ModuleFormatterModel()
        child_types = [type(c) for c in m.children()]
        assert MaskedModelLogitFomatter in child_types

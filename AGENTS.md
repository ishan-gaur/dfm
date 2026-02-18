# Agent Guidelines

## Project Management

- Use `uv` for all package management and running Python code
  - Install dependencies: `uv add <package>`
  - Run scripts: `uv run python <script>`
  - Sync environment: `uv sync`
  - Install package in editable mode: `uv pip install -e .`
  - Run tests: `uv run pytest tests/ -v`
  - Run formatter: `uv run ruff format`
  - Run lineter: `uv run ruff check`
- Use this file to note down project-related info important to know across sessions
- **Discuss design decisions before implementing** - especially for abstractions and class structures
- **Ask for clarification when instructions seem contradictory** - don't guess intent, surface the confusion

## Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Keep functions focused and modular
- Make user-provided inputs required, not optional - if the user needs to provide something, the interface should demand it
- Prefer strict interfaces that prevent misuse over permissive ones that raise errors at runtime
- Let code crash naturally rather than wrapping in try/catch - silent failures hide bugs
- Use assert statements to validate inputs/outputs in complex pipelines and tricky functions - they serve as executable documentation and catch issues early
- Follow John Ousterhouts A Philosophy of Software Design
- Use `raise ValueError` for input validation at API boundaries, asserts for internal invariants
- Prefer torch.Tensor over numpy arrays when data will be used in training
- Keep return structures flat and minimal - only include what's actually needed
- Implement functionality or remove it - no silent no-ops (e.g. don't accept a parameter and ignore it)
- Direct indexing over clever fallbacks - if something should be indexable, just index it and let it crash if not

## Project Structure

- `src/dfm/` — core library (installed as editable package; run `uv pip install -e .` after changes)
  - `data.py` — `GuidanceDataset` base class, `NoiseSchedule` type alias, schedule functions
  - `generative_model.py` — `TransitionModel` ABC, `LogitFormatter` protocol, `MaskedModelLogitFomatter`, `PassThroughLogitFormatter`, `MPNNTokenizer`
  - `predictive_model.py` — `OneHotMLP` (frozen one-hot embedding → MLP), `LinearProbe`
- `tests/` — pytest tests (`test_guidance_data.py`, `test_logit_formatter.py`, `test_transition_model.py`)
- `TODO.md` — phased roadmap (Phase 1 done, Phase 2–4 pending)

## TransitionModel Design

- `TransitionModel.__init__` takes `tokenizer` and `format_logits` as **required** positional args — child classes pass them via `super().__init__(tokenizer=..., format_logits=...)`
- Only `forward()` is abstract; `tokenizer` and `format_logits` are plain instance attributes set by `__init__`
- Design decision: avoided `@property @abstractmethod` for tokenizer/format_logits because it blocks the ESM pattern of `self.tokenizer = X` in child `__init__` (abstract properties are data descriptors that intercept assignment)
- The ESM child class in `~/PALM/esm-cath/src/esm_cath/model.py` is the reference consumer — don't introduce ABC changes that break its pattern
- `get_transition_log_probs` is incomplete (no return statement) — known bug, needs design discussion re: conditional models
- `generative_model.py` uses `from __future__ import annotations` for lazy annotation eval (needed because `TransitionModel` references `LogitFormatter` which is defined later in the file)
- Run tests with `uv run python -m pytest` (not `uv run pytest` — pytest not on PATH directly) [×1]

## LogitFormatter / MaskedModelLogitFomatter Design

- `LogitFormatter` is a `@runtime_checkable Protocol` — defines `__call__(logits, input_ids) -> FloatTensor`
- `MaskedModelLogitFomatter` inherits `(nn.Module, LogitFormatter)` — **nn.Module must come first** in MRO or Protocol's `__call__` shadows nn.Module's (returns None instead of dispatching to `forward`)
- Uses `nn.Module.__init__(self)` instead of `super().__init__()` because Protocol in the MRO breaks cooperative `super()` chain for nn.Module init
- The mask matrix uses `register_buffer` for device tracking (no gradients, moves with `.to(device)`)
- Uses **direct indexing** (`mask_matrix[token_ids]`) NOT one-hot matmul — `0.0 * (-inf) = NaN` in IEEE floats kills the matmul approach
- Uses **additive masking** (0.0 pass-through, -inf block) NOT multiplicative — multiplying logits by -inf gives wrong signs for negative logits
- `output_dim` can exceed `vocab_size` for memory alignment (e.g. ESM's 33-token vocab → 64-dim output)
- ESM tokenizer v3.0.3: `mask_token_id` is `None` — always use `tokenizer.vocab["<mask>"]` instead
- Current constructor is HF-tokenizer-specific (uses `.vocab`, `.added_tokens_decoder`) — TODO to add a general constructor taking primitive ids
- Reference consumer: `~/PALM/esm-cath/src/esm_cath/model.py` ESM class
- Tests in `tests/test_logit_formatter.py` (24 tests) cover ESM and BERT tokenizers

## Stale Tests

- `test_guidance_data.py::TestGuidanceDataset` — 3 tests fail because they construct `GuidanceDataset` without the now-required `tokenize`, `noise_schedule`, `mask_token` args

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `MPNNTokenizer` in `generative_model.py` wraps PMPNN's `MPNN_TOKEN_ENCODING` (21 tokens: 20 standard AAs + UNK at idx 20)
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — these are harmless

## Tokenization

- PMPNN vocabulary: 20 standard amino acids + UNK (X), indexed 0–20
- Mapping: one-letter AA → three-letter code (atomworks `DICT_THREE_TO_ONE`) → PMPNN index (`MPNN_TOKEN_ENCODING.token_to_idx`)
- `MPNNTokenizer()`: encode("ACDE") → [0,4,3,6], decode([0,4,3,6]) → "ACDE", __call__(["ACDE"]) → {"input_ids": tensor}

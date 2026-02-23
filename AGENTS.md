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
  - `probability_model.py` — `ProbabilityModel` ABC (shared base for transition and predictive models: temp, with_temp, abstract forward, concrete get_log_probs)
  - `generative_modeling.py` — `TransitionModel` ABC, `ConditionalTransitionModel`, `LogitFormatter` protocol, `MaskedModelLogitFormatter`, `PassThroughLogitFormatter`, `MPNNTokenizer`
  - `predictive_modeling.py` — `PredictiveModel` (inherits `ProbabilityModel`), `ClassValuedPredictiveModel`, `RealValuedPredictiveModel`, `ConditionalPredictiveModel`, `OneHotMLP`, `LinearProbe`
  - `mixins.py` — `ConditionableMixin` (conditioning state management: set_condition_, conditioned_on, etc.)
  - `guide.py` — `TAG`, `DEG`, `TokenizerTranslator` (guidance algorithms)
  - `sampling.py` — `sample_any_order_ancestral`
  - `data.py` — `GuidanceDataset` base class, `NoiseSchedule` type alias, schedule functions
  - `models/esm.py` — `ESMC`, `ESM3IF` wrappers (ESM3IF is a stub)
  - `models/rocklin_ddg/` — stability predictor (StabilityPMPNN, PreTrainedStabilityPredictor), data_utils, guidance_utils
  - `models/utils.py` — `pdb_to_atom37_and_seq` (incomplete)
- `examples/stability_guidance/main.py` — cleaned-up stability guidance example (uses dfm abstractions, many TODOs)
- `tests/` — pytest tests (`test_guidance_data.py`, `test_logit_formatter.py`, `test_transition_model.py`)
- `TODO.md` — phased roadmap (Phase 1 done, Phase 2–4 pending)

## TransitionModel Design

- `TransitionModel.__init__` takes `tokenizer` and `logit_formatter` as **required** positional args — child classes pass them via `super().__init__(tokenizer=..., logit_formatter=...)`
- Only `forward()` is abstract; `tokenizer` and `logit_formatter` are plain instance attributes set by `__init__`
- Design decision: avoided `@property @abstractmethod` for tokenizer/logit_formatter because it blocks the ESM pattern of `self.tokenizer = X` in child `__init__` (abstract properties are data descriptors that intercept assignment)
- The ESM child class in `~/PALM/esm-cath/src/esm_cath/model.py` is the reference consumer — don't introduce ABC changes that break its pattern
- `generative_modeling.py` uses `from __future__ import annotations` for lazy annotation eval (needed because `TransitionModel` references `LogitFormatter` which is defined later in the file)
- Run tests with `uv run python -m pytest` (not `uv run pytest` — pytest not on PATH directly) [×1]
- **Planned**: TransitionModel will eventually inherit from ProbabilityModel (same as PredictiveModel does now)

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

## ProbabilityModel / PredictiveModel Design

- `ProbabilityModel(nn.Module, ABC)` in `probability_model.py` — shared base for both transition and predictive model hierarchies
- Has `temp`, `with_temp()`, `set_temp_()`, abstract `forward(x, **kwargs)`, concrete `get_log_probs(x, **kwargs)` = `log_softmax(forward()/temp)`
- Default `get_log_probs` does log_softmax (fine for class-valued models); real-valued models will override when TargetProbabilityMixin lands
- `PredictiveModel` inherits from `ProbabilityModel`; temp/with_temp come from ProbabilityModel, NOT duplicated
- Child classes must set `self.input_dim` for `target_log_probs_given_seq` to work (no enforcement, crashes at call time if missing)
- **Planned refactors**:
  - `ClassValuedPredictiveModel`, `RealValuedPredictiveModel`, `EnsemblePredictiveModel` will be rewritten as `TargetProbabilityMixin` variants (ClassTargetMixin, RealValuedTargetMixin, etc.) — these define how raw model output → log p(target_event | x)
  - TransitionModel will inherit from ProbabilityModel

## ConditionableMixin Design

- Lives in `mixins.py`, inherits from `ABC` (so @abstractmethod enforcement propagates via ABCMeta)
- HuggingFace mixin pattern: **no `__init__`**, class-level `observations = None` default. Avoids all MRO/cooperative init issues with nn.Module
- Provides concrete: `set_condition_()`, `set_condition()`, `conditioned_on()` (context manager with revert, supports nesting)
- Requires abstract: `preprocess_observations()` (one-time expensive caching), `observation_collator()` (**static method** — callable on class without instance, reusable by datasets/dataloaders)
- Used by `ConditionalTransitionModel(TransitionModel, ConditionableMixin)` and `ConditionalPredictiveModel(PredictiveModel, ConditionableMixin)`
- `ConditionalTransitionModel.transition_log_probs` handles tiling cached observations to match batch size, then passes collated obs as **kwargs to forward
- **ABC on mixin is required**: without it, @abstractmethod decorators on the mixin don't propagate through ABCMeta's collection (ABCMeta only checks `base.__abstractmethods__` which isn't set on non-ABC classes)
- **MRO note**: `ConditionableMixin(ABC)` composes cleanly with `TransitionModel(nn.Module, ABC)` and `PredictiveModel(ProbabilityModel, ABC)` — ABCMeta is the metaclass for all, no conflicts

## Stale Tests / Broken Imports

- `test_guidance_data.py::TestGuidanceDataset` — 3 tests fail because they construct `GuidanceDataset` without the now-required `tokenize`, `noise_schedule`, `mask_token` args
- `tests/test_logit_formatter.py` and `tests/test_transition_model.py` import from `dfm.generative_model` (should be `dfm.generative_modeling`)
- `tests/test_esm.py` fails because `MaskedModelLogitFormatter` constructor call in `models/esm.py` passes wrong number of args
- `guide.py` imports from `dfm.predictive_model` (should be `dfm.predictive_modeling`)
- `__init__.py` had the same stale import — fixed in this session

## Stability Predictor (rocklin_ddg)

- `StabilityPMPNN` in `models/rocklin_ddg/stability_predictor.py` — PMPNN-based stability predictor with encode/decode split
- `encode_structure()` is expensive (runs once per structure), `decode()` is cheap (runs per sample) — this is the conditioning pattern that ConditionableMixin formalizes
- `PreTrainedStabilityPredictor(ClassValuedPredictiveModel)` wraps StabilityPMPNN — syntax errors fixed (missing self, `class` keyword), but forward/conditioning not yet implemented
- The old working example (`models/rocklin_ddg/example_usage.py`) uses local `data_utils.py` and `guidance_utils.py` — these do NOT use dfm abstractions
- `data_utils.py` has ~300 lines of PMPNN-specific featurization (featurize, prepare_conditioning_inputs, token conversion, PDB loading via biotite)
- `guidance_utils.py` has flow matching Euler sampling + TAG guidance + ESM3 inverse folding wrappers — most of this is replicated by `guide.py` (TAG/DEG) and `sampling.py`
- The new example (`examples/stability_guidance/main.py`) uses dfm abstractions but has many unresolved TODOs
- **Next steps**: implement PreTrainedStabilityPredictor.forward using ConditionableMixin (preprocess_observations = encode_structure, forward uses cached embeddings + decoder), finish pdb_to_atom37_and_seq in models/utils.py, get the new example working, then delete the old code

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `MPNNTokenizer` in `generative_model.py` wraps PMPNN's `MPNN_TOKEN_ENCODING` (21 tokens: 20 standard AAs + UNK at idx 20)
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — these are harmless

## Tokenization

- PMPNN vocabulary: 20 standard amino acids + UNK (X), indexed 0–20
- Mapping: one-letter AA → three-letter code (atomworks `DICT_THREE_TO_ONE`) → PMPNN index (`MPNN_TOKEN_ENCODING.token_to_idx`)
- `MPNNTokenizer()`: encode("ACDE") → [0,4,3,6], decode([0,4,3,6]) → "ACDE", __call__(["ACDE"]) → {"input_ids": tensor}

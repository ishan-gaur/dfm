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

- `src/guidance/` — core library (installed as editable package; run `uv pip install -e .` after changes)
  - `data.py` — `GuidanceDataset` base class, `NoiseSchedule` type alias, schedule functions
  - `generative_model.py` — `MPNNTokenizer` (wraps PMPNN's vocab as a HuggingFace-style tokenizer: encode/decode/__call__)
  - `predictive_model.py` — `OneHotMLP` (frozen one-hot embedding → MLP), `LinearProbe`
- `tyrosine_kinase/` — tyrosine kinase specific code and data (not a package in `src/`)
  - `data.py` — `load_dataset()` returns a `GuidanceDataset` for kinase data
  - `train_ohe_mlp.py` — training script for OneHotMLP (wandb logging, train/val/test split)
  - `data/` — raw CSVs (`final_single_data.csv`, `final_combi_data.csv`)
  - `checkpoints/` — saved model weights (gitignored)
  - `analysis/` — `analyze_data.py` regenerates all README plots; output PNGs live here
  - `notebooks/` — `explore_data.ipynb` for interactive exploration
  - `structures/` — empty, placeholder for PDB/structure files
- `tests/` — pytest tests (`test_guidance_data.py`)
- `TODO.md` — phased roadmap (Phase 1 done, Phase 2–4 pending)

## Data

### Key rules
- **Important**: Use `norm_rate` (expression-normalized activity) as the target metric, NOT raw `slope` (which is just total fluorescence)
- The `conc` column is raw expression; normalize it to WT the same way as activity (per-set for single, mean-WT for combi)
- All loading code clips negative activity values to 0 after normalization (measurement noise artifact; currently only affects combi but applied to both datasets)

### Single mutation data (`final_single_data.csv`)
- 434 variants + 2 WT controls (one per experimental set, rows where `mut == "wt"`)
- Columns: `mut`, `slope`, `conc`, `norm_rate`, `set`, `seq`
- `mut` format is e.g. "D105A" (position 105, Asp→Ala) — positions are relative to the 295-residue construct, not UniProt numbering
- Normalization: divide by WT `norm_rate`/`conc` from the same `set`
- 76 unique positions, 1–18 mutations per position
- Activity range after normalization: 0.09–1.87 (mean 0.83)

### Combinatorial mutation data (`final_combi_data.csv`)
- 537 variants, no WT rows in this file (WT reference comes from single data)
- Columns: (unnamed index), `id`, `seq`, `slope`, `conc`, `norm_rate`, `design`
- `design` is either "f2s" (Frame2Seq, 257) or "mpnn" (ProteinMPNN, 280)
- Normalization: divide by mean WT values across both experimental sets
- 26–51 mutations per variant from WT (heavily redesigned)
- Activity range after normalization + clipping: 0.00–1.36 (mean 0.22)

### Loading data
- `tyrosine_kinase.data.load_dataset(source, tokenize, noise_schedule, mask_token)` — returns a `GuidanceDataset`; `source` is "single", "combi", or "both"
- For quick pandas exploration without the full GuidanceDataset machinery, load CSVs directly (see notebook)
- All 295-residue sequences share the same WT backbone; WT sequence is in the `seq` column of WT rows

## External Dependencies

- ProteinMPNN via Foundry: `rc-foundry[all]` — provides `mpnn` and `atomworks` packages
- `MPNNTokenizer` in `generative_model.py` wraps PMPNN's `MPNN_TOKEN_ENCODING` (21 tokens: 20 standard AAs + UNK at idx 20)
- Importing from `atomworks` prints env var warnings (CCD_MIRROR_PATH, PDB_MIRROR_PATH) — these are harmless

## Tokenization

- PMPNN vocabulary: 20 standard amino acids + UNK (X), indexed 0–20
- Mapping: one-letter AA → three-letter code (atomworks `DICT_THREE_TO_ONE`) → PMPNN index (`MPNN_TOKEN_ENCODING.token_to_idx`)
- All kinase sequences are 295 residues, standard AAs only (no UNK in data)
- `MPNNTokenizer()`: encode("ACDE") → [0,4,3,6], decode([0,4,3,6]) → "ACDE", __call__(["ACDE"]) → {"input_ids": tensor}

## Baseline Results (OneHotMLP)

- 826K trainable params (model_dim=128, n_layers=3, output_dim=2: [activity, expression])
- 80/10/10 split (777/97/97), seed=42, 100 epochs, AdamW lr=1e-3
- Overfits significantly by epoch ~10-20 (train loss keeps dropping, val loss plateaus/rises)
- Best val loss at epoch ~10: test activity R²=0.73, expression R²=0.32
- Per-source test R²: single mutations near 0, combi near 0 — the mixed-source R² is inflated by the large gap between single (high activity) and combi (low activity) clusters
- The model learns the cluster-level distinction but not within-cluster variation
- Needs regularization (dropout, smaller model) or better features (embeddings from PMPNN)

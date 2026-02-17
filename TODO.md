# TODO

## Phase 1: Data Analysis & Setup

- [x] Create data analysis script for tyrosine kinase data
- [x] Add data file format documentation to README
- [x] Run analysis and generate figures for README
- [x] Create unified `GuidanceDataset` base class in `src/guidance/data.py`
- [x] Create `TyrosineKinaseDataset` implementing the interface

## Phase 2: ProteinMPNN Integration

- [ ] Install ProteinMPNN from Foundry (https://github.com/RosettaCommons/foundry/tree/production)
- [ ] Download/locate the structure file used in the tyrosine kinase paper for design
- [ ] Figure out structure integration with tokenizer (pass structure data via `sequence_metadata`)
- [ ] Implement teacher forcing through ProteinMPNN to score all variants
- [ ] Score single mutation variants
- [ ] Score combinatorial mutation variants

## Phase 3: Predictive Modeling (Create NoisyClassifier, finish GuidanceDataset, and apply to Kinase project)

- [ ] Add reproducible train/dev/test split support to `GuidanceDataset`
- [ ] Implement noising in `GuidanceDataset.__getitem__`:
  - Sample timestep t from noise_schedule()
  - For each position, mask if random() < t
  - Return noised tokens, timestep, and mask indicator
- [ ] Brainstorm classifier architectures:
  - ProteinMPNN embeddings + small attention layer + MLP
  - ProteinMPNN embeddings + CNN
  - Two output heads (expression + activity?)
- [ ] Evaluate self-prediction: single→single holdout, combi→combi holdout
- [ ] Evaluate cross-prediction: single→combi, combi→single

## Phase 4: Guidance Implementation

- [ ] Implement Taylor-Approximate Guidance (TAG)
- [ ] Implement Discrete-time Exact Guidance (DEG)
- [ ] Add conditional sampling functionality to `src/guidance/`

## Notes

- The tyrosine kinase data uses UniProt P54762, residues 602-896
- Single mutations are saturation mutagenesis at specific positions
- Combinatorial mutations are from Frame2Seq (f2s) and ProteinMPNN (mpnn)

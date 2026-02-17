# Guidance

A minimal library for protein sequence guidance using conditional generative models.

## References

- **ProteinGuide Paper**: [arXiv:2505.04823](https://arxiv.org/abs/2505.04823)
- **ProteinGuide Code**: [github.com/junhaobearxiong/protein_discrete_guidance](https://github.com/junhaobearxiong/protein_discrete_guidance)
- **Tyrosine Kinase Experiment Paper**: [bioRxiv:2025.08.03.668353v1](https://www.biorxiv.org/content/10.1101/2025.08.03.668353v1)
- **ProteinMPNN (Foundry)**: [github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry/tree/production)

## Overview

ProteinGuide is a method to materialize a conditional generative model on the fly to sample proteins or score existing sequences according to a desired property. It combines the outputs of a predictive model and a generative model using Bayes' rule.

This library will implement:
- **Taylor-Approximate Guidance (TAG)**: Approximate guidance method
- **Discrete-time Exact Guidance (DEG)**: Exact guidance for discrete sequences

## Tyrosine Kinase Data

This project includes experimental data for human EphB1 receptor tyrosine kinase (UniProt [P54762](https://www.uniprot.org/uniprotkb/P54762), residues 602-896) mutated using ProteinMPNN and Frame2Seq to improve kinase activity.

### Data Files

Located in `tyrosine_kinase/data/`:

#### `final_single_data.csv`
Single-position mutations with the following columns:
| Column | Description |
|--------|-------------|
| `mut` | Mutation identifier (e.g., "D105A" = position 105, Asp→Ala) |
| `slope` | Raw fluorescence readout (do not use directly) |
| `conc` | Expression level (protein concentration) |
| `norm_rate` | **Expression-normalized activity (use this as target)** |
| `set` | Experimental batch/set number |
| `seq` | Full protein sequence (295 residues) |

#### `final_combi_data.csv`
Combinatorial mutations (multiple positions) with the following columns:
| Column | Description |
|--------|-------------|
| `id` | Design identifier (e.g., "f2s-19-18027-temp0.3") |
| `seq` | Full protein sequence (295 residues) |
| `slope` | Raw fluorescence readout (do not use directly) |
| `conc` | Expression level |
| `norm_rate` | **Expression-normalized activity (use this as target)** |
| `design` | Design method ("f2s" = Frame2Seq, "mpnn" = ProteinMPNN) |

### Data Statistics

#### Single Mutation Data
- **Total variants**: 434 (+ 2 wild-type controls, one per experimental set)
- **Unique positions mutated**: 76
- **Mutations per position**: 1-18 (mean: 5.7)
- **Activity (relative to WT=1.0)**: 0.09 - 1.87 (mean: 0.83)
- **Expression (relative to WT=1.0)**: 0.18 - 1.99 (mean: 1.11)
- **Experimental sets**: 2 (Set 1: 91, Set 2: 343 variants)

![Expression vs Activity - Single](tyrosine_kinase/analysis/single_expression_vs_activity.png)

![Mutations per Position](tyrosine_kinase/analysis/single_position_histogram.png)

#### Combinatorial Mutation Data
- **Total variants**: 537
- **Mutations per variant**: 26-51 (mean: 36.9, median: 37)
- **Design methods**: Frame2Seq (257), ProteinMPNN (280)
- **Activity (relative to WT=1.0)**: 0.00 - 1.36 (mean: 0.22)
  - Some variants have small negative raw activity values (measurement noise); these are clipped to 0 after normalization
- **Expression (relative to WT=1.0)**: 0.26 - 4.06 (mean: 1.52)

![Expression vs Activity - Combi](tyrosine_kinase/analysis/combi_expression_vs_activity.png)

![Mutation Count Distribution](tyrosine_kinase/analysis/combi_mutation_count_distribution.png)

![Activity vs Mutation Count](tyrosine_kinase/analysis/combi_activity_vs_mutations.png)

### Key Observations

1. **Single mutations** show activity mostly below wildtype (mean 0.83× WT), with best variants reaching ~1.9× WT activity
2. **Combinatorial designs** have much lower activity (mean 0.22× WT), with most designs being non-functional
3. Combinatorial variants have 26-51 mutations from wild-type, indicating these are heavily redesigned sequences
4. Frame2Seq and ProteinMPNN designs show similar distributions of activity and expression

**Note**: All activity and expression values are normalized to wildtype levels (WT = 1.0) per experimental set.

## Installation

```bash
uv sync
uv run foundry install proteinmpnn
```

## Usage

```bash
# Run data analysis
uv run python tyrosine_kinase/analysis/analyze_data.py
```

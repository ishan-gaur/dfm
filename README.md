# Guidance

# TODO[pi] integrate with https://github.com/COLA-Laboratory/GraphFLA to get the different fitness landscapes
A minimal library for protein sequence guidance using conditional generative models.

## References

- **ProteinGuide Paper**: [arXiv:2505.04823](https://arxiv.org/abs/2505.04823)
- **ProteinGuide Code**: [github.com/junhaobearxiong/protein_discrete_guidance](https://github.com/junhaobearxiong/protein_discrete_guidance)
- **ProteinMPNN (Foundry)**: [github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry/tree/production)

## Overview

ProteinGuide is a method to materialize a conditional generative model on the fly to sample proteins or score existing sequences according to a desired property. It combines the outputs of a predictive model and a generative model using Bayes' rule.

This library will implement:
- **Taylor-Approximate Guidance (TAG)**: Approximate guidance method
- **Discrete-time Exact Guidance (DEG)**: Exact guidance for discrete sequences

## Installation

```bash
uv sync
uv run foundry install proteinmpnn
```

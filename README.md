# Evaluating Satisfiability Reasoning in Transformer-Based Language Models Across Logical Fragments

This repository accompanies my MSc Artificial Intelligence dissertation at the University of Manchester. It studies how **logical expressiveness**, **dataset construction**, **training configuration**, and **model architecture** affect Transformer performance on natural-language satisfiability classification.

Given a small theory written in controlled English, the task is to determine whether all statements can be true simultaneously (`satisfiable`) or whether they are inconsistent (`unsatisfiable`). Theories are generated from three nested logical fragments:

- **S**: a syllogistic fragment with inclusion, exclusion, and existential relations;
- **W**: S extended with relative clauses;
- **V**: W extended with transitive verbs and quantified relations.

The generator creates controlled theories and uses Z3 to derive SAT/UNSAT labels. The experiments compare ordinary sampling with a difficulty-controlled construction informed by phase-change calibration and structural weighting.

## Dissertation findings

The central result is that theoretical logical complexity alone does not determine empirical model difficulty. Under the difficulty-controlled dataset and joint DeBERTa-v3-large training, performance followed the expected `S > W > V` ordering across three random seeds. The same pattern did not hold consistently for separate training, T5-large, or the supplementary zero-shot evaluation.

| Model and condition | S | W | V | Overall |
| --- | ---: | ---: | ---: | ---: |
| DeBERTa-v3-large, joint training | 95.27% +/- 3.22% | 93.23% +/- 2.05% | 90.10% +/- 1.42% | 92.87% +/- 2.22% |
| T5-large, joint training | 63.58% | 65.85% | 68.73% | 66.06% |

### T5-large experiments

T5-large was evaluated alongside DeBERTa-v3-large under joint training, separate training, and a ten-epoch diagnostic run. The corresponding cluster job scripts are in `scripts/`; they use `google-t5/t5-large` and expose project and data locations as configuration variables.

See [results/main_results.md](results/main_results.md) for interpretation and scope.

## Repository layout

```text
configs/    Experiment configuration
examples/   Six generated, labelled example theories
results/    Concise results and interpretation
scripts/    DeBERTa and T5 training / cluster job scripts
src/        Dataset generation, curation, audits, and upstream-derived support code
```

Full datasets, checkpoints, cached model files, and training logs are deliberately excluded. The dataset-generation pipeline can recreate data from the documented configuration; the small files in `examples/` are only for format inspection.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Calibrate a compact SAT phase-change region for S, W, and V:

```bash
python src/pipeline.py calibrate-region \
  --config configs/data4_replication.json \
  --samples 120 \
  --output artifacts/phase_region.json
```

## Attribution

Parts of `src/official_fragments.py` and `src/official_data_construction.py` are adapted from the publicly released NL-SAT implementation (Madusanka et al., ACL 2024); see [NOTICE](NOTICE) for attribution.

## Reproducibility notes

- The original paper phase-sampling CSV was not publicly released. This repository therefore re-estimates a phase region from the released generator rather than recreating the exact original dataset.
- Dataset construction combines phase-region sampling, vocabulary conditions, structural weighting, and fragment metadata; the dissertation does not claim to isolate the causal effect of phase-change sampling alone.
- Reported joint DeBERTa results use three random seeds; separate-training results were materially less stable.

## License

Repository-specific code and documentation are released under the MIT License. Upstream-derived files remain available under Apache-2.0; see [NOTICE](NOTICE).

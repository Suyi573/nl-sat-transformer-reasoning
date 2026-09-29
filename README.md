# Evaluating Satisfiability Reasoning in Transformer-Based Language Models Across Logical Fragments

This repository accompanies my MSc Artificial Intelligence dissertation at the University of Manchester. It studies how **logical expressiveness**, **dataset construction**, **training configuration**, and **model architecture** affect Transformer performance on natural-language satisfiability classification.

Given a small theory written in controlled English, the task is to determine whether all its statements can be true simultaneously (`satisfiable`) or whether they are inconsistent (`unsatisfiable`). Theories are generated from three nested logical fragments:

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

See [results/main_results.md](results/main_results.md) for interpretation and scope.

## Repository layout

```text
pipeline.py                  Phase-region calibration and candidate generation
official_fragments.py        Upstream S/W/V template implementation (Apache-2.0)
official_data_construction.py Upstream vocabulary and support code (Apache-2.0)
build_difficulty_controlled_dataset.py  Difficulty-controlled dataset curation
train_deberta.py             DeBERTa fine-tuning and evaluation
audit_final.py               Split-integrity audit
data4_replication.json       Experiment configuration
sample_theories.jsonl        Six generated, labelled example theories
main_results.md              Concise results and interpretation
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
python pipeline.py calibrate-region \
  --config data4_replication.json \
  --samples 120 \
  --output artifacts/phase_region.json
```

Generate a small balanced S candidate pool from this calibration:

```bash
python pipeline.py generate-region \
  --config data4_replication.json \
  --calibration artifacts/phase_region.json \
  --fragment S \
  --min-per-label 50 \
  --output artifacts/S_candidates.jsonl \
  --report artifacts/S_generation_report.json
```

Fine-tune a DeBERTa classifier after preparing train, validation, and test JSONL files:

```bash
python train_deberta.py \
  --train path/to/train.jsonl \
  --validation path/to/validation.jsonl \
  --test path/to/test.jsonl \
  --output-dir runs/deberta \
  --model microsoft/deberta-v3-large \
  --max-length 512 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --epochs 5 \
  --early-stopping-patience 2 \
  --seed 42
```

Training requires a CUDA-capable GPU. The script refuses to truncate an instance, because truncation can change its satisfiability.

## Provenance and attribution

This is a dissertation implementation and experimental extension, not a claim of authorship of the original NL-SAT benchmark. The controlled S/W/V templates and released vocabulary used by the generator are derived from the public implementation accompanying:

> Tharindu Madusanka, Ian Pratt-Hartmann, and Riza Batista-Navarro. 2024. *Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-Based Language Models.* ACL 2024. https://aclanthology.org/2024.acl-long.815/

The source-derived files `official_fragments.py` and `official_data_construction.py` retain the upstream Apache-2.0 licence and attribution in [NOTICE](NOTICE). The phase-region calibration, dataset curation, experiment configuration, training workflow, and dissertation analysis are the work documented by this repository.

## Reproducibility notes

- The original paper's phase-sampling CSV was not publicly released. The calibration here therefore re-estimates a phase region from the released generator rather than recreating the authors' exact dataset.
- Dataset construction combines several controls (phase-region sampling, vocabulary conditions, structural weighting, and fragment metadata). The dissertation does **not** claim to isolate the causal effect of phase-change sampling alone.
- Reported results use three random seeds. Separate-training results were materially less stable than joint training.

## License

Repository-specific code and documentation are released under the MIT License. Upstream-derived files remain available under Apache-2.0; see [NOTICE](NOTICE).

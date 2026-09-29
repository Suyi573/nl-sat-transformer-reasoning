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


### T5-large experiments

T5-large was evaluated alongside DeBERTa-v3-large under joint training, separate training, and a ten-epoch diagnostic run. The associated cluster job scripts are `train_t5_joint_multiseed.slurm`, `train_t5_separate_multiseed.slurm`, and `train_t5_diagnostic_10_epochs.slurm`. They use `google-t5/t5-large` and expose the project and data locations as configuration variables.

See [main_results.md](main_results.md) for interpretation and scope.


## Repository layout


```text
pipeline.py                  Phase-region calibration and candidate generation
official_fragments.py        Upstream S/W/V template implementation (Apache-2.0)
official_data_construction.py Upstream vocabulary and support code (Apache-2.0)
build_difficulty_controlled_dataset.py  Difficulty-controlled dataset curation
train_deberta.py             DeBERTa fine-tuning and evaluation
train_t5_joint_multiseed.slurm     T5-large joint experiments (three seeds)
train_t5_separate_multiseed.slurm  T5-large separate experiments (S/W/V)
train_t5_diagnostic_10_epochs.slurm T5-large ten-epoch diagnostic run

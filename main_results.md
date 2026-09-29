# Main results

The main supervised experiment used a difficulty-controlled dataset with 24,000 training, 2,000 validation, and 2,000 test examples for each of S, W, and V. SAT and UNSAT labels were approximately balanced. DeBERTa-v3-large was evaluated under joint training over three random seeds.

| Fragment | Mean test accuracy +/- SD |
| --- | ---: |
| S | 95.27% +/- 3.22% |
| W | 93.23% +/- 2.05% |
| V | 90.10% +/- 1.42% |
| Overall | 92.87% +/- 2.22% |

The `S > W > V` ordering appeared in each joint-training run. On an ordinary, flatter dataset, joint DeBERTa training instead produced `S > V > W` (86.86% overall), showing that observed fragment difficulty is sensitive to data construction.

Separate DeBERTa training was substantially less stable: S-only, W-only, and V-only achieved 74.85% +/- 22.00%, 70.42% +/- 17.77%, and 79.70% +/- 2.61%, respectively. T5-large did not reproduce the expected ordering in the joint experiment (S 63.58%, W 65.85%, V 68.73%).

These results support a qualified conclusion: logical complexity can be reflected in model performance, but only in interaction with the data distribution, model architecture, optimisation, and training configuration.

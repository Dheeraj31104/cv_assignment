| Model | Clean Test | Patch-16 | Patch-8 | Short Take |
|---|---:|---:|---:|---|
| Plain-Old-CIFAR10-FC | 60.92% | 28.27% | 21.97% | Best FC clean accuracy, weak shuffle robustness |
| D-shuffletruffle-FC | 50.20% | 50.20% | 29.99% | Better patch-16 robustness, lower clean accuracy |
| N-shuffletruffle-FC | 46.03% | 46.03% | 46.03% | Fully shuffle-invariant, but weakest FC clean accuracy |
| Plain-Old-CIFAR10-CNN | 87.48% | 60.70% | 36.07% | Best overall clean accuracy |
| D-shuffletruffle-CNN | 70.89% | 70.89% | 47.57% | Strong clean/robustness tradeoff |
| N-shuffletruffle-CNN | 56.08% | 56.08% | 56.08% | Fully invariant, moderate robustness |
| Plain-Old-CIFAR10-Attention | 73.07% | 58.33% | 58.81% | Strong attention baseline, surprisingly robust |
| D-shuffletruffle-Attention | 51.78% | 51.78% | 28.26% | Moderate patch-16 robustness, weak patch-8 |
| N-shuffletruffle-Attention | Running | Running | Running | Final job still running |

| Overall Summary | Result |
|---|---|
| Best clean model | Plain-Old-CIFAR10-CNN |
| Best completed shuffle-robust model | N-shuffletruffle-CNN |
| General pattern | Plain models win on clean accuracy; `N-shuffletruffle` wins on invariance; `D-shuffletruffle` sits in between |

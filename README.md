# CIFAR-10 BN Project

Code for Project 2 of the course **Neural Network and Deep Learning**.

This repository contains:

- CIFAR-10 classification experiments
- architecture, loss, activation, optimizer, and scheduler ablations
- Batch Normalization analysis on VGG-style models

## Main entry

```bash
cd codes/VGG_BatchNorm
python run_project2.py --output-root outputs_search --data-root data --device cuda --resume --skip-completed
```

## Structure

```text
codes/VGG_BatchNorm/
  models/
  utils/
  run_project2.py
  requirements.txt

setup_project2_env.sh
run_project2_experiments.sh
```

## Notes

- This GitHub repository stores **code only**.
- Datasets, checkpoints, figures, and full experiment outputs are not included.
- Final report assets and model weights should be shared separately.

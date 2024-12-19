# SMPL experiments

## Files need to be modified before running the code
1. `offline_training.py`:
    - use_gpu: set to `True` if you want to use GPU
    - uncomment code containing 'wandb' if you want to use wandb for logging
2. `offline_experiments.yaml`:
    - N_EPOCHS: number of epochs to train the model
    - DYNAMICS_N_EPOCHS: number of epochs to train the dynamics model
    - training_dataset_loc: location of the training dataset
    - eval_dataset_loc: location of the evaluation dataset
3. `OFFLINE_BEST.yaml`:
    - best_loc: location to save the best model

## Offline Experiments

### Data Generation
To generate the data, run the following command:

```bash
uv run offline_temporal_dataset_generation.py
```

### Training
To train the model, run the following command:

```bash
uv run offline_training.py
```

### Evaluation
To evaluate the model, run the following command:

The 'OFFLINE_BEST.yaml' file should contain the location of the best model. For example:
```yaml
best_loc: "d3rlpy_logs/4042_${timestamp}"
```

```bash
uv run offline_inference.py
```

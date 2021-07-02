# README
This repoistory contains the code for the experiments in the paper "Asymptotics of Network Embeddings Learned via Subsampling" [arXiv]().

## Installation instructions

- Install the relationalERM software from the tensorflow-v2 branch (https://github.com/wooden-spoon/relational-ERM/tree/tensorflow-v2), with a version of Tensorflow with version >= 2.3.
- Place the custom_scripts directory inside of the src directory.

## Experiments

We describe how to reproduce the experiments within the paper. All code extracts are assumed to be called from the command line in the src directory as described above.

### Latent Gaussian recovery experiments:

1. Generate the model data

2. Training on the generated data

### Sampling formula verification experiments:

1. Generate the model data

2. Training on the generated data

### Real data experiments:

Run the following code

```
python -m custom_scripts.learn_embed --train-dir TRAIN_DIR --data-dir DATA_DIR --eval-dir EVAL_DIR --embedding-dim 128 --max-steps 400000 --exotic-evaluation --sampler SAMPLER --num-edges 3000 --label-task-weight 0.01 --embedding_learning_rate 0.01 --num-negative 5 --indef-ip
```

where 

- `TRAIN_DIR` and `EVAL_DIR` are directories for where the training and evaluation results should be saved
- `DATA_DIR` links to either the blogs or homo_sapiens datasets (both of which can be found in the data directory inside this repisitory)
- `SAMPLER` is one of 'p-sampling' or 'biased-walk'
- the flag `--indef-ip` indicates to use a Krein inner product as part of training (remove this flag to only use the regular inner product)

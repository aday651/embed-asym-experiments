# README
This repository contains the code for the experiments in the paper "Asymptotics of Network Embeddings Learned via Subsampling" ([arXiv](https://arxiv.org/abs/2107.02363))

## Installation instructions

- Install the relationalERM software from the tensorflow-v2 branch (https://github.com/wooden-spoon/relational-ERM/tree/tensorflow-v2), with a version of Tensorflow with version >= 2.3.
- Place the custom_scripts directory inside of the src directory.

## Experiments

We describe how to reproduce the experiments within the paper. All code extracts are assumed to be called from the command line in the src directory as described above. Variables which varied across experiment runs are detailed below; the meaning of the other command line arguments can be identified either through the source code or by using `-h` on the command line.

### Latent Gaussian recovery experiments:

#### Step 1: Generate the model data

Run the following code

```
python -m custom_scripts.gen_data --num-vertices NUM_OF_VERTICES --latent-dim DATA_LATENT_DIM --filename DATA_DIR --latent-dgp 'normal' --indef-ip
```

where

- `NUM_OF_VERTICES` is the number of vertices of the generated network
- `DATA_LATENT_DIM` is the dimension of the per vertex latent vectors used to generate the network
- the flag `--indef-ip` indicates to use a Krein inner product - namely, the inner product $$\langle Z, \mathrm{diag}(I_{d/2}, -I_{d/2}) Z' \rangle$$
 where $d$ is `DATA_LATENT_DIM` - as part of generating the model; remove this flag to only use the regular inner product between latent variables
- `DATA_DIR` denotes the save location of the generated data

#### Step 2: Training on the generated data and evaluating the performance

Run the following code

```
python -m custom_scripts.learn_embed --data-dir DATA_DIR --eval-dir EVAL_DIR --embedding-dim MODEL_LATENT_DIM --max-steps MAX_STEPS --sampler p-sampling-induced --num-edges 1000 --embedding_learning_rate 0.01 --indef-ip
```

where

- `DATA_DIR` is as above, and `EVAL_DIR` is the directory to save the evaluation results
- `MODEL_LATENT_DIM` is the dimension of the embedding vectors used to fit the model; the flag `--indef_ip` indicates that we want to train using a Krein inner product as part of training (removing this flag means the regular inner product between embedding vectors is used)
- `MAX_STEPS` is the number of steps used for training (for example, for a 3200 vertex network, we used 102400 steps)

The saved evaluation files then contain the average error between the actual logits used to generate the network, and the predicted logits formed by using the learned embedding vectors.

### Sampling formula verification experiments:

#### Step 1: Generate the model data

Run the following code

```
python -m custom_scripts.gen_data --num-vertices NUM_OF_VERTICES --sbm --sbm-filename SBM_FILENAME --filename DATA_DIR
```

where

- `NUM_OF_VERTICES` is the number of vertices of the generated network
- `SBM_FILENAME` points to either `data/sbm/sbm_1.npz` or `data/sbm/sbm_2.npz`
- `DATA_DIR` denotes the save location of the generated data

#### Step 2: Training on the generated data

Run the following code

```
python -m custom_scripts.learn_embed --embedding_learning_rate 0.01 --embedding-dim 8 --num-edges 1000 --max-steps MAX_STEPS --data-dir DATA_DIR --eval-dir EVAL_DIR --sampler SAMPLER --window-size 1 --indef-ip
```

where

- `DATA_DIR` is as above, and `EVAL_DIR` is the directory to save the evaluation results
- the flag `--indef-ip` indicates to use a Krein inner product between embedding vectors as part of training; remove this flag to only use the regular inner product
- `MAX_STEPS` is the maximum number of iterations (for example, for a network with 3200 vertices we used `MAX_STEPS=204800`)
- `SAMPLER` is one of 'biased-walk' or 'p-sampling-induced', where in the case when `SAMPLER='biased-walk'` we also added the argument `--num-negative 2`

The saved evaluation files then contains the information describing the average error between the actual logits used to generate the network, and the calculated value of the minima of the population risk under different sampling schemes and assumptions of the choice of inner product used between embedding vectors as part of training.

### Real data experiments:

Run the following code

```
python -m custom_scripts.learn_embed --train-dir TRAIN_DIR --data-dir DATA_DIR --eval-dir EVAL_DIR --embedding-dim 128 --max-steps MAX_STEPS --exotic-evaluation --sampler SAMPLER --num-edges 3000 --label-task-weight 0.01 --embedding_learning_rate 0.01 --num-negative 5 --indef-ip
```

where 

- `TRAIN_DIR` and `EVAL_DIR` are directories for where the training and evaluation results should be saved
- `DATA_DIR` links to either the blogs or homo_sapiens datasets (both of which can be found in the data directory inside this repository)
- `SAMPLER` is one of 'p-sampling' or 'biased-walk'
- the flag `--indef-ip` indicates to use a Krein inner product between embedding vectors as part of training; remove this flag to only use the regular inner product
- `MAX_STEPS` was set to 100000 for the homo-sapiens experiments, and 400000 for the blogs data set experiments

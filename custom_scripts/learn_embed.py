import argparse
import tensorflow as tf
import numpy as np
import json
import itertools
import os
from pathlib import Path

from relational_erm.sampling import adapters, factories
from custom_scripts.load_data import load_graph_data
from custom_scripts.skipgram_modified import make_skipgram, make_multilabel_logistic_regression

# class to hanle jsonifying numpy datatypes
# https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parse_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=np.random.randint(2**30))
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--train-dir', type=str, default=None)
    parser.add_argument('--eval-dir', type=str, default=None)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--proportion-censored', type=float, default=0.5,
                        help='Prop of censored vertex labels at train time.')
    parser.add_argument('--label-task-weight', type=float, default=1e-30,
                        help='Weight to assign to label task.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Minibatch size')
    parser.add_argument('--dataset-shards', type=int, default=None,
                        help='Dataset parallelism')
    # parser.add_argument('--use-xla', action='store_true',
    #                    help='Use XLA JIT compilation')
    parser.add_argument('--exotic-evaluation',
                        action='store_true', help='perform exotic evaluation.')

    parser.add_argument('--sampler', type=str, default=None,
                        choices=factories.dataset_names(),
                        help='The sampler to use. biased-walk gives a skipgram random-walk with unigram negative sampling; p-sampling gives p-sampling with unigram negative sampling; uniform-edge gives uniform edge sampling with unigram negative sampling; biased-walk-induced-uniform gives induced random-walk with unigram negative-sampling; p-sampling-induced gives p-sampling with induced non-edges.')

    parser.add_argument('--sampler-test', type=str, default=None,
                        choices=factories.dataset_names(),
                        help='if not None, the sampler to use for testing')

    # defaults set to match Node2Vec
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--indef-ip', default=False, action='store_true',
                        help='Uses a krein inner product instead of the regular inner product.')

    parser.add_argument('--num-edges', type=int, default=800,
                        help='Number of edges per sample. Note the walk length is equal to the number of edges divided by the window size for the skipgram. This also equals the average number of edges in a given sample for p-sampling, where p is calculated as the square root of (num-edges/number of edges in the graph.')

    parser.add_argument('--p-sample-prob', type=float, default=None,
                        help='Probability of samping a vertex for p-sampling. Only used if the sampling scheme is a p-sampling scheme, in which case this is used to override the num-edges argument.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Window size for Skipgram. Default is 10.')

    parser.add_argument('--num-negative', type=int, default=5,
                        help='Negative examples per vertex for negative sampling.')

    parser.add_argument('--num-negative-total', type=int, default=None,
                        help='Total number of negative vertices sampled.')

    parser.add_argument('--embedding_learning_rate', type=float, default=0.025,
                        help='SGD learning rate for embedding updates.')

    parser.add_argument('--global_learning_rate', type=float, default=1.,
                        help='SGD learning rate for global updates.')

    parser.add_argument('--global_regularization', type=float, default=1.,
                        help='Regularization scale for global variables.')

    return parser.parse_args()


def get_dataset_fn(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def sbm_eval(embeds, latents, block_matrix):
    block_num = block_matrix.shape[0]
    n = embeds.shape[0]
    local_eval = np.zeros((block_num, block_num))

    for i, j in itertools.product(np.arange(block_num), repeat=2):
        sub_embed = embeds[np.ix_(latents == i, latents == j)]
        local_eval[i, j] = np.sum(np.abs(sub_embed - block_matrix[i, j]))

    return(np.sum(local_eval)/(n**2))


def make_input_fn(graph_data, args, dataset_fn=None, num_samples=None):
    def input_fn():
        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),
            adapters.append_vertex_labels(graph_data.labels),
            adapters.split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed)),
            adapters.add_sample_size_info(),
            adapters.format_features_labels())

        dataset = dataset.map(data_processing, 4)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        if args.batch_size is not None:
            dataset = dataset.apply(
                adapters.padded_batch_samples(args.batch_size))

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    return input_fn


def _adjust_learning_rate(learning_rate, batch_size):
    if batch_size is not None:
        return learning_rate * batch_size

    return learning_rate


def _adjust_regularization(regularization, batch_size):
    if batch_size is not None:
        return regularization / batch_size

    return regularization


def _make_global_optimizer(args):
    def fn():
        learning_rate = args.global_learning_rate
        return tf.compat.v1.train.GradientDescentOptimizer(
            _adjust_learning_rate(learning_rate, args.batch_size))
    return fn


def make_n2v_test_dataset_fn(args, graph_data):
    rng = np.random.RandomState(args.seed)
    in_train = rng.binomial(1, 1 - args.proportion_censored,
                            size=graph_data.num_vertices).astype(np.int32)

    pred_features = {'vertex_index': np.expand_dims(np.array(range(graph_data.num_vertices)), 1),
                     'is_positive': np.expand_dims(np.array(range(graph_data.num_vertices)), 1)}
    pred_labels = {'labels': np.expand_dims(graph_data.labels, 1),
                   'split': np.expand_dims(in_train, 1)}

    def n2v_test_dataset_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (pred_features, pred_labels))
        return dataset

    return n2v_test_dataset_fn


def main():
    # Set logging verbosity, parse arguments
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    args = parse_arguments()

    # Loading data
    # note: just for graphon data for now, will do for SBM later
    graph_data, sbm_flag, latent_dgp = load_graph_data(args.data_dir)

    # Override num of edges if scheme is p-sampling
    if args.sampler is not None:
        if ("p-sampling" in args.sampler) and args.p_sample_prob is not None:
            args.num_edges = int((args.p_sample_prob**2)
                                 * np.size(graph_data.adjacency_list))

        if "induced" in args.sampler:
            args.num_negative = None

    sg_model = make_multilabel_logistic_regression(
        label_task_weight=args.label_task_weight,
        regularization=_adjust_regularization(
            args.global_regularization, args.batch_size),
        global_optimizer=_make_global_optimizer(args),
        embedding_optimizer=lambda: tf.compat.v1.train.GradientDescentOptimizer(
            _adjust_learning_rate(args.embedding_learning_rate, args.batch_size)),
        polyak=False)

    # Setting up some other parameters/config stuff
    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': True,
        'embedding_checkpoint': None
    }

    session_config = tf.compat.v1.ConfigProto()

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1000, session_config=session_config)

    node_classifier = tf.estimator.Estimator(
        model_fn=sg_model,
        params={
            **vertex_embedding_params,
            'n_labels': graph_data.num_labels,
            'num_vertices': graph_data.num_vertices,
            'batch_size': args.batch_size,
            'indef_ip': args.indef_ip
        },
        model_dir=args.train_dir,
        config=run_config)

    hooks = [
        tf.estimator.LoggingTensorHook({
            'kappa_insample': 'kappa_insample_batch/value',
            'kappa_outsample': 'kappa_outsample_batch/value',
            'kappa_edges': 'kappa_edges_in_batch/value'},
            every_n_iter=100)
    ]

    # Define a tf.estimator.Estimator object using the model function,
    # the above parameters
    if args.profile:
        hooks.append(tf.estimator.ProfilerHook(save_secs=30))

    if args.debug:
        from tensorflow.python import debug as tfdbg
        hooks.append(tfdbg.TensorBoardDebugHook('localhost:6004'))

    dataset_fn_train = get_dataset_fn(args.sampler, args)

    node_classifier.train(
        input_fn=make_input_fn(graph_data, args, dataset_fn_train),
        max_steps=args.max_steps,
        hooks=hooks)

    # Evaluation code
    eval_dict = {}

    if sbm_flag is not None:
        # Load variables
        embeddings = node_classifier.get_variable_value(
            'input_layer/vertex_index_embedding/embedding_weights')

        if args.indef_ip:
            pm_diag = np.concatenate((np.ones(int(args.embedding_dim/2)),
                                      -1*np.ones(int(args.embedding_dim/2))), axis=None)
            pm_diag = np.diag(pm_diag)
        else:
            pm_diag = np.diag(np.ones(args.embedding_dim))

        embed_gram = np.matmul(embeddings, np.matmul(pm_diag, embeddings.T))

        if sbm_flag:
            # If we have a SBM, compare this to different functions of the
            # underlying SBM, including what we expected given the
            # hyperparameters of the model
            methods = ['unif_reg_ip', 'unif_krein_ip', 'rw_reg_ip', 'rw_krein_ip']
            for meth in methods:
                eval_dict[meth] = sbm_eval(embed_gram,
                                           graph_data.latents,
                                           getattr(graph_data, meth))
        else:
            # Compute the avg absolute error between the gram matrix
            # formed by the learned embeddings and the true underlying latents
            # Save data latent dimension, method type
            eval_dict["latent_dim"] = np.shape(graph_data.latents)[1]
            latent_dim = int(graph_data.latents.shape[1])

            if graph_data.indef_ip:
                latent_pm_diag = np.concatenate((np.ones(int(latent_dim/2)),
                                                 -1*np.ones(int(latent_dim/2))), axis=None)
                latent_pm_diag = np.diag(latent_pm_diag)
            else:
                latent_pm_diag = np.diag(np.ones(latent_dim))

            latent_gram = np.matmul(graph_data.latents, np.matmul(
                latent_pm_diag, graph_data.latents.T))

            eval_dict["error"] = np.mean(np.abs(embed_gram - latent_gram))
    else:
        eval_dict["node2vec_eval"] = node_classifier.evaluate(
            input_fn=make_n2v_test_dataset_fn(args, graph_data),
            name="node2vec_eval")

        if args.exotic_evaluation:
            samplers = factories.dataset_names()
            for sampler in samplers:
                dataset_fn_test = get_dataset_fn(sampler, args)

                eval_dict[sampler] = node_classifier.evaluate(
                    input_fn=make_input_fn(graph_data, args, dataset_fn_test, 2000),
                    name=sampler)

    eval_dict["args"] = vars(args)
    eval_dict["num_vertices"] = graph_data.num_vertices
    eval_dict["data_indef_ip"] = graph_data.indef_ip

    if latent_dgp is not None:
        eval_dict["latent_dgp"] = latent_dgp

    # Save eval_dict somewhere
    if args.eval_dir is not None:
        # Create directory if needed
        directory = os.path.dirname(args.eval_dir)
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Handle class conversions to save dict as json
        eval_json = json.dumps(
            eval_dict, cls=NumpyEncoder, sort_keys=True)

        with open(args.eval_dir, "a+") as f:
            json.dump(eval_json, f)


if __name__ == '__main__':
    main()

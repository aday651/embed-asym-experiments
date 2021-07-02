"""
Example script for training and evaluating relational ERM model
"""

import json
import itertools
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from relational_erm.sampling import factories

from node_classification_with_features.dataset_logic.dataset_logic import load_data_graphsage

from node_classification_with_features.sample import make_sample, augment_sample
from custom_scripts.predictor_class_and_losses import make_nn_class_predictor

# helpers for batching
from custom_scripts.learn_embed import parse_arguments, \
    _adjust_regularization, _adjust_learning_rate, _make_global_optimizer


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


def main():
    # tf.enable_eager_execution()
    #
    # # create fake args for debugging
    # sys.argv = ['']
    # args = parse_arguments()
    # args.batch_size = 10
    # args.max_steps = 5000
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    args = parse_arguments()

    graph_data = load_data_graphsage(args.data_dir)

    """
    Sample defined as a custom tf dataset
    """
    sample_train = make_sample(args.sampler, args)  # sample subgraph according to graph sampling scheme args.sampler
    input_fn = augment_sample(graph_data, args, sample_train)  # augment subgraph with vertex labels and features

    """
    Predictor class and loss function
    """
    # hyperparams
    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': True,
        'embedding_checkpoint': None
    }

    params={
        **vertex_embedding_params,
        'hidden_units' : [200, 200], # Jaan net
        'n_classes':  max(graph_data.classes)+1,
        'num_vertices': graph_data.num_vertices,
        'batch_size': args.batch_size,
        'indef_ip': args.indef_ip
    }

    classifier_predictor_and_loss = make_nn_class_predictor(
        label_task_weight=args.label_task_weight,
        regularization=_adjust_regularization(args.global_regularization, args.batch_size),
        global_optimizer=_make_global_optimizer(args),
        embedding_optimizer=lambda: tf.compat.v1.train.GradientDescentOptimizer(
            _adjust_learning_rate(args.embedding_learning_rate, args.batch_size))
    )

    node_classifier = tf.estimator.Estimator(
        model_fn=classifier_predictor_and_loss,
        params=params,
        model_dir=args.train_dir)


    """
    Put it together for the optimization
    """
    # some extra logging
    hooks = [
        tf.estimator.LoggingTensorHook({
            'kappa_edges': 'kappa_edges_in_batch/value'},
            every_n_iter=100)
    ]

    if args.profile:
        hooks.append(tf.estimator.ProfilerHook(save_secs=30))

    if args.debug:
        from tensorflow.python import debug as tfdbg
        hooks.append(tfdbg.TensorBoardDebugHook('localhost:6004'))

    node_classifier.train(
        input_fn=input_fn,
        max_steps=args.max_steps,
        hooks=hooks)


    """
    Evaluate
    """
    eval_dict = {}
    eval_dict["node2vec_eval"] = node_classifier.evaluate(input_fn=augment_sample(graph_data, args, sample_train, 2000),
                                                          name="node2vec_eval")

    for sampler in ["biased-walk", "p-sampling"]:
        sample_test = make_sample(sampler, args)
        eval_dict[sampler] = node_classifier.evaluate(
            input_fn=augment_sample(graph_data, args, sample_test, 2000),
            name=sampler)

    # code using in learn_embed, add to node_classification_template.py
    # in node_classification with features)
    # TODO: need to fix the macro F1 code....

    #File "/mnt/c/Users/adavi/Documents/Research/relational-ERM/src/node_classification_with_features/node_classification_template.py", line 33, in _make_metrics
#     macro_f1 = metrics.macro_f1(
#   File "/mnt/c/Users/adavi/Documents/Research/relational-ERM/src/relational_erm/models/metrics.py", line 68, in macro_f1
#     true_positives = tf.compat.v1.get_variable('true_positives', shape=num_classes, dtype=tf.int64,
#   File "/home/ad/miniconda3/envs/tensorflow/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 1556, in get_variable
#     return get_variable_scope().get_variable(
#   File "/home/ad/miniconda3/envs/tensorflow/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 1299, in get_variable
#     return var_store.get_variable(
#   File "/home/ad/miniconda3/envs/tensorflow/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 554, in get_variable
#     return _true_getter(
#   File "/home/ad/miniconda3/envs/tensorflow/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 507, in _true_getter
#     return self._get_single_variable(
#   File "/home/ad/miniconda3/envs/tensorflow/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 919, in _get_single_variable
#     raise ValueError("The initializer passed is not valid. It should "
# ValueError: The initializer passed is not valid. It should be a callable with no arguments and the shape should not be provided or an instance of `tf.keras.initializers.*' and `shape` should be fully defined.
    #samplers = factories.dataset_names()
    #for sampler in samplers:
    #    dataset_fn_test = make_sample(sampler, args)

    #    eval_dict[sampler] = node_classifier.evaluate(
    #                input_fn=augment_sample(graph_data, args, dataset_fn_test, 2000),
    #                name=sampler)

    eval_dict["args"] = vars(args)

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

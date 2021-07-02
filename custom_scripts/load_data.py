import numpy as np
import tensorflow as tf

from collections import namedtuple

from relational_erm.graph_ops.representations import create_packed_adjacency_list, edge_list_to_adj_list, PackedAdjacencyList

GraphDataN2V = namedtuple('GraphDataN2V', ['edge_list',
                                           'weights',
                                           'labels',
                                           'latents',
                                           'adjacency_list',
                                           'num_vertices',
                                           'num_labels',
                                           'sbm_probs',
                                           'sbm_matrix',
                                           'unif_reg_ip',
                                           'unif_krein_ip',
                                           'rw_reg_ip',
                                           'rw_krein_ip',
                                           'indef_ip'])


def load_graph_data(data_path=None):
    """
    Loads data
    """
    if data_path is None:
        raise ValueError('Data path needs to be specified')

    # Use tensorflow loading
    with tf.io.gfile.GFile(data_path, mode='rb') as f:
        loaded = np.load(f, allow_pickle=False)

    # Extracting the relevant parts from the stored data
    edge_list = loaded['edge_list'].astype(np.int32)

    if 'weights' in loaded:
        weights = loaded['weights'].astype(np.float32)
    else:
        weights = np.ones(edge_list.shape[0], dtype=np.float32)

    # Remove self-edges
    not_self_edge = edge_list[:, 0] != edge_list[:, 1]
    edge_list = edge_list[not_self_edge, :]
    weights = weights[not_self_edge]

    if 'latents' in loaded:
        latents = loaded['latents'].astype(np.float32)
    else:
        latents = None

    if 'neighbours' in loaded:
        neighbours = loaded['neighbours']
        lengths = loaded['lengths']

        offsets = np.empty_like(lengths)
        np.cumsum(lengths[:-1], out=offsets[1:])
        offsets[0] = 0

        adjacency_list = PackedAdjacencyList(neighbours, None, offsets, lengths, np.arange(len(lengths)))

        num_vertices = len(lengths)
    else:
        adjacency_list = edge_list_to_adj_list(edge_list, weights)
        num_vertices = len(adjacency_list)
        adjacency_list = create_packed_adjacency_list(adjacency_list)

    if 'sbm' in loaded:
        if loaded['sbm']:
            # Make one hot representation of the labels cause why not
            labels = np.zeros((loaded['latents'].size, 
                               loaded['sbm_probs'].size),
                               dtype=np.float32)
            labels[np.arange(loaded['latents'].size), loaded['latents']] = 1
        else:
            labels = np.array([np.ones(num_vertices),
                               np.zeros(num_vertices)]).astype(np.float32)
            labels = np.transpose(labels)
    elif 'classes' in loaded:
        classes = loaded['classes'].astype(np.int32)
        num_classes = np.amax(classes) + 1
        labels = np.zeros((classes.size, num_classes), dtype=np.float32)
        labels[np.arange(classes.size), classes] = 1
    else:
        labels = loaded['group'].astype(np.float32)

    num_labels = labels.shape[1]

    sbm_probs = loaded['sbm_probs'].astype(np.float32) if 'sbm_probs' in loaded else None
    sbm_matrix = loaded['sbm_matrix'].astype(np.float32) if 'sbm_matrix' in loaded else None

    if 'unif_reg_ip' in loaded:
        unif_reg_ip = loaded['unif_reg_ip'].astype(np.float32)
        unif_krein_ip = loaded['unif_krein_ip'].astype(np.float32)
        rw_reg_ip = loaded['rw_reg_ip'].astype(np.float32)
        rw_krein_ip = loaded['rw_krein_ip'].astype(np.float32)
    else:
        unif_reg_ip, unif_krein_ip, rw_reg_ip, rw_krein_ip = [None]*4

    indef_ip = loaded['indef_ip'] if 'indef_ip' in loaded else None

    # Values to be passed not as part of the graph data object as for some reason
    # tensorflow gets unhappy about theses
    sbm_flag = loaded['sbm'] if 'sbm' in loaded else None
    latent_dgp = str(loaded['latent_dgp']) if 'latent_dgp' in loaded else None

    return GraphDataN2V(edge_list=edge_list,
                        weights=weights,
                        labels=labels,
                        latents=latents,
                        adjacency_list=adjacency_list,
                        num_vertices=num_vertices,
                        num_labels=num_labels,
                        sbm_probs=sbm_probs,
                        sbm_matrix=sbm_matrix,
                        unif_reg_ip=unif_reg_ip,
                        unif_krein_ip=unif_krein_ip,
                        rw_reg_ip=rw_reg_ip,
                        rw_krein_ip=rw_krein_ip,
                        indef_ip=indef_ip), sbm_flag, latent_dgp

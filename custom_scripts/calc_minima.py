import itertools
import matplotlib
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# colorbar fix taken from https://joseph-long.com/writing/colorbars/
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def pop_risk_constrained_optim(sbm_probs, samp_formula_edge, samp_formula_non_edge):
    """
    Returns the constrained global minima to the population risk for
    a SBM, given the community sizes (sbm_probs) and the sampling formula for
    a pair of vertices (either an edge or a non-edge) in different communities.
    """
    k = sbm_probs.size
    samp_formula_total = samp_formula_edge + samp_formula_non_edge

    K = cp.Variable(shape=(k, k), symmetric=True)
    constraints = [K >> 0]

    element_wise_loss = (cp.multiply(samp_formula_total, cp.logistic(K)) -
                         cp.multiply(samp_formula_edge, K))
    loss_fn = cp.quad_form(sbm_probs, element_wise_loss)

    prob = cp.Problem(cp.Minimize(loss_fn), constraints)
    prob.solve()

    return K.value


def pop_risk_optim(sbm_probs, samp_formula_edge, samp_formula_non_edge):
    """
    Returns the unconstrained global minima of the population risk
    for a SBM, given the community sizes (although this doesn't play a
    role here) and the sampling formula for edges and non-edges.
    """
    return np.log(samp_formula_edge/samp_formula_non_edge)


def degree_fn(sbm_probs, sbm_matrix):
    """
    Returns the degree function for a SBM with community sizes given by
    sbm_probs and edge probabilities between vertices in different
    communities given by sbm_matrix.
    """
    return np.matmul(sbm_probs, sbm_matrix)


def edge_density_fn(sbm_probs, sbm_matrix, alpha=1):
    """
    Returns the integrated degree function (where the degree is
    raised to the power of alpha) for a SBM with community sizes given by
    sbm_probs and edge probabilities between vertices in different
    communities given by sbm_matrix.
    """
    if (alpha <= 0):
        raise ValueError("alpha must be a positive number")

    degree = degree_fn(sbm_probs, sbm_matrix)

    return np.inner(sbm_probs, degree**alpha)


def unif_vertex_formula(sbm_probs, sbm_matrix, params):
    """
    Returns the sampling formula (up to constant scaling which is
    identical for both formulae) for uniform vertex sampling.
    """
    samp_formula_edge = sbm_matrix
    samp_formula_non_edge = 1 - sbm_matrix

    return samp_formula_edge, samp_formula_non_edge


def unif_edge_ns_formula(sbm_probs, sbm_matrix, params):
    """
    Returns the sampling formula for uniform edge sampling + unigram
    negative sampling.
    """
    k = params["num_edges"]
    l = params["num_negative"]
    alpha = params["alpha"]
    ns_non_edge = params["ns_non_edge"]

    edge_density = edge_density_fn(sbm_probs, sbm_matrix)
    edge_density_alpha = edge_density_fn(sbm_probs, sbm_matrix, alpha)
    degree = degree_fn(sbm_probs, sbm_matrix)

    samp_formula_edge = 2*k*sbm_matrix/edge_density
    samp_formula_non_edge = 2*k*l*(np.outer(degree, degree**alpha) +
                                                    np.outer(degree**alpha, degree))/(edge_density*edge_density_alpha)

    if ns_non_edge:
        samp_formula_non_edge = samp_formula_non_edge*(1 - sbm_matrix)

    return samp_formula_edge, samp_formula_non_edge


def unif_edge_ind_formula(sbm_probs, sbm_matrix, params):
    """
    Returns the sampling formula for uniform edge sampling + induced
    subgraph negative sampling.
    """
    k = params["num_edges"]
    degree = degree_fn(sbm_probs, sbm_matrix)
    edge_density = edge_density_fn(sbm_probs, sbm_matrix)

    induced_term = 4*k*(k-1)*np.outer(degree, degree)/(edge_density**2)

    samp_formula_edge = (4*k/edge_density + induced_term)*sbm_matrix
    samp_formula_non_edge = induced_term*(1 - sbm_matrix)

    return samp_formula_edge, samp_formula_non_edge


def rw_ns_formula(sbm_probs, sbm_matrix, params):
    """
    Returns the sampling formula for random walk sampling + unigram
    negative sampling.
    """
    k = params["num_edges"]
    l = params["num_negative"]
    alpha = params["alpha"]
    ns_non_edge = params["ns_non_edge"]

    edge_density = edge_density_fn(sbm_probs, sbm_matrix)
    edge_density_alpha = edge_density_fn(sbm_probs, sbm_matrix, alpha)
    degree = degree_fn(sbm_probs, sbm_matrix)

    samp_formula_edge = 2*k*sbm_matrix/edge_density
    samp_formula_non_edge = l*(k+1)*(np.outer(degree, degree **alpha) + np.outer(degree**alpha, degree))/(edge_density*edge_density_alpha)

    if ns_non_edge:
        samp_formula_non_edge = samp_formula_non_edge*(1 - sbm_matrix)

    return samp_formula_edge, samp_formula_non_edge


def skipgram_formula(sbm_probs, sbm_matrix, params):
    """
    Returns the sampling formula for using a skipgram sampler
    based from a random walk sampler on the underlying graph,
    plus unigram negative sampling.
    """
    def _int(probs, matrix):
        return np.matmul(probs, matrix)

    def _int_gram(mat1, mat2, probs, weights):
        """
        Given mat1, mat2 and weights, returns the matrix
        W(i, j) = sum_{k} mat1(i, k)*mat2(k, j)*probs(k)*weights(k).
        """
        diag = np.diag(probs*weights)
        return np.matmul(mat1, np.matmul(diag, mat2))

    # Extract parameters
    k = params["num_edges"]/params["window_length"]
    l = params["num_negative"]
    alpha = params["alpha"]
    ns_non_edge = params["ns_non_edge"]
    window_length = params["window_length"]

    # Get degree function and edge density
    edge_density = edge_density_fn(sbm_probs, sbm_matrix)
    edge_density_alpha = edge_density_fn(sbm_probs, sbm_matrix, alpha)
    degree = degree_fn(sbm_probs, sbm_matrix)
    weights = 1/degree
   
    # Form the contributions for the edge part of the formula
    samp_formula_edge = np.zeros_like(sbm_matrix)

    for i in range(window_length):
        # Get the value of the i-th contribution
        if i == 0:
            part_sum = sbm_matrix
        else:
            part_sum = _int_gram(part_sum, sbm_matrix, sbm_probs, weights)

        samp_formula_edge = samp_formula_edge + 2*(k - i)*part_sum/edge_density

    # Form the contribution for the negative sample part of the formula
    samp_formula_non_edge = l*(k+1)*(np.outer(degree, degree **alpha) + np.outer(degree**alpha, degree))/(edge_density*edge_density_alpha)

    if ns_non_edge:
        samp_formula_non_edge = samp_formula_non_edge*(1 - sbm_matrix)

    return samp_formula_edge, samp_formula_non_edge


def sigmoid(x):
    return 1/(1+np.exp(-x))


def plot_block_fn(block_probs, block_matrix):
    """
    Visualizes a SBM with community sizes block_probs and within/between
    community probabilities given by block_matrix.
    """
    vertex = np.pad(np.cumsum(block_probs), (1, 0), 'constant')
    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.pcolormesh(vertex, vertex,
                       block_matrix, shading='flat', vmin=0.0, vmax=1.0)

    colorbar(im)
    ax.set_aspect('equal')

    plt.show()


def calc_optima_dict(sbm_probs, sbm_matrix, params=None):
    """
    Given a SBM with community sizes sbm_probs and transition matrix
    sbm_matrix, returns the minima of the population objective when using:
        - uniform vertex sampling and a regular inner product
        - uniform vertex sampling and a krein inner product
        - random walk + unigram ns and a regular inner product
        - random walk + unigram ns and a krein inner product
    """
    if params is None:
        params = {'num_edges': 100, 'num_negative': 5,
                  'alpha': 0.75, 'num_vertex': 2, 'ns_non_edge': True,
                  'window_length': 10}

    formula_list = [unif_vertex_formula, rw_ns_formula, skipgram_formula]
    opt_list = [pop_risk_constrained_optim, pop_risk_optim]
    formula_str = ["unif", "rw", "skipgram"]
    opt_str = ["_reg_ip", "_indef_ip"]

    optima_dict = {}

    for i, j in itertools.product(range(3), range(2)):
        edge_form, non_edge_form = formula_list[i](
            sbm_probs, sbm_matrix, params)
        optima = opt_list[j](sbm_probs, edge_form, non_edge_form)

        dict_str = formula_str[i] + opt_str[j]
        optima_dict[dict_str] = optima

    return optima_dict


def plot_block_fn_grid(sbm_probs, sbm_matrix, params=None):
    """
    Given a SBM with community sizes sbm_probs and transition matrix
    sbm_matrix, visualizes the learned SBMs using:
        - uniform vertex sampling and a regular inner product
        - uniform vertex sampling and a krein inner product
        - random walk + unigram ns and a regular inner product
        - random walk + unigram ns and a krein inner product
    """
    if params is None:
        params = {'num_edges': 100, 'num_negative': 5,
                  'alpha': 0.75, 'num_vertex': 2, 
                  'ns_non_edge': True, 'window_length': 10}

    formula_list = [unif_vertex_formula, rw_ns_formula, skipgram_formula]
    opt_list = [pop_risk_constrained_optim, pop_risk_optim]
    title_formula = ['Uniform vertex sampling', 
                     'Random walk + unigram ns',
                     'Skipgram + unigram ns']
    ip_formula = ['Using regular inner product', 'Using krein inner product']

    vertex = np.pad(np.cumsum(sbm_probs), (1, 0), 'constant')

    fig, axs = plt.subplots(2, 2,
                            sharex=True,
                            sharey=True,
                            figsize=(10, 8))

    # Loop over i (rows) gives the sampling scheme, loop over j (columns)
    # gives unconstrained
    for i, j in itertools.product(range(2), range(2)):
        edge_form, non_edge_form = formula_list[i](
            sbm_probs, sbm_matrix, params)
        optima = opt_list[j](sbm_probs, edge_form, non_edge_form)
        im = axs[i, j].pcolormesh(vertex, vertex, sigmoid(optima),
                                  shading='flat', vmin=0.0, vmax=1.0,
                                  cmap='viridis')
        colorbar(im)
        axs[i, j].set_aspect('equal')
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

    for ax, col in zip(axs[0], ip_formula):
        ax.set_title(col)

    for ax, row in zip(axs[:, 0], title_formula):
        ax.set_ylabel(row, rotation=90, size='large')

    plt.show()


def calc_gap_between_optima(sbm_probs, sbm_matrix, params=None):
    """
    Given a SBM with community sizes sbm_probs and transition matrix
    sbm_matrix, calculates the pairwise L1 error between the learned SBMs using:
        - uniform vertex sampling and a regular inner product
        - uniform vertex sampling and a krein inner product
        - random walk + unigram ns and a regular inner product
        - random walk + unigram ns and a krein inner product
    """
    def L1_error(probs, mat1, mat2):
        return np.dot(probs.T, np.dot(np.abs(mat1 - mat2), probs))

    if params is None:
        params = {'num_edges': 50, 'num_negative': 5,
                  'alpha': 0.75, 'num_vertex': 2, 'ns_non_edge': True,
                  'window_length': 10}

    formula_list = [unif_vertex_formula, rw_ns_formula, skipgram_formula]
    opt_list = [pop_risk_constrained_optim, pop_risk_optim]

    optima_list = []
    for i, j in itertools.product(range(3), range(2)):
        edge_form, non_edge_form = formula_list[i](
            sbm_probs, sbm_matrix, params)
        optima_list.append(opt_list[j](sbm_probs, edge_form, non_edge_form))

    distances = np.zeros((6, 6))
    for i, j in itertools.product(range(6), repeat=2):
        distances[i, j] = L1_error(sbm_probs, optima_list[i], optima_list[j])

    print('Row/column 0: Uniform vertex sampling with regular IP')
    print('Row/column 1: Uniform vertex sampling with krein IP')
    print('Row/column 2: Random walk + unigram NS with regular IP')
    print('Row/column 3: Random walk + unigram NS with krein IP')
    print('Row/column 4: Skipgram + unigram NS with regular IP')
    print('Row/column 5: Skipgram + unigram NS with krein IP')
    print(np.around(distances, 4))

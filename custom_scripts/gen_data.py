import numpy as np
import pathlib
import os
import argparse
from pathlib import Path


def parse_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--num-vertices', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=4)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--latent-dgp', type=str, default='normal',
                        choices=['normal', 'normal-lin', 'normal-exp',
                                 'uniform', 'uniform-lin', 'uniform-exp'])
    parser.add_argument('--sbm', default=False, action='store_true')
    parser.add_argument('--indef-ip', default=False, action='store_true')
    parser.add_argument('--sbm-filename', type=str)

    return parser.parse_args()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def gen_graphon_data(args):
    # Create random latent variables
    if ('uniform' in args.latent_dgp):
        latents = np.random.uniform(-1, 1, 
                                    size=(args.num_vertices, args.latent_dim))
    elif ('normal' in args.latent_dgp):
        latents = np.random.normal(size=(args.num_vertices, args.latent_dim))

    if ('lin' in args.latent_dgp):
        scale_fac = np.sqrt(1/(1 + np.arange((int(args.latent_dim/2)))))
        scale_fac = np.concatenate((scale_fac, scale_fac), axis=None)
        latents = np.matmul(latents, np.diag(scale_fac))
    elif ('exp' in args.latent_dgp):
        scale_fac = np.sqrt(np.exp2(-0.5*np.arange(int(args.latent_dim/2))))
        scale_fac = np.concatenate((scale_fac, scale_fac), axis=None)
        latents = np.matmul(latents, np.diag(scale_fac))

    # Generate the edge list of the graph
    if args.indef_ip is True:
        pm_diag = np.concatenate((np.ones(int(args.latent_dim/2)),
                                  -1*np.ones(int(args.latent_dim/2))), axis=None)
        pm_diag = np.diag(pm_diag)

        probs = sigmoid(np.matmul(latents, np.matmul(pm_diag, latents.T)))
    else:
        probs = sigmoid(np.matmul(latents, latents.T))

    unifs = np.random.uniform(size=(args.num_vertices, args.num_vertices))
    edge_list = np.argwhere(unifs < probs)
    edge_list = edge_list[edge_list[:, 0] < edge_list[:, 1], :]
    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))

    # Save or return the latents and the edge list
    if args.filename is None:
        print("Generated graph returned.")
        return latents, edge_list
    else:
        directory = os.path.dirname(args.filename)
        Path(directory).mkdir(parents=True, exist_ok=True)

        np.savez_compressed(args.filename,
                            edge_list=edge_list,
                            latents=latents,
                            latent_dgp=args.latent_dgp,
                            sbm=False,
                            indef_ip=args.indef_ip)
        print("Generated graph saved.")


def gen_sbm_data(sbm_data, args):
    sbm_probs = sbm_data['sbm_probs']
    sbm_matrix = sbm_data['sbm_matrix']

    # Create community assignments
    latents = np.random.choice(sbm_probs.size, size=args.num_vertices,
                               p=sbm_probs)

    # Create matrix of community probabilities from sbm_matrix
    px = np.tile(latents[np.newaxis].T, (1, args.num_vertices))
    py = np.tile(latents, (args.num_vertices, 1))
    probs = sbm_matrix[px, py]
    
    # Get the edges
    unifs = np.random.uniform(size=(args.num_vertices, args.num_vertices))
    edge_list = np.argwhere(unifs < probs)
    edge_list = edge_list[edge_list[:, 0] < edge_list[:, 1], :]
    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    
    # Save or return the latents and the edge list
    if args.filename is None:
        print("Generated graph returned.")
        return latents, edge_list
    else:
        directory = os.path.dirname(args.filename)
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(args.filename,
                            edge_list=edge_list,
                            latents=latents,
                            sbm=True,
                            sbm_probs=sbm_probs,
                            sbm_matrix=sbm_matrix,
                            unif_reg_ip = sbm_data['unif_reg_ip'],
                            unif_krein_ip = sbm_data['unif_indef_ip'],
                            rw_reg_ip = sbm_data['rw_reg_ip'],
                            rw_krein_ip = sbm_data['rw_indef_ip'])
        print("Generated graph saved.")


def main():
    args = parse_arguments()
    if args.sbm:
        sbm_data = np.load(args.sbm_filename)
        gen_sbm_data(sbm_data, args)
    else:
        gen_graphon_data(args)


if __name__ == '__main__':
    main()

""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates.
    However this is not a "must" and you can code however you like.
    You can write helper functions etc however you want.

    If you want, you can use the class structures provided to you (Node and Tree classes in Tree.py
    file), and modify them as needed. In addition to the data files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want.

    For this task, we gave you three different trees (q2_2_small_tree, q2_2_medium_tree, q2_2_large_tree).
    Each tree has 5 samples (the inner nodes' values are masked with np.nan).
    We want you to calculate the likelihoods of each given sample and report it.

    Note:   The alphabet "K" is K={0,1,2,3,4}.

    Note:   A VERY COMMON MISTAKE is to use incorrect order of nodes' values in CPDs.
            theta is a list of lists, whose shape is approximately (num_nodes, K, K).
            For instance, if node "v" has a parent "u", then p(v=Zv | u=Zu) = theta[v][Zu][Zv].

            If you ever doubt your useage of theta, you can double check this marginalization:
            \sum_{k=1}^K p(v = k | u=Zu) = 1
"""

import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: list of numpy arrays. Dimensions (approximately): (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """

    print("\tCalculating the likelihood...")
    likelihood = 0
    K = len(theta[0])
    num_nodes = tree_topology.size
    s_matrix = np.zeros(shape=(num_nodes, K))

    for node, value in reversed(list(enumerate(beta))):
        if not np.isnan(value):  # leaf node
            s_matrix[node][int(value)] = 1

        else:  # inner node
            children = np.argwhere(tree_topology == node).flatten()
            for k in range(K):
                s_matrix[node][k] = 1
                for child in children:
                    s_matrix[node][k] = s_matrix[node][k] * \
                        np.dot(s_matrix[child], theta[child][k])
    # End: Example Code Segment
    likelihood = np.dot(theta[0], s_matrix[0])
    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filenames = ["2_2/data/q2_2_small_tree.pkl",
                 "2_2/data/q2_2_medium_tree.pkl", "2_2/data/q2_2_large_tree.pkl"]
    for f in filenames:
        print('#'*70)
        print("##########\tFilename: ", f, '   ##########')
        print('#'*70)

        t = Tree()
        t.load_tree(f)
        # t.print()
        print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

        print("\n2. Calculate likelihood of each FILTERED sample\n")
        # These filtered samples already available in the tree object.
        # Alternatively, if you want, you can load them from corresponding .txt or .npy files

        for sample_idx in range(t.num_samples):
            beta = t.filtered_samples[sample_idx]
            # print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
            sample_likelihood = calculate_likelihood(
                t.get_topology_array(), t.get_theta_array(), beta)
            print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()

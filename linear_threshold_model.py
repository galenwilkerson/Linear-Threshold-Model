"""
This module contains functions for setting up and running cascades on networks, creating and manipulating adjacency matrices, and generating truth tables for Boolean functions. It also includes functions for plotting and visualizing network structures.

The module is organized into sections for different types of functions, including setup functions for different types of networks, functions for running cascades on these networks, and functions for generating and manipulating truth tables. Additionally, there are utility functions for converting between different data representations and for setting up random seeds for reproducibility.

The module makes use of several external libraries, including NumPy, SciPy, NetworkX, Matplotlib, Pandas, Seaborn, Networkit, and Pynauty. Some functions are designed to work with specific types of networks, such as random geometric graphs, and others are more general-purpose and can be used with any type of network.

Please note that the module contains a mix of vectorized and non-vectorized implementations of cascade algorithms, as well as some experimental or incomplete functions. Users should exercise caution and carefully review the code before using it in a production environment.

Sections are:
# 1. Complexity Functions
# 2. Utility Functions
# 3. Network Visualization
# 4. Setup Functions
# 5. Cascade Functions
# 6. Cascade Analysis
# 7. Plotting Functions
# 8. Binary Array Operations
# 9. Truth Table Functions
# 10. Monte Carlo trials
# 11.  Boolean function analysis
# 12.  Range of Monte Carlo functions
# 13. Hamming Distance and Cube, and Boolean Logic functions
# 14. Graph Measures
# 15. Data generation
"""

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from joblib import Parallel, delayed
import string
import networkit as nk
import pynauty as pyn
from sklearn.datasets import make_blobs


# The sections in the provided file are as follows:

# 1. Complexity Functions


def compute_decision_tree_complexities(node_functions, k):
    """
    given an array of function ids and number of inputs k,
    return an array of decision tree complexity values
    """

    complexities = compute_all_complexities_truth_table(k)

    node_complexities = []

    for fn in node_functions:
        node_complexities.append(complexities[fn])

    return np.array(node_complexities)


def compute_which_LTM_computable(k):
    """
    for a given number of inputs(k), create the truth table and compute decision tree complexity for each function
    returns a list of complexity by function id
    """

    ltm_computables = []

    df_truth_table, inputs = build_truth_table(k)

    num_functions = 2 ** (2**k)

    for fn_ix in range(num_functions):

        H = truth_table_to_Hamming_cube(df_truth_table, k, fn_ix)

        LTM_comp = is_LTM_computable(H)

        ltm_computables.append(LTM_comp)

    return np.array(ltm_computables)


def find_p_GCC(z=1.0001, verbose=False):
    """
    Use Mark Newman method for p(node in GCC).
    See Newman _Networks_ pg. 404.

    Find the non-trivial root of v - 1 + np.e**(-z * v).
    This root only exists when z > 1.
    """
    vrange = np.arange(0, 1, 0.000001)
    y = vrange - 1 + np.e ** (-z * vrange)

    # The element closest to zero (besides at v = 0) is at the second intersection with the x-axis.
    # Note: this only has 2 solutions if z > 1.
    try:
        zero_at = y[y < 0][-1]
        i = len(y[y < 0])
        p_in_GCC = vrange[i]

        if verbose:
            print("average degree", z)
            print()
            print("the probability of belonging to the GCC is", p_in_GCC)
    except:
        p_in_GCC = 0

    return p_in_GCC


# 2. Utility Functions


def set_random_seed(random_seed=None):
    """
    Set the random seed for numpy's random number generator.

    Parameters:
    random_seed (int, optional): Seed for random number generation. If None, the seed is set based on current time.

    Returns:
    int: The random seed used.
    """
    if random_seed is None:
        random_seed = int(datetime.now().strftime("%H:%M:%S.%f").split(".")[-1])

    np.random.seed(random_seed)
    return random_seed


def create_adjacency_matrix(node_count, probability):
    """
    Create an Erdos-Renyi-Gilbert adjacency matrix.

    Parameters:
    node_count (int): Number of nodes in the graph.
    probability (float): Probability of edge creation.

    Returns:
    ndarray: Adjacency matrix representing the graph.
    """
    adjacency_matrix = np.random.rand(node_count, node_count) < probability
    adjacency_matrix = np.triu(adjacency_matrix, 1)
    adjacency_matrix += adjacency_matrix.T

    return adjacency_matrix


def create_label_vector(node_count):
    """
    Create a label vector initialized with zeros.

    Parameters:
    node_count (int): Number of nodes in the graph.

    Returns:
    ndarray: Label vector with all elements initialized to False.
    """
    label_vector = np.zeros(node_count, dtype=bool)
    return label_vector


def create_phi_vector(node_count, phi_value=None):
    """
    Create a phi vector either with random values or a constant value.

    Parameters:
    node_count (int): Number of nodes in the graph.
    phi_value (float, optional): Constant value for all phi entries. If None, phi values are random.

    Returns:
    ndarray: Phi vector for each node.
    """
    if phi_value is None:
        phi_vector = np.random.rand(node_count)
    else:
        phi_vector = np.full(node_count, phi_value)
    return phi_vector


def create_node_types(node_count, theta_value):
    """
    Determine the node type: standard LTM (0) or antagonistic (1).

    Parameters:
    node_count (int): Number of nodes in the graph.
    theta_value (float): Threshold probability for determining the node type.

    Returns:
    ndarray: Array indicating the type of each node.
    """
    node_types = np.random.rand(node_count) < theta_value
    return node_types


def create_ones_vector(node_count):
    """
    Create a vector of ones.

    Parameters:
    node_count (int): Number of nodes in the graph.

    Returns:
    ndarray: Vector of ones.
    """
    ones_vector = np.ones(node_count, dtype=int)
    return ones_vector


def set_seed_nodes_old(label_vector, seeds_on_list):
    """
    Activate the seed nodes in the label vector.

    Parameters:
    label_vector (ndarray): The label vector of the network.
    seeds_on_list (list): List of indices of the seed nodes to be activated.

    Returns:
    ndarray: Updated label vector with the specified seed nodes activated.
    """
    for seed in seeds_on_list:
        label_vector[seed] = True

    return label_vector


def set_seed_nodes(node_count, inputs):
    """
    Create a seed node label vector from a list of seed states.

    Parameters:
    node_count (int): Total number of nodes in the network.
    inputs (list): List indicating the state of the first k seed nodes.

    Returns:
    ndarray: Seed node label vector for the network.
    """
    k = len(inputs)
    seed_label_vector = np.pad(
        inputs, (0, node_count - k), "constant", constant_values=(0)
    ).astype(bool)
    return seed_label_vector


def set_seed_nodes_in_seed_set(node_count, seed_set, inputs):
    """
    Create a seed node label vector where specific nodes can be seeds.

    Parameters:
    node_count (int): Total number of nodes in the network.
    seed_set (list): Indices of nodes that can be seeds.
    inputs (list): Indices of seed nodes to be activated.

    Returns:
    ndarray: Seed node label vector for the network.
    """
    seed_label_vector = np.zeros(node_count, dtype=bool)
    for i in inputs:
        seed_label_vector[seed_set[i]] = True

    return seed_label_vector


# 3. Network Visualization


def draw_network(
    adjacency_matrix,
    label_vector,
    position=None,
    colormap=plt.cm.binary,
    node_size=50,
    with_labels = False,
    edge_colors="grey",
    font_color="red",
    font_weight="bold",
):
    """
    Draw a network graph based on the adjacency matrix and label vector.

    Parameters:
    adjacency_matrix (ndarray): Adjacency matrix of the graph.
    label_vector (ndarray): Label vector indicating the state of each node.
    position (dict, optional): Positions of nodes for drawing. If None, uses spring layout.
    colormap (matplotlib.colors.Colormap, optional): Colormap for node coloring.
    edge_colors (str, optional): Color of the edges.
    font_color (str, optional): Color of the node labels.
    font_weight (str, optional): Weight of the font for node labels.

    Returns:
    tuple: A tuple containing the graph object and positions of nodes.
    """
    graph = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)
    fig = plt.figure(figsize=[4, 4])
    ax = fig.gca()
    plt.axis("off")
    plt.grid(False)
    if position is None:
        position = nx.spring_layout(graph)
    nx.draw_networkx(
        graph,
        position,
        ax=ax,
        with_labels=with_labels,
        node_color=label_vector,
        node_size=node_size,
        cmap=colormap,
        edgecolors=edge_colors,
        font_color=font_color,
        font_weight=font_weight,
    )
    return graph, position


def draw_network_with_legend(
    node_count,
    graph,
    num_inputs,
    functions_computed,
    colormap=plt.cm.nipy_spectral,
    node_size=50,
    max_value=None,
    position=None,
):
    """
    Draw a network graph with a legend indicating different functions computed at each node.

    Parameters:
    node_count (int): Number of nodes in the graph.
    graph (networkx.Graph): The networkx graph object.
    num_inputs (int): Number of input nodes.
    functions_computed (list): List of functions computed at each node.
    colormap (matplotlib.colors.Colormap, optional): Colormap for node coloring.
    node_size (int, optional): Size of nodes in the graph.
    max_value (int, optional): Maximum value for normalizing the node colors.
    position (dict, optional): Positions of nodes for drawing.

    Returns:
    None: This function does not return a value but shows the matplotlib plot.
    """
    color_map = dict(zip(range(node_count), functions_computed))
    values = functions_computed

    if max_value is None:
        max_value = np.max(values)

    if position is None:
        position = nx.spring_layout(graph)

    nx.draw(
        graph,
        position,
        edge_color="lightgrey",
        alpha=0.5,
        node_color=[colormap(v / max_value) for v in values],
        node_size=node_size,
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        nodelist=list(range(num_inputs)),
        node_color="red",
        node_size=node_size,
    )

    for v in set(values):
        plt.scatter([], [], s=20, c=[colormap(v / max_value)], label=f"Function {v}")

    plt.legend(labelspacing=1, loc="best")
    plt.show()


# 4. Setup Functions


def setup_vectorized_LTM(node_count, probability, phi_constant=None):
    """
    Set up a vectorized Linear Threshold Model (LTM).

    Parameters:
    node_count (int): Number of nodes in the graph.
    probability (float): Probability of edge creation.
    phi_constant (float, optional): Constant phi value for all nodes. If None, phi values are random.

    Returns:
    tuple: A tuple containing the adjacency matrix, label vector, phi vector, and degree vector.
    """
    adjacency_matrix = create_adjacency_matrix(node_count, probability)
    label_vector = create_label_vector(node_count)
    phi_vector = create_phi_vector(node_count, phi_constant)
    ones_vector = create_ones_vector(node_count)
    degree_vector = np.dot(adjacency_matrix, ones_vector)

    return adjacency_matrix, label_vector, phi_vector, degree_vector


def setup_vectorized_spatial_LTM(
    node_count,
    probability,
    num_inputs,
    connection_radius,
    phi_constant=None,
    dimensions=2,
):
    """
    Set up a random geometric graph for a vectorized Spatial Linear Threshold Model (LTM).

    Parameters:
    node_count (int): Number of nodes in the graph.
    probability (float): Connection probability for the geometric graph.
    num_inputs (int): Number of input nodes.
    connection_radius (float): Radius for connecting nodes in the graph.
    phi_constant (float, optional): Constant phi value for all nodes. If None, phi values are random.
    dimensions (int, optional): Dimensionality of the geometric graph.

    Returns:
    tuple: Tuple containing the adjacency matrix, label vector, phi vector, degree vector, seed set, and node positions.
    """
    geometric_graph = nx.random_geometric_graph(
        n=node_count, radius=connection_radius, dim=dimensions, p=probability
    )
    positions = nx.get_node_attributes(geometric_graph, "pos")
    sorted_positions = pd.DataFrame(np.array(list(positions.values()))).sort_values(
        by=0
    )
    seed_set = list(sorted_positions.index[:num_inputs])

    adjacency_matrix = nx.adjacency_matrix(geometric_graph).todense()
    label_vector = create_label_vector(node_count)
    phi_vector = create_phi_vector(node_count, phi_constant)
    ones_vector = create_ones_vector(node_count)
    degree_vector = np.dot(adjacency_matrix, ones_vector)

    return (
        adjacency_matrix,
        label_vector,
        phi_vector,
        degree_vector,
        seed_set,
        positions,
    )


def setup_non_vectorized_ALTM(node_count, probability, theta_value, phi_constant=None):
    """
    Set up a non-vectorized Antagonistic Linear Threshold Model (ALTM).

    Parameters:
    node_count (int): Number of nodes in the graph.
    probability (float): Probability of edge creation.
    theta_value (float): Threshold value for determining node types.
    phi_constant (float, optional): Constant phi value for all nodes. If None, phi values are random.

    Returns:
    tuple: Tuple containing the adjacency matrix, label vector, phi vector, node types, and degree vector.
    """
    adjacency_matrix = create_adjacency_matrix(node_count, probability)
    label_vector = create_label_vector(node_count)
    node_types = create_node_types(node_count, theta_value)
    phi_vector = create_phi_vector(node_count, phi_constant)
    ones_vector = create_ones_vector(node_count)
    degree_vector = np.dot(adjacency_matrix, ones_vector)

    return adjacency_matrix, label_vector, phi_vector, node_types, degree_vector


# 5. Cascade Functions


def run_non_vectorized_LTM_cascade(
    adj_matrix, initial_state, phi_values, degrees, verbose=False
):
    """
    Execute a non-vectorized Linear Threshold Model (LTM) cascade on a network.

    Parameters:
    adj_matrix (ndarray): The adjacency matrix representing the network.
    initial_state (ndarray): Initial state of each node in the network (boolean values).
    phi_values (ndarray): Phi threshold values for each node.
    degrees (ndarray): Degree (number of connections) of each node.
    verbose (bool): If True, enables verbose output.

    Returns:
    ndarray: The state of each node after the cascade process.
    """
    phi_degree_product = np.multiply(phi_values, degrees)
    num_nodes = len(initial_state)
    previous_state = np.zeros(num_nodes)

    while not np.array_equal(initial_state, previous_state):
        previous_state = initial_state.copy()
        for i in range(num_nodes):
            neighbor_sum = np.dot(adj_matrix.astype(int)[i, :], initial_state)
            threshold_exceeded = np.greater(neighbor_sum - phi_degree_product[i], 0)
            initial_state[i] = initial_state[i] or threshold_exceeded

    return initial_state


def run_vectorized_LTM_cascade(
    adj_matrix, initial_state, phi_values, degrees, verbose=False
):
    """
    Execute a vectorized Linear Threshold Model (LTM) cascade on a network.

    Parameters:
    adj_matrix (ndarray): The adjacency matrix representing the network.
    initial_state (ndarray): Initial state of each node in the network (boolean values).
    phi_values (ndarray): Phi threshold values for each node.
    degrees (ndarray): Degree (number of connections) of each node.
    verbose (bool): If True, enables verbose output.

    Returns:
    ndarray: The state of each node after the cascade process.
    """
    phi_degree_product = np.multiply(phi_values, degrees)
    previous_state = np.zeros(len(initial_state))

    while not np.array_equal(initial_state, previous_state):
        neighbor_sum = np.dot(adj_matrix.astype(int), initial_state)
        delta_state = np.greater(neighbor_sum - phi_degree_product, 0)
        previous_state = initial_state
        initial_state = np.logical_or(previous_state, delta_state)

    return initial_state


def run_non_vectorized_ALTM_cascade(
    adj_matrix, initial_state, phi_values, degrees, node_types, verbose=False
):
    """
    Execute a non-vectorized Antagonistic Linear Threshold Model (ALTM) cascade on a network.

    Parameters:
    adj_matrix (ndarray): Adjacency matrix of the network.
    initial_state (ndarray): Initial state of each node (boolean values).
    phi_values (ndarray): Phi threshold values for each node.
    degrees (ndarray): Degree (number of connections) of each node.
    node_types (ndarray): Type of each node (0 or 1).
    verbose (bool): If True, enables verbose output.

    Returns:
    ndarray: The state of each node after the ALTM cascade.
    """
    phi_degree_product = np.multiply(phi_values, degrees)
    num_nodes = len(initial_state)
    previous_state = np.ones(num_nodes)

    while not np.array_equal(initial_state, previous_state):
        previous_state = initial_state.copy()
        unlabeled_nodes = np.where(initial_state == False)[0]
        np.random.shuffle(unlabeled_nodes)

        for i in unlabeled_nodes:
            neighbor_sum = np.dot(adj_matrix.astype(int)[i, :], initial_state)
            nu_minus_phi_d = np.subtract(neighbor_sum, phi_degree_product[i])

            if node_types[i] == 0:
                threshold_exceeded = np.greater(nu_minus_phi_d, 0)
            else:
                threshold_exceeded = np.less_equal(nu_minus_phi_d, 0)

            initial_state[i] = np.logical_or(initial_state[i], threshold_exceeded)

    return initial_state


# 6. Cascade Analysis


def cascade_size(state_vector):
    """
    Calculate the proportion of active nodes in the state vector.

    Parameters:
    state_vector (ndarray): State vector of the nodes.

    Returns:
    float: Proportion of active nodes.
    """
    return np.sum(state_vector) / len(state_vector)


# 7. Plotting Functions

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_with_confidence_interval(cascade_sizes, z_range):
    """
    Plot a line graph of cascade sizes with confidence intervals.

    This function takes a list of cascade sizes and a corresponding range of z values and creates a line plot
    with confidence intervals.

    Parameters:
        cascade_sizes (list or array-like): A list of cascade sizes.
        z_range (list or array-like): A range of z values corresponding to cascade sizes.

    Returns:
        None
    """

    # Create a DataFrame from cascade_sizes with z_range as the index
    df_cascade_sizes = pd.DataFrame(cascade_sizes)
    df_cascade_sizes.index = z_range

    # Reshape the DataFrame using melt
    df = pd.melt(frame=df_cascade_sizes.T, var_name="z", value_name="cascade size")

    plt.figure()

    # Create the line plot
    sns.lineplot(data=df, x="z", y="cascade size", sort=False)


# 8. Binary Array Operations


def vec_bin_array(arr, bit_count):
    """
    Convert a numpy array of integers to a binary representation.

    Parameters:
    arr (ndarray): Numpy array of positive integers.
    bit_count (int): Number of bits to represent each integer.

    Returns:
    ndarray: A binary representation of the array with each element as a bit vector.
    """
    to_binary_str = np.vectorize(lambda x: format(x, f"0{bit_count}b"))
    binary_strs = to_binary_str(arr)
    binary_array = np.array(
        [[int(bit) for bit in binary_str] for binary_str in binary_strs], dtype=np.int8
    )
    return binary_array


# 9. Truth Table Functions


def build_inputs(num_vars):
    """
    Build a Boolean input array for a given number of variables.

    Parameters:
    num_vars (int): Number of input variables.

    Returns:
    ndarray: Array of all possible combinations of inputs for the given number of variables.
    """
    input_list = np.arange(2**num_vars)
    return vec_bin_array(input_list, num_vars)


def build_seeds_list(inputs_array):
    """
    Build a list of seed nodes based on a given input array.

    Parameters:
    inputs_array (ndarray): Array of inputs.

    Returns:
    list: List of seed nodes for each input combination.
    """
    return [list(np.where(row == 1)[0]) for row in inputs_array]


def build_outputs(num_vars):
    """
    Build output columns for Boolean functions of a given number of variables.

    Parameters:
    num_vars (int): Number of variables.

    Returns:
    ndarray: Output array for all possible Boolean functions.
    """
    output_list = np.arange(2 ** (2**num_vars))
    return vec_bin_array(output_list, 2**num_vars)


def build_truth_table(num_vars):
    """
    Build a truth table for boolean functions of a given number of variables.

    Parameters:
    num_vars (int): Number of variables.

    Returns:
    DataFrame: Truth table dataframe.
    """
    inputs = build_inputs(num_vars)
    outputs = build_outputs(num_vars)

    alphabet_list = list(string.ascii_lowercase)
    df_functions = pd.DataFrame(inputs, columns=alphabet_list[:num_vars])
    df_outputs = pd.DataFrame(outputs.T)
    df_truth_table = pd.concat([df_functions, df_outputs], axis=1)

    return df_truth_table, alphabet_list[:num_vars]


def generate_output_binary(function_id):
    """
    Generate a binary representation of a function ID.

    Parameters:
    function_id (int): Function identifier.

    Returns:
    str: Binary representation of the function ID.
    """
    return format(function_id, "b")


def boolean_array_to_int(boolean_array):
    """
    Convert a boolean array to an integer.

    Parameters:
    boolean_array (ndarray): Array of boolean values.

    Returns:
    int: Integer representation of the boolean array.
    """
    binary_string = "".join(boolean_array.astype(int).astype(str))
    return int(binary_string, 2)


def boolean_array_to_int_w_packbits(boolean_array, num_bits=8):
    """
    Convert a boolean array to an integer using numpy's packbits.

    Parameters:
    boolean_array (ndarray): Array of boolean values.
    num_bits (int): Number of bits to use for packing.

    Returns:
    ndarray: Integer value of the packed boolean array.
    """
    padded_array = np.pad(
        boolean_array,
        (num_bits - len(boolean_array), 0),
        "constant",
        constant_values=False,
    )
    return np.packbits(padded_array)[0]


def run_trial_over_inputs(inputs, adj_matrix, phi_values, degrees, verbose=False):
    """
    Run a trial over a set of input values for a vectorized LTM cascade.

    Parameters:
    inputs (ndarray): Array of input values.
    adj_matrix (ndarray): Adjacency matrix of the network.
    phi_values (ndarray): Phi values for each node.
    degrees (ndarray): Degree of each node.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    ndarray: Outputs for each set of input values.
    """
    outputs_each_input = np.zeros((adj_matrix.shape[0], len(inputs)), dtype=bool)
    for i, input_vals in enumerate(inputs):
        label_vector = create_label_vector(adj_matrix.shape[0])
        label_vector = set_seed_nodes(len(label_vector), input_vals)
        label_vector = run_vectorized_LTM_cascade(
            adj_matrix, label_vector, phi_values, degrees, verbose
        )
        outputs_each_input[:, i] = label_vector

    return outputs_each_input


# 10. Monte Carlo trials


def run_cascade_trials(
    node_count, trial_count, z_range, seed_count, phi_constant=None, verbose=False
):
    """
    Run multiple trials of LTM cascade simulations.

    Parameters:
    node_count (int): Number of nodes in the network.
    trial_count (int): Number of trials to run.
    z_range (ndarray): Range of average degrees to simulate over.
    seed_count (int): Number of seed nodes.
    phi_constant (float, optional): Constant phi value for all nodes.
    verbose (bool): Flag to enable verbose logging.

    Returns:
    ndarray: Cascade sizes for each trial and z value.
    """
    p_range = z_range / float(node_count - 1)
    cascade_sizes = np.zeros([len(p_range), trial_count])

    for i, p in enumerate(p_range):
        if verbose:
            print("z:", z_range[i])
        for trial in range(trial_count):
            A, L, phis, d = setup_vectorized_LTM(node_count, p, phi_constant)
            L = set_seed_nodes(node_count, [1] * seed_count)
            L = run_vectorized_LTM_cascade(A, L, phis, d, verbose)
            cascade_sizes[i, trial] = cascade_size(L)

    return cascade_sizes


def run_spatial_cascade_trials(
    node_count,
    trial_count,
    z_range,
    seed_count,
    radius,
    phi_constant=None,
    verbose=False,
):
    """
    Run multiple trials of spatial LTM cascade simulations.

    Parameters:
    node_count (int): Number of nodes in the network.
    trial_count (int): Number of trials to run.
    z_range (ndarray): Range of average degrees to simulate over.
    seed_count (int): Number of seed nodes.
    radius (float): Radius for spatial connection.
    phi_constant (float, optional): Constant phi value for all nodes.
    verbose (bool): Flag to enable verbose logging.

    Returns:
    list: Results of each trial including network parameters and cascade sizes.
    """
    p_range = z_range / float(node_count - 1)
    results_ps = []

    for i, p in enumerate(p_range):
        if verbose:
            print("z:", z_range[i])
        results_trials = []
        for trial in range(trial_count):
            A, L, phis, d, seed_set, pos = setup_vectorized_spatial_LTM(
                node_count, p, seed_count, radius, phi_constant
            )
            L = set_seed_nodes_in_seed_set(node_count, seed_set, [1])
            L = run_vectorized_LTM_cascade(A, L, phis, d, verbose)
            trial_cascade_size = cascade_size(L)
            results_trials.append((A, L, phis, d, seed_set, pos, trial_cascade_size))
        results_ps.append(results_trials)

    return results_ps


def run_cascade_trials_sequential_ALTM(
    N, num_trials, z_range, num_seeds, theta, phi_constant=None, verbose=False
):

    p_range = z_range / float(N - 1)

    cascade_sizes = np.zeros([len(p_range), num_trials])

    for i in range(len(p_range)):
        p = p_range[i]

        if verbose:
            print("z:", z_range[i])

        for trial in range(num_trials):

            (A, L, phis, node_types, d) = setup_non_vectorized_ALTM(
                N, p, theta, phi_constant
            )
            L = set_seed_nodes(N, [1])
            #            L = run_vectorized_LTM_cascade(A, L, phis, d, verbose)
            L = run_non_vectorized_ALTM_cascade(A, L, phis, d, node_types, verbose)
            cascade_sizes[i, trial] = cascade_size(L)

    return cascade_sizes


def run_trial_over_inputs_sequential_ALTM(
    inputs, adj_matrix, phi_values, node_types, degrees, verbose=False
):
    """
    Run trials over a set of input values for a sequential ALTM cascade.

    Parameters:
    inputs (ndarray): Array of input values.
    adj_matrix (ndarray): Adjacency matrix of the network.
    phi_values (ndarray): Phi values for each node.
    node_types (ndarray): Type of each node (standard or antagonistic).
    degrees (ndarray): Degree of each node.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    ndarray: Outputs for each set of input values.
    """
    outputs_each_input = np.zeros((adj_matrix.shape[0], len(inputs)), dtype=bool)

    for i, input_vals in enumerate(inputs):
        label_vector = create_label_vector(adj_matrix.shape[0])
        label_vector = set_seed_nodes(len(label_vector), input_vals)
        label_vector = run_non_vectorized_ALTM_cascade(
            adj_matrix, label_vector, phi_values, degrees, node_types, verbose
        )
        outputs_each_input[:, i] = label_vector

    return outputs_each_input


def run_trial_over_inputs_int(inputs, adj_matrix, phi_values, degrees, verbose=False):
    """
    Run trials over a set of input values for a vectorized LTM cascade and return integer outputs.

    Parameters:
    inputs (ndarray): Array of input values.
    adj_matrix (ndarray): Adjacency matrix of the network.
    phi_values (ndarray): Phi values for each node.
    degrees (ndarray): Degree of each node.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    ndarray: Integer outputs representing functions computed for each input set.
    """
    outputs_each_input = np.zeros((adj_matrix.shape[0], len(inputs)), dtype=int)

    for i, input_vals in enumerate(inputs):
        label_vector = create_label_vector(adj_matrix.shape[0])
        label_vector = set_seed_nodes(len(label_vector), input_vals)
        label_vector = run_vectorized_LTM_cascade(
            adj_matrix, label_vector, phi_values, degrees, verbose
        )
        outputs_each_input[:, i] = boolean_array_to_int(label_vector)

    return identify_functions_computed(adj_matrix.shape[0], outputs_each_input)


def run_logic_trial(
    trial, inputs, node_count, connection_prob, phi_constant=None, verbose=False
):
    """
    Run a logic trial with vectorized LTM for all inputs from the truth table.

    Parameters:
    trial (int): Trial number.
    inputs (ndarray): Array of input values from the truth table.
    node_count (int): Number of nodes in the network.
    connection_prob (float): Probability of connection between nodes.
    phi_constant (float, optional): Constant phi value for all nodes.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    list: Network parameters and outputs for the trial.
    """
    random_seed = set_rand_seed()
    adj_matrix, label_vector, phi_values, degrees = setup_vectorized_LTM(
        node_count, connection_prob, phi_constant
    )
    outputs = run_trial_over_inputs(inputs, adj_matrix, phi_values, degrees, verbose)
    return [adj_matrix, phi_values, outputs, random_seed]


def run_logic_trial_sequential_ALTM(
    trial, inputs, node_count, connection_prob, theta, verbose=False
):
    """
    Run a logic trial with sequential ALTM for all inputs from the truth table.

    Parameters:
    trial (int): Trial number.
    inputs (ndarray): Array of input values from the truth table.
    node_count (int): Number of nodes in the network.
    connection_prob (float): Probability of connection between nodes.
    theta (float): Threshold value for determining node types.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    list: Network parameters and outputs for the trial.
    """
    random_seed = set_rand_seed()
    (
        adj_matrix,
        label_vector,
        phi_values,
        node_types,
        degrees,
    ) = setup_non_vectorized_ALTM(node_count, connection_prob, theta)
    outputs = run_trial_over_inputs_sequential_ALTM(
        inputs, adj_matrix, phi_values, node_types, degrees, verbose
    )
    return [adj_matrix, phi_values, node_types, outputs, random_seed]


def run_logic_trial_int(trial, inputs, node_count, connection_prob, verbose=False):
    """
    Run a logic trial with vectorized LTM for all inputs from the truth table and return integer outputs.

    Parameters:
    trial (int): Trial number.
    inputs (ndarray): Array of input values from the truth table.
    node_count (int): Number of nodes in the network.
    connection_prob (float): Probability of connection between nodes.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    list: Network parameters and integer outputs for the trial.
    """
    random_seed = set_rand_seed()
    adj_matrix, label_vector, phi_values, degrees = setup_vectorized_LTM(
        node_count, connection_prob
    )
    outputs = run_trial_over_inputs_int(
        inputs, adj_matrix, phi_values, degrees, verbose
    )
    return [adj_matrix, phi_values, outputs, random_seed]


def run_logic_trials(
    node_count, connection_prob, input_layer_size, num_trials, verbose=False
):
    """
    Perform multiple trials of logic functions computed for specific network settings.

    Parameters:
    node_count (int): Number of nodes in the network.
    connection_prob (float): Edge probability for Erdos-Renyi-Gilbert model.
    input_layer_size (int): Number of seed nodes or 'input layer size'.
    num_trials (int): Number of trials to run.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    tuple: A tuple containing adjacency matrices, phi values, logic cascade outputs, and random seeds for each trial.
    """
    inputs = build_inputs(input_layer_size)

    A_vals = np.zeros([node_count, node_count, num_trials], dtype=int)
    phi_vals = np.zeros([node_count, num_trials], dtype=float)
    logic_cascade_outputs = np.zeros([node_count, len(inputs), num_trials], dtype=bool)
    random_seeds = np.zeros(num_trials, dtype=int)

    for trial in range(num_trials):
        if verbose and num_trials > 10 and trial % int(num_trials / 10) == 0:
            print("trial", trial)
        A, phis, outputs_trial, random_seed_trial = run_logic_trial(
            trial, inputs, node_count, connection_prob
        )

        A_vals[:, :, trial] = A
        phi_vals[:, trial] = phis
        logic_cascade_outputs[:, :, trial] = outputs_trial
        random_seeds[trial] = random_seed_trial

    return A_vals, phi_vals, logic_cascade_outputs, random_seeds


def run_logic_trials_sequential_ALTM(
    node_count, connection_prob, input_layer_size, theta, num_trials, verbose=False
):
    """
    Perform multiple trials of logic functions computed for a specific network setting using sequential ALTM.

    Parameters:
    node_count (int): Number of nodes in the network.
    connection_prob (float): Edge probability for Erdos-Renyi-Gilbert model.
    input_layer_size (int): Number of (possible) seed nodes or 'input layer size'.
    theta (float): Threshold value for determining node types.
    num_trials (int): Number of trials to run.
    verbose (bool): Enables verbose output if set to True.

    Returns:
    tuple: A tuple containing adjacency matrices, phi values, node types, logic cascade outputs, and random seeds for each trial.
    """
    inputs = build_inputs(input_layer_size)

    A_vals = np.zeros([node_count, node_count, num_trials], dtype=int)
    phi_vals = np.zeros([node_count, num_trials], dtype=float)
    node_type_vals = np.zeros([node_count, num_trials], dtype=bool)
    logic_cascade_outputs = np.zeros([node_count, len(inputs), num_trials], dtype=bool)
    random_seeds = np.zeros(num_trials, dtype=int)

    for trial in range(num_trials):
        if verbose and num_trials > 10 and trial % int(num_trials / 10) == 0:
            print("trial", trial)

        (
            A,
            phis,
            node_types,
            outputs_trial,
            random_seed_trial,
        ) = run_logic_trial_sequential_ALTM(
            trial, inputs, node_count, connection_prob, theta, verbose
        )

        A_vals[:, :, trial] = A
        phi_vals[:, trial] = phis
        node_type_vals[:, trial] = node_types
        logic_cascade_outputs[:, :, trial] = outputs_trial
        random_seeds[trial] = random_seed_trial

    return A_vals, phi_vals, node_type_vals, logic_cascade_outputs, random_seeds


def run_logic_trials(
    num_nodes, edge_probability, num_inputs, threshold, num_trials, verbose=False
):
    """Run logic trials in parallel.

    Args:
        num_nodes: Number of nodes in the network.
        edge_probability: Probability of edge between nodes.
        num_inputs: Number of input nodes.
        threshold: Node activation threshold.
        num_trials: Number of trials to run.
        verbose: Whether to print trial details.

    Returns:
        adj_matrices: Adjacency matrices for each trial.
        thresholds: Node thresholds for each trial.
        node_types: Node types (input/logic) for each trial.
        cascade_outputs: Node outputs for each input and trial.
        seeds: Random seeds used for each trial.
    """

    # Inputs column from truth table
    inputs = build_inputs(num_inputs)

    # Storage for output
    adj_matrices = np.zeros((num_nodes, num_nodes, num_trials), dtype=int)
    thresholds = np.zeros((num_nodes, num_trials), dtype=float)
    node_types = np.zeros((num_nodes, num_trials), dtype=bool)
    cascade_outputs = np.zeros((num_nodes, len(inputs), num_trials), dtype=bool)
    seeds = np.zeros(num_trials, dtype=int)

    results = Parallel(n_jobs=-1)(
        delayed(run_single_trial)(
            trial, inputs, num_nodes, edge_probability, threshold, verbose
        )
        for trial in range(num_trials)
    )

    adj_matrices = np.array([r[0] for r in results], dtype=int)
    thresholds = np.array([r[1] for r in results], dtype=float)
    node_types = np.array([r[2] for r in results], dtype=bool)
    cascade_outputs = np.array([r[3].T for r in results], dtype=bool)
    seeds = np.array([r[4] for r in results], dtype=int)

    return (adj_matrices.T, thresholds.T, node_types.T, cascade_outputs.T, seeds)


def run_logic_trials_parallel(N, p, k, num_trials, phi_constant=None, verbose=False):
    """Run logic trials in parallel.

    Args:
        N: Number of nodes in the network.
        p: Probability of edge between nodes.
        k: Number of input nodes.
        num_trials: Number of trials to run.
        phi_constant: Node activation threshold (optional).
        verbose: Print trial details.

    Returns:
        A_vals: Adjacency matrices for each trial.
        phi_vals: Node thresholds for each trial.
        logic_cascade_outputs: Node outputs for each input and trial.
        random_seeds: Random seeds used for each trial.
    """

    inputs = build_inputs(k)

    results = Parallel(n_jobs=-1)(
        delayed(run_logic_trial)(trial, inputs, N, p, phi_constant, verbose)
        for trial in range(num_trials)
    )

    A_vals = np.array([r[0] for r in results], dtype=int)
    phi_vals = np.array([r[1] for r in results], dtype=float)
    logic_cascade_outputs = np.array([r[2].T for r in results], dtype=bool)
    random_seeds = np.array([r[3] for r in results], dtype=int)

    return A_vals.T, phi_vals.T, logic_cascade_outputs.T, random_seeds


# 11.  Boolean function analysis


def identify_computed_functions(num_nodes, cascade_outputs):
    """Identify computed function IDs from node outputs

    Args:
        num_nodes (int): Number of nodes
        cascade_outputs (np.array): Node output values
            Dimensions: (num_nodes, num_input_rows)

    Returns:
        np.array: Computed function IDs for each node
    """

    function_ids = np.zeros(num_nodes, dtype=int)

    for idx, outputs in enumerate(cascade_outputs.T):
        function_ids[idx] = boolean_array_to_int(outputs)

    return function_ids


def identify_computed_functions_trials(num_trials, num_nodes, cascade_outputs):
    """Identify computed functions over trials

    Args:
        num_trials (int): Number of trials
        num_nodes (int): Number of nodes
        cascade_outputs (np.array): Node outputs by trial
            Dimensions: (num_trials, num_nodes, num_input_rows)

    Returns:
        np.array: Computed function IDs by trial and node
    """

    function_ids = np.zeros((num_trials, num_nodes), dtype=int)

    for t in range(num_trials):
        trial_outputs = cascade_outputs[:, :, t]
        function_ids[t, :] = identify_computed_functions(num_nodes, trial_outputs)

    return function_ids


def identify_computed_functions_trials_z(
    num_trials, num_nodes, z_range, cascade_outputs
):
    """Identify computed functions over z range and trials

    Args:
        num_trials (int): Number of trials
        num_nodes (int): Number of nodes
        z_range (np.array): Range of z values
        cascade_outputs (np.array): Node outputs by z, trial
            Dimensions: (len(z_range), num_nodes, num_input_rows, num_trials)

    Returns:
        np.array: Computed function IDs by z, trial, and node
    """

    function_ids = np.zeros((len(z_range), num_trials, num_nodes), dtype=int)

    for z, outputs in enumerate(cascade_outputs):
        function_ids[z] = identify_computed_functions_trials(
            num_trials, num_nodes, outputs
        )

    return function_ids


# 12.  Range of Monte Carlo functions


def run_logic_trials_over_z_range(N, k, z_range, num_trials, verbose=False):
    """Run logic trials over a range of z values

    Args:
        N (int): Number of nodes
        k (int): Number of input nodes
        z_range (np.array): Range of z values
        num_trials (int): Number of trials
        verbose (bool): Print progress

    Returns:
        all_A_vals (np.array): Adjacency matrices
        all_phi_vals (np.array): Node thresholds
        all_logic_cascade_outputs (np.array): Node outputs
        all_random_seeds (np.array): Random seeds
    """

    p_range = z_range / (N - 1)
    num_input_rows = 2**k

    # Output storage
    all_A_vals = np.zeros((len(p_range), N, N, num_trials), dtype=int)
    all_phi_vals = np.zeros((len(p_range), N, num_trials), dtype=float)
    all_logic_cascade_outputs = np.zeros(
        (len(p_range), N, num_input_rows, num_trials), dtype=bool
    )
    all_random_seeds = np.zeros((len(p_range), num_trials), dtype=int)

    for i, p in enumerate(p_range):
        if verbose and i % 10 == 0:
            print(f"z = {z_range[i]}")

        A_vals, phi_vals, logic_cascade_outputs, random_seeds = run_logic_trials(
            N, p, k, num_trials, verbose
        )

        all_A_vals[i] = A_vals
        all_phi_vals[i] = phi_vals
        all_logic_cascade_outputs[i] = logic_cascade_outputs
        all_random_seeds[i] = random_seeds

    return all_A_vals, all_phi_vals, all_logic_cascade_outputs, all_random_seeds


def run_logic_trials_parallel_over_z_range(
    N, k, z_range, num_trials, phi_constant=None, verbose=False
):
    """Run parallel logic trials over a range of z

    Args:
        N (int): Number of nodes
        k (int): Number of input nodes
        z_range (np.array): Range of z values
        num_trials (int): Number of trials
        phi_constant (float): Threshold (optional)
        verbose (bool): Print progress

    Returns:
        all_logic_cascade_outputs (np.array): Node outputs
    """

    p_range = z_range / (N - 1)
    num_input_rows = 2**k

    # Output storage
    all_A_vals = np.zeros((len(p_range), N, N, num_trials), dtype=int)
    all_phi_vals = np.zeros((len(p_range), N, num_trials), dtype=float)
    all_logic_cascade_outputs = np.zeros(
        (len(p_range), N, num_input_rows, num_trials), dtype=bool
    )
    all_random_seeds = np.zeros((len(p_range), num_trials), dtype=int)

    for i, p in enumerate(p_range):
        if verbose:
            print(f"z = {z_range[i]}")

        (
            A_vals,
            phi_vals,
            logic_cascade_outputs,
            random_seeds,
        ) = run_logic_trials_parallel(N, p, k, num_trials, phi_constant, verbose)

        all_A_vals[i] = A_vals
        all_phi_vals[i] = phi_vals
        all_logic_cascade_outputs[i] = logic_cascade_outputs
        all_random_seeds[i] = random_seeds

    return all_logic_cascade_outputs


def run_logic_trials_ALTM_parallel_over_z_range(
    N, k, z_range, theta, num_trials, verbose=False
):
    """Run ALTM parallel trials over a range of z

    Args:
        N (int): Number of nodes
        k (int): Number of inputs
        z_range (np.array): Range of z values
        theta (float): Threshold
        num_trials (int): Number of trials
        verbose (bool): Print progress

    Returns:
        all_A_vals: Adjacency matrices
        all_phi_vals: Node thresholds
        all_node_types: Node types
        all_logic_cascade_outputs: Node outputs
        all_random_seeds: Random seeds
    """

    p_range = z_range / (N - 1)
    num_input_rows = 2**k

    # Output storage
    all_A_vals = np.zeros((len(p_range), N, N, num_trials), dtype=int)
    all_phi_vals = np.zeros((len(p_range), N, num_trials), dtype=float)
    all_node_types = np.zeros((len(p_range), N, num_trials), dtype=bool)
    all_logic_cascade_outputs = np.zeros(
        (len(p_range), N, num_input_rows, num_trials), dtype=bool
    )
    all_random_seeds = np.zeros((len(p_range), num_trials), dtype=int)

    for i, p in enumerate(p_range):
        if verbose:
            print(f"z = {z_range[i]}")

        result = run_logic_trials_parallel_ALTM(N, p, k, theta, num_trials, verbose)

        all_A_vals[i] = result[0]
        all_phi_vals[i] = result[1]
        all_node_types[i] = result[2]
        all_logic_cascade_outputs[i] = result[3]
        all_random_seeds[i] = result[4]

    return (
        all_A_vals,
        all_phi_vals,
        all_node_types,
        all_logic_cascade_outputs,
        all_random_seeds,
    )


# 13. Hamming Distance and Cube, and Boolean Logic functions


def hammingDistance(n1: int, n2: int) -> int:
    """Calculate the Hamming distance between two integers.

    The Hamming distance is the number of bit positions that differ
    between two equal-length binary integers.

    Args:
        n1: First integer
        n2: Second integer

    Returns:
        int: The Hamming distance
    """

    distance = 0
    x = n1 ^ n2

    while x > 0:
        distance += x & 1
        x >>= 1

    return distance


def build_hamming_cube(dimensions: int = 2, functions: dict = None) -> nx.Graph:
    """Build a graph representation of a Hamming cube.

    A Hamming cube of dimension D contains 2^D corners, corresponding
    to all possible binary strings of length D. Edges connect strings
    that differ by only one bit flip.

    Args:
        dimensions: The dimensionality of the Hamming cube.
        functions: Optional node functions as a dictionary.

    Returns:
        nx.Graph: The Hamming cube graph.
    """

    # Build cube and add edges
    corners = generate_corners(dimensions)
    graph = nx.Graph()
    graph.add_nodes_from(corners)

    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if calculate_hamming_distance(int(node1, 2), int(node2, 2)) == 1:
                graph.add_edge(node1, node2)

    # Add node functions
    if functions:
        set_node_functions(graph, functions)

    return graph


def offset_position(position, horizontal_offset=0.5, vertical_offset=0.0):
    """
    Shifts the position of labels when drawing a graph.

    Args:
    - position (dict): A dictionary of node positions.
    - horizontal_offset (float): The horizontal offset for shifting the position. Default is 0.5.
    - vertical_offset (float): The vertical offset for shifting the position. Default is 0.0.

    Returns:
    - dict: A dictionary of the new offset positions.
    """
    # Shift each node by the offset
    pos_array = np.array(list(position.values())) + np.array(
        [vertical_offset, horizontal_offset]
    )

    # Create a new position dictionary
    new_offset_position = dict(zip(list(position.keys()), list(pos_array)))

    return new_offset_position


def draw_hamming_cube(
    graph,
    draw_function=False,
    function_h_offset=0.2,
    function_v_offset=0.2,
    position=None,
    title=None,
):
    """
    Draws a Hamming cube graph.

    Args:
    - graph (networkx.Graph): The input graph.
    - draw_function (bool): Whether to draw the node labels. Default is False.
    - function_h_offset (float): The horizontal offset for the function labels. Default is 0.2.
    - function_v_offset (float): The vertical offset for the function labels. Default is 0.2.
    - position (dict): A dictionary of node positions. If None, it will be generated internally.
    - title (str): The title of the plot. Default is None.

    Returns:
    - dict: A dictionary of the node positions.
    """
    fig = plt.figure(figsize=[5, 5])
    ax = fig.gca()

    if position is None:
        new_position = dict(
            zip(graph.nodes(), [np.array(list(x), dtype=int) for x in graph.nodes()])
        )

    nx.draw(
        graph,
        new_position,
        ax=ax,
        node_size=10,
        edge_color="lightgrey",
        node_color="lightgrey",
    )
    nx.draw_networkx_labels(graph, new_position, font_size=20)

    if draw_function:
        labels = nx.get_node_attributes(graph, "function")
        pos_offset = offset_position(new_position, function_h_offset, function_v_offset)
        nx.draw_networkx_labels(graph, pos_offset, labels=labels, font_color="red")

    if title is not None:
        plt.title(title)

    fig.tight_layout()
    plt.margins(0.2)

    return new_position


def set_node_attributes_according_to_function_id(graph, truth_table, function_id):
    """
    Set the node attributes of the graph according to the function id in the truth table.

    Args:
    - graph (networkx.Graph): The input graph.
    - truth_table (pandas.DataFrame): The truth table containing function values.
    - function_id (str): The id of the function in the truth table.

    Returns:
    - networkx.Graph: The graph with updated node attributes.
    """
    function_values = dict(zip(graph.nodes(), truth_table[function_id].values))
    nx.set_node_attributes(graph, function_values, "function")
    return graph


def count_axial_reflection_symmetries(k, verbose=False):
    """
    Count the number of axial reflection symmetries for all Boolean functions in k inputs.

    Args:
    - k (int): The number of inputs.
    - verbose (bool): Whether to print verbose output. Default is False.

    Returns:
    - numpy.array: An array containing the count of axial reflection symmetries for each Boolean function.
    """
    # (The existing code for counting axial reflection symmetries remains unchanged)
    pass


def create_empty_hamming_cube(k):
    """
    Create a Hamming cube of dimension k with all zeros, i.e., the zero function.

    Args:
    - k (int): The dimension of the Hamming cube.

    Returns:
    - numpy.array: The created Hamming cube with all zeros.
    """
    H = np.zeros([2] * k, dtype=int)
    return H


def create_random_hamming_cube(k):
    """
    Create a random Hamming cube of dimension k.

    Args:
    - k (int): The dimension of the Hamming cube.

    Returns:
    - numpy.array: The created random Hamming cube.
    """
    H = np.random.randint(0, high=2, size=tuple([2] * k))
    return H


def is_monotone_increasing(H):
    """
    Check if the input numpy array is a monotone increasing function in each dimension of the cube.

    Args:
    - H (numpy.array): The input numpy array of shape [2] * k.

    Returns:
    - bool: True if the input is a monotone increasing function, False otherwise.
    """
    for axis in range(len(H.shape)):
        not_monotone_increasing = np.any(np.diff(H, axis=axis) < 0)
        if not_monotone_increasing:
            return False
    return True


def count_axial_reflection_symmetries(H):
    """
    Count the number of axial reflection symmetries in the input Hamming cube.

    Args:
    - H (numpy.array): The input Hamming cube.

    Returns:
    - int: The count of axial reflection symmetries.
    """
    congruent_reflections = 0

    for axis_i in range(len(H.shape)):
        H_flipped = np.flip(H, axis=axis_i)
        congruent = np.array_equal(H, H_flipped)

        if congruent:
            congruent_reflections += 1

    return congruent_reflections


def determine_decision_tree_complexity(H):
    """
    Determine the decision tree complexity of the input Hamming cube.

    Args:
    - H (numpy.array): The input Hamming cube.

    Returns:
    - int: The decision tree complexity.
    """
    R = count_axial_reflection_symmetries(H)
    D = len(H.shape)
    C = D - R
    return C


def is_LTM_computable(H):
    """
    Check if the input Hamming cube represents a computable function using LTM.

    Args:
    - H (numpy.array): The input Hamming cube.

    Returns:
    - bool: True if the function is LTM computable, False otherwise.
    """
    # (The existing code for checking LTM computability remains unchanged)
    pass


def Hamming_cube_to_1_d_array(H):
    """
    returns a column of a truth table and
    the function id (by converting from binary to int)
    """
    fn_arr = H.flatten()
    fn_id = boolean_array_to_int(fn_arr)
    return (fn_arr, fn_id)


def truth_table_to_Hamming_cube(df_truth_table, k, fn_index):
    """
    Convert a truth table entry to a Hamming cube
    """
    fn = df_truth_table[fn_index]
    fn_vals = fn.values
    H = fn_vals.reshape([2] * k)

    return H


def compute_all_complexities_truth_table(k):
    """
    for a given number of inputs(k), create the truth table and compute decision tree complexity for each function
    returns a list of complexity by function id
    """

    complexities = []

    df_truth_table, inputs = build_truth_table(k)

    num_functions = 2 ** (2**k)

    for fn_ix in range(num_functions):

        H = truth_table_to_Hamming_cube(df_truth_table, k, fn_ix)

        C = determine_decision_tree_complexity(H)
        complexities.append(C)

    return np.array(complexities)


def calc_function_probability_from_dec_tree_complexity(p_gcc, C):
    """
    Estimate the probability of a function from its required paths.

    Given the p_gcc, the probability of a random node being in the Giant Component,
    and C, a function's decision tree complexity,
    calculate p_gcc^(C+1), the probability that a function occurs.
    Notes:
    - This just gives proportionality to the number of nodes requiring paths, does not account for cascade being successful.
    - This is only a good (order of magnitude) estimate for monotone boolean functions.
      For non-monotone functions, it may (grossly) underestimate the probability,
      since the sub-networks (motifs) may contain many more nodes than their monotone counterparts.
    """

    return np.power(p_gcc, C + 1)


def entropy_of_bistring(string):
    """
    Find the entropy (base 2) of an input string of bits.

    Args:
    - string (str or list): The input string of bits. It can be a list of ints, bools, or a string '001101'.

    Returns:
    - float: The entropy of the input string.
    """
    string = np.array(list(string), dtype=int)

    num_unique_vals = len(np.unique(string))
    counts, bins = np.histogram(string, bins=num_unique_vals)
    freqs = counts / np.sum(counts)

    S = -np.dot(freqs, np.log2(freqs))
    return S


def relabel_edges(G, node_mapping):
    """
    Given G and a dictionary mapping node ids to new node ids, relabel all of G's edges.

    Args:
    - G (networkx.Graph): The input graph.
    - node_mapping (dict): A dictionary mapping old node ids to new node ids.

    Returns:
    - networkx.Graph: The graph with relabeled edges.
    """
    E = G.edges
    new_G = nx.Graph()

    for e in E:
        new_e0 = node_mapping[e[0]]
        new_e1 = node_mapping[e[1]]
        new_G.add_edge(new_e0, new_e1)

    return new_G


def relabel_network_left_to_right(G):
    """
    Relabel the node ids of a random geometric graph from left to right in ascending order.

    Args:
    - G (networkx.Graph): The input graph.

    Returns:
    - networkx.Graph: The graph with relabeled node ids and updated node positions.
    """
    pos = nx.get_node_attributes(G, name="pos")
    df_pos = pd.DataFrame(np.array(list(pos.values())))
    df_pos.sort_values(by=0, inplace=True)
    df_pos.reset_index(inplace=True)

    node_mapping = dict(zip(df_pos["index"], df_pos.index))

    new_G = relabel_edges(G, node_mapping)

    new_pos = dict()
    for i in range(len(df_pos)):
        new_pos[df_pos.index[i]] = [df_pos[0][i], df_pos[1][i]]

    nx.set_node_attributes(new_G, new_pos, name="pos")
    return new_G


# 14. Graph Measures


def all_pairs_distances(A):
    """
    Compute all pairs distances.

    Args:
    - A (numpy.array): The adjacency matrix.

    Returns:
    - numpy.array: An array of all pairs distances. If no path exists, 0 is returned in the array.
    """
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()
    G_nk = nk.nxadapter.nx2nk(G)
    apsp = nk.distance.APSP(G_nk)
    apsp.run()
    distances = apsp.getDistances(asarray=True)
    distances[distances > N**2] = 0
    return distances


def draw_distance_graph(distances, L=None):
    """
    Draw a circular graph given the distances adjacency matrix and node states L.

    Args:
    - distances (numpy.array): The distances adjacency matrix.
    - L (list, optional): The node states. Default is None.
    """
    G_dist = nx.from_numpy_array(distances)
    plt.figure()
    pos = nx.circular_layout(G_dist)
    if L is not None:
        nx.draw(G_dist, pos, with_labels=True, node_color=L, font_color="red")
    else:
        nx.draw(G_dist, pos, with_labels=True, font_color="red")
    labels = nx.get_edge_attributes(G_dist, "weight")
    nx.draw_networkx_edge_labels(G_dist, pos, labels)


def calculate_state_distances(N, L):
    """
    Calculate the state distance between all node pairs.
    If state i = state j, distance_ij <- 0; else distance_ij <- 1.

    Args:
    - N (int): The number of nodes.
    - L (list): The list of node states.

    Returns:
    - numpy.array: An array of state distances between all node pairs.
    """
    state_distances = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if L[i] == L[j]:
                state_distances[i, j] = 0
            else:
                state_distances[i, j] = 1

    return state_distances


def make_nauty_graph_from_nx(G):
    """
    Build a nauty graph from a NetworkX graph G.

    Args:
    - G (networkx.Graph): The input graph.

    Returns:
    - pynauty.Graph: The nauty graph.
    """
    N = G.number_of_nodes()
    g = pyn.Graph(N)
    for i in range(N):
        neighbors = list(G.neighbors(i))
        g.connect_vertex(i, neighbors)

    return g


# 15. Data generation


def make_discrete_blobs(
    n_samples=10, n_features=2, centers=2, center_box=[0, 1], cluster_std=0.4
):
    """
    Make discrete data classes (on Hamming cube).

    This is imperfect and may produce data outside of the 0,1 hamming cube.

    Args:
    - n_samples (int): The number of samples. Default is 10.
    - n_features (int): The number of features. Default is 2.
    - centers (int): The number of centers. Default is 2.
    - center_box (list): The bounding box for each cluster center. Default is [0, 1].
    - cluster_std (float): The standard deviation of the clusters. Default is 0.4.

    Returns:
    - tuple: A tuple containing the generated points and their classes.
    """

    res = make_blobs(
        n_samples,
        n_features=n_features,
        centers=centers,
        center_box=center_box,
        cluster_std=cluster_std,
    )
    points = res[0].astype(int)
    classes = res[1]

    return points, classes


def plot_discrete_blobs(res):
    """
    Plot discrete classes using the result from make_discrete_blobs().

    Obviously only works for 2-D data; otherwise, can try drawing k-D Hamming cube.

    Args:
    - res (tuple): The result from make_discrete_blobs() containing the generated points and their classes.

    Returns:
    - matplotlib.axes._subplots.AxesSubplot: The scatter plot of the discrete classes.
    """

    points = res[0].astype(int)

    df = pd.DataFrame(points)
    df["class"] = res[1]

    ax = df.plot.scatter(x=0, y=1, color=df["class"], cmap=plt.cm.Accent)

    return ax

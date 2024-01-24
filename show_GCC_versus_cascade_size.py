import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def simulate_network(n, z, threshold=0.5):
    """
    Simulate a network using the Linear Threshold Model.

    Parameters:
    n (int): Number of nodes in the graph.
    z (float): Mean degree of the nodes in the graph.
    threshold (float): Activation threshold for the nodes.

    Returns:
    tuple: A tuple containing the size of the largest connected component (GCC)
           and the size of the cascade, both as integers.
    """
    
    # Create a random graph
    G = nx.erdos_renyi_graph(n, z / (n - 1))

    # Find the largest connected component (GCC)
    gcc_size = len(max(nx.connected_components(G), key=len))

    # Initialize all nodes as inactive
    active = np.zeros(n, dtype=bool)
    
    # Randomly choose an initial active node
    initial_node = np.random.choice(n)
    active[initial_node] = True

    # Simulate the cascade
    new_activations = True
    while new_activations:
        new_activations = False
        for node in range(n):
            if not active[node]:
                neighbors = list(G.neighbors(node))
                if len(neighbors) > 0:
                    # Activate if the fraction of active neighbors is above the threshold
                    if sum(active[neighbors]) / len(neighbors) > threshold:
                        active[node] = True
                        new_activations = True

    # Calculate cascade size
    cascade_size = sum(active)

    return gcc_size, cascade_size


def run_simulation_for_z_range(n, z_range, num_z_steps, num_trials, threshold=0.5):
    """
    Run the simulation for a range of mean degree values, multiple times for each value.

    Parameters:
    n (int): Number of nodes in the graph.
    z_range (tuple): A tuple of two floats indicating the start and end of the z values range.
    num_z_steps (int): Number of z values to simulate.
    num_trials (int): Number of simulations to run for each z value.
    threshold (float): Activation threshold for the nodes.

    Returns:
    pandas.DataFrame: A DataFrame containing the normalized sizes of GCC and cascades for each z value and simulation.
    """
    z_values = np.linspace(z_range[0], z_range[1], num_z_steps)
    results = []

    for z in z_values:
        for _ in range(num_trials):
            gcc_size, cascade_size = simulate_network(n, z, threshold)
            results.append({'Mean Degree': z, 'Normalized Size': gcc_size / n, 'Type': 'GCC'})
            results.append({'Mean Degree': z, 'Normalized Size': cascade_size / n, 'Type': 'Cascade'})

    return pd.DataFrame(results)



def plot_data(df, plot_choice):
    """
    Plot the data with confidence intervals based on the specified choice.

    Parameters:
    df (pd.DataFrame): DataFrame containing the simulation results.
    plot_choice (str): Choice of data to plot ('GCC', 'Cascade', or 'Both').
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if plot_choice in ['GCC', 'Both']:
        df_gcc = df[df['Type'] == 'GCC']
        sns.lineplot(data=df_gcc, x='Mean Degree', y='Normalized Size', errorbar='sd', label='GCC Size', color='blue', ax=ax1)
        ax1.set_xlabel('Mean Degree (z)')
        ax1.set_ylabel('Normalized GCC Size', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

    if plot_choice in ['Cascade', 'Both']:
        df_cascade = df[df['Type'] == 'Cascade']
        sns.lineplot(data=df_cascade, x='Mean Degree', y='Normalized Size', errorbar='sd', label='Cascade Size', color='green', ax=ax1)
        ax1.set_ylabel('Normalized Cascade Size', color='green')
        ax1.tick_params(axis='y', labelcolor='green')

    plt.title(f'Normalized {plot_choice} Sizes vs Mean Degree with Confidence Intervals')
    plt.legend(title='', loc='upper left')

    plt.show()

   
def main():
    parser = argparse.ArgumentParser(description='Run Linear Threshold Model Simulation')
    
    parser.add_argument('--n', type=int, default=100, help='Number of nodes in the graph (default: 100)')
    parser.add_argument('--z_start', type=float, default=0, help='Start of the mean degree range (default: 0)')
    parser.add_argument('--z_end', type=float, default=5, help='End of the mean degree range (default: 5)')
    parser.add_argument('--num_z_steps', type=int, default=20, help='Number of z values to simulate (default: 20)')
    parser.add_argument('--threshold', type=float, default=0.1, help='Activation threshold for the nodes (default: 0.1)')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of simulations per z value (default: 10)')
    
    parser.add_argument('--plot_data', choices=['GCC', 'Cascade', 'Both'], default='Both',
                        help='Specify what to plot: GCC, Cascade, or Both (default: Both)')

    args = parser.parse_args()
    df = run_simulation_for_z_range(args.n, (args.z_start, args.z_end), args.num_z_steps, args.num_trials, args.threshold)
    plot_data(df, args.plot_data)

if __name__ == '__main__':
    main()



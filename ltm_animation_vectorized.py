import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio.v2 as imageio
import os
from graphviz import Graph
import matplotlib.patches as mpatches


def initialize_network(num_nodes, mean_degree):
    """Initialize a random network with given number of nodes and mean degree."""
    edge_prob = mean_degree / (num_nodes - 1)
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    thresholds = np.random.rand(num_nodes)  # Vector of thresholds
    active = np.zeros(num_nodes, dtype=bool)  # Vector to track active nodes
    return G, thresholds, active


def threshold_to_grey(threshold):
    """Map a threshold value to a shade of grey."""
    # Convert threshold to a grey scale (0 is light grey, 1 is black)
    grey_scale = int((1 - threshold) * 255)
    return f'#{grey_scale:02x}{grey_scale:02x}{grey_scale:02x}'


def activate_initial_nodes(active, num_initial_active):
    """Randomly activate a set number of nodes and return their indices."""
    initial_active_nodes = np.random.choice(len(active), num_initial_active, replace=False)
    active[initial_active_nodes] = True
    return initial_active_nodes  # Return the indices of initially activated nodes

    
    
def step(G, thresholds, active):
    """Perform one step of the LTM using vectorized operations."""
    to_activate = np.zeros_like(active, dtype=bool)

    for node in range(active.size):
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            # Calculate the fraction of active neighbors
            fraction_active_neighbors = sum(active[neighbors]) / len(neighbors)

            # Activate the node if the fraction of active neighbors is greater than or equal to its threshold
            if fraction_active_neighbors >= thresholds[node] and not active[node]:
                to_activate[node] = True

    active[to_activate] = True

    return np.any(to_activate)  # Return True if any new node was activated



def create_threshold_legend():
    """Create and save a legend for different threshold values."""
    # Create a list of thresholds to display in the legend
    example_thresholds = [0, 0.25, 0.5, 0.75, 1.0]
    
    # Create corresponding grey colors for each threshold
    colors = [threshold_to_grey(th) for th in example_thresholds]

    # Create legend patches
    patches = [mpatches.Patch(color=colors[i], label=f'Threshold: {example_thresholds[i]}') for i in range(len(colors))]

    # Create the legend
    plt.figure(figsize=(5, 3))
    plt.legend(handles=patches, title="Thresholds")
    
    # Save legend to file
    plt.savefig("threshold_legend.png")
    plt.close()


def visualize_with_graphviz(G, active, thresholds, initial_active_nodes, current_step):
    """Visualize the network using Graphviz with specific styles for initial and active nodes."""
    dot = Graph(comment=f'LTM Step {current_step}', engine='neato')

    # Adjust global graph attributes
    dot.attr(overlap='false', size='10,10', dpi='100')

    for node in G.nodes:
        border_color = 'green' if node in initial_active_nodes else threshold_to_grey(thresholds[node])
        fillcolor = 'black' if active[node] else 'white'
        dot.node(str(node), '', style='filled', shape='circle', color=border_color, fillcolor=fillcolor, width='0.3', height='0.3', penwidth='5')

    for u, v in G.edges:
        dot.edge(str(u), str(v), color='black', penwidth='2')

    output_path = f'ltm_step_{current_step}'
    dot.render(output_path, format='png', cleanup=True)
    return output_path + '.png'




def run_simulation(num_nodes, mean_degree, num_initial_active, max_steps):
    G, thresholds, active = initialize_network(num_nodes, mean_degree)
    initial_active_nodes = activate_initial_nodes(active, num_initial_active)  # Capture the returned value
    
    images = []
    file_paths = []
    for current_step in range(max_steps):
        img_path = visualize_with_graphviz(G, active, thresholds, initial_active_nodes, current_step)
        file_paths.append(img_path)
        images.append(imageio.imread(img_path))
        if not step(G, thresholds, active):
            break

    imageio.mimsave('ltm_simulation.gif', images, duration=0.5)

    # Clean up individual step images
    for path in file_paths:
        os.remove(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Linear Threshold Model Simulation')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes in the graph')
    parser.add_argument('--mean_degree', type=float, default=3.0, help='Mean degree of the nodes in the graph')
    parser.add_argument('--num_initial_active', type=int, default=1, help='Number of initially active nodes')
    parser.add_argument('--max_steps', type=int, default=12, help='Maximum number of steps in the simulation')

    args = parser.parse_args()

    run_simulation(args.num_nodes, args.mean_degree, args.num_initial_active, args.max_steps)


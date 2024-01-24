import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio.v2 as imageio
import os
from graphviz import Graph
import itertools
from graphviz import Digraph
from PIL import Image

def initialize_network(num_nodes, mean_degree):
    """Initialize a random network with given number of nodes and mean degree."""
    edge_prob = mean_degree / (num_nodes - 1)
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    thresholds = np.random.rand(num_nodes)  # Vector of thresholds
    active = np.zeros(num_nodes, dtype=bool)  # Vector to track active nodes
    return G, thresholds, active

def activate_initial_nodes(active, num_initial_active):
    """Randomly activate a set number of nodes."""
    initial_active_nodes = np.random.choice(active.size, num_initial_active, replace=False)
    active[initial_active_nodes] = True

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

def threshold_to_color(threshold):
    """Map a threshold value to a color."""
    # Example: simple mapping from threshold to a shade of red
    intensity = int(255 * threshold)
    return f'#{intensity:02x}0000'

def threshold_to_grey(threshold):
    """Map a threshold value to a shade of grey."""
    # Convert threshold to a grey scale (0 is light grey, 1 is black)
    grey_scale = int((1 - threshold) * 255)
    return f'#{grey_scale:02x}{grey_scale:02x}{grey_scale:02x}'

def create_truth_table_graph(num_input_nodes, current_input_comb):
    tt = Digraph('truth_table')
    tt.attr('node', shape='plaintext')

    # Generate all combinations for the truth table
    combinations = list(itertools.product([False, True], repeat=num_input_nodes))
    
    # Node labels (a, b, c, ...)
    node_labels = [chr(i) for i in range(97, 97 + num_input_nodes)]
    
    # Create the table content
    table_content = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
    table_content += '<TR>' + ''.join(f'<TD>{label}</TD>' for label in node_labels) + '</TR>'
    for comb in combinations:
        if comb == current_input_comb:
            # Highlight the current combination
            table_content += '<TR>' + ''.join('<TD BGCOLOR="lightblue">{}</TD>'.format(int(val)) for val in comb) + '</TR>'
        else:
            table_content += '<TR>' + ''.join('<TD>{}</TD>'.format(int(val)) for val in comb) + '</TR>'
    table_content += '</TABLE>'

    # Add the table to the graph
    tt.node('tt', f'<{table_content}>')
    return tt

def visualize_with_graphviz(G, active, thresholds, current_step, num_input_nodes, input_combination):
    """Visualize the network using Graphviz with green borders for input nodes, a title with the input combination, and node thresholds."""
    dot = Graph(comment=f'LTM Step {current_step}', engine='neato', format='png')

    # Create a title with the input combination
    input_comb_str = ', '.join(['1' if i else '0' for i in input_combination])
    title = f"Input Combination: {input_comb_str}"

    # Add title to the graph
    dot.attr(label=title, fontsize='20')

    # Adjust global graph attributes
    dot.attr(overlap='false', size='10,10', dpi='100')

    for node in G.nodes:
        label = chr(97 + node) if node < num_input_nodes else ''
        border_color = threshold_to_grey(thresholds[node]) if node >= num_input_nodes else 'green'
        fillcolor = 'black' if active[node] else 'white'
        font_color = 'grey' if node < num_input_nodes else 'black'
        font_size = '20'
        dot.node(str(node), label, style='filled', shape='circle', color=border_color, fillcolor=fillcolor, fontcolor=font_color, fontsize=font_size, width='0.3', height='0.3', penwidth='5')

    for u, v in G.edges:
        dot.edge(str(u), str(v), color='black', penwidth='2')

    output_path = f'ltm_step_{current_step}'
    dot.render(output_path, cleanup=True)
    return output_path + '.png'


def combine_graphs(network_graph_path, truth_table_graph_path, output_path, padding=20):
    # Open the images
    network_img = Image.open(network_graph_path)
    truth_table_img = Image.open(truth_table_graph_path)

    # Calculate the desired height for the truth table image to match the height of the network graph image
    desired_height = network_img.height

    # Calculate the new width while maintaining the aspect ratio
    truth_table_width = int(truth_table_img.width * (desired_height / truth_table_img.height))

    # Resize the truth table image with the calculated width and height
    truth_table_img = truth_table_img.resize((truth_table_width, desired_height))

    # Calculate the total width and height including padding
    total_width = network_img.width + truth_table_img.width + 2 * padding
    max_height = max(network_img.height, truth_table_img.height) + 2 * padding

    # Calculate individual paddings for top, bottom, left, and right
    top_padding = padding
    bottom_padding = max_height - network_img.height - padding
    left_padding = padding
    right_padding = total_width - truth_table_img.width - padding

    # Create a new image with the appropriate size and white background
    combined_img = Image.new('RGB', (total_width, max_height), color='white')

    # Paste the images with padding on all sides
    combined_img.paste(truth_table_img, (left_padding, top_padding))
    combined_img.paste(network_img, (truth_table_img.width + left_padding, top_padding))

    # Save the combined image
    combined_img.save(output_path)

    

    

def run_simulation(num_nodes, num_input_nodes, mean_degree, max_steps):
    G, thresholds, active = initialize_network(num_nodes, mean_degree)
    # Freeze the network and thresholds here

    combined_images = []

    # Iterate over all 2^k combinations of input nodes
    for idx, input_combination in enumerate(itertools.product([False, True], repeat=num_input_nodes)):
        # Reset active states for each combination
        active[:] = False
        active[:num_input_nodes] = input_combination

        # Run the cascade to completion
        while step(G, thresholds, active):
            pass

        # Save the final state for the network graph
        network_graph_path = visualize_with_graphviz(G, active, thresholds, idx, num_input_nodes, input_combination)

        # Generate and save the truth table graph
        truth_table_graph = create_truth_table_graph(num_input_nodes, input_combination)
        truth_table_path = f"truth_table_{idx}.png"
        truth_table_graph.render(truth_table_path, format='png', cleanup=True)

        # Combine the network graph and the truth table
        combined_path = f"combined_{idx}.png"
        combine_graphs(network_graph_path, truth_table_path + ".png", combined_path)
        combined_images.append(combined_path)

        # Cleanup individual images if desired
        os.remove(network_graph_path)
        os.remove(truth_table_path + ".png")

    # Create and save the animation
    images_for_animation = [imageio.imread(img_path) for img_path in combined_images]
    imageio.mimsave('ltm_simulation.gif', images_for_animation, duration=1)

    # Cleanup combined images
    for img_path in combined_images:
        os.remove(img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Linear Threshold Model Simulation')
    parser.add_argument('--num_nodes', type=int, default=10, help='Number of nodes in the graph')
    parser.add_argument('--num_input_nodes', type=int, default=2, help='Number of input nodes')
    parser.add_argument('--mean_degree', type=float, default=3.0, help='Mean degree of the nodes in the graph')
    parser.add_argument('--max_steps', type=int, default=12, help='Maximum number of steps in the simulation')

    args = parser.parse_args()
    
    # Display a message if using default arguments
    if not any(vars(args).values()):
        print("Running with default arguments: num_nodes=10, num_input_nodes=2, mean_degree=3.0, max_steps=12")

    run_simulation(args.num_nodes, args.num_input_nodes, args.mean_degree, args.max_steps)


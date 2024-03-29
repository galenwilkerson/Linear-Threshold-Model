<!DOCTYPE html>
<html>
<body>
  <figure>
     <div style="width: 100%;">
    <img src="zoo/ltm_simulation_N=50_k=5_z=1.7.gif" width="600" alt="LTM Logic">
    <img src="zoo/ltm_simulation_defaults.gif" width="450" alt="LTM Logic">
    <img src="zoo/combined_graph_legend_defaults.png" width="200" alt="LTM Logic">
    <img src="zoo/gcc_ltm_animation.gif" width="600" alt="Second Image">
</div>
    <figcaption>Linear Threshold Model simulation.  First and Second: Creating a network then activating seed nodes as the inputs, then running cascades yields patterns of activation corresponding to Boolean logic functions at network nodes.
      Third:  The functions computed in this LTM.
      Third: Cascade sizes correspond to percolation - formation of the Giant Connected Component (GCC).</figcaption>
</figure>


 

<div class="title-image-container">
  <div class="title-text">
    <h1>Linear Threshold Model:</h1>
    <h2>Network Cascades and Boolean Functions Toolkit</h2>
  </div>
 
</div>

<p>This repository contains a Python module that provides a toolkit for working with network cascades, adjacency matrices, Boolean functions, and network visualization. It is designed to assist in various network-related tasks and analyses.</p>

<h2>Features</h2>
<ul>
  <li>Setup and execution of cascades on networks.</li>
  <li>Creation and manipulation of adjacency matrices.</li>
  <li>Generation and manipulation of truth tables for Boolean functions.</li>
  <li>Visualization of network structures.</li>
  <li>Support for different types of networks, including random geometric graphs.</li>
  <li>Utilizes external libraries like NumPy, SciPy, NetworkX, Matplotlib, Pandas, Seaborn, Networkit, and Pynauty.</li>
  <li>Includes both vectorized and non-vectorized implementations of cascade algorithms.</li>
  <li>Organized into sections for different types of functions.</li>
  <li>Some experimental or incomplete functions (exercise caution in production use).</li>
</ul>

<h2>Sections</h2>
<ol>
  <li><strong>Complexity Functions</strong></li>
  <li><strong>Utility Functions</strong></li>
  <li><strong>Network Visualization</strong></li>
  <li><strong>Setup Functions</strong></li>
  <li><strong>Cascade Functions</strong></li>
  <li><strong>Cascade Analysis</strong></li>
  <li><strong>Plotting Functions</strong></li>
  <li><strong>Binary Array Operations</strong></li>
  <li><strong>Truth Table Functions</strong></li>
  <li><strong>Monte Carlo Trials</strong></li>
  <li><strong>Boolean Function Analysis</strong></li>
  <li><strong>Range of Monte Carlo Functions</strong></li>
  <li><strong>Hamming Distance and Cube, and Boolean Logic Functions</strong></li>
  <li><strong>Graph Measures</strong></li>
  <li><strong>Data Generation</strong></li>
</ol>

<h2>Usage</h2>
<p>To use this toolkit, import the relevant functions and classes from the module into your Python code. You can then leverage the provided tools for your network-related tasks.</p>

<h2>Dependencies</h2>
<p>This module relies on several external libraries, including NumPy, SciPy, NetworkX, Matplotlib, Pandas, Seaborn, Networkit, and Pynauty. Make sure to install these dependencies before using the toolkit.</p>

<h2>Research Papers</h2>
<p>This toolkit is based on research presented in the following papers:</p>

<ol>
  <li><a href="https://www.nature.com/articles/s41598-022-19218-0">Spontaneous emergence of computation in network cascades</a>
    <ul>
      <li>Authors: Wilkerson, Galen; Moschoyiannis, Sotiris; Jensen, Henrik Jeldtoft</li>
      <li>Published in Scientific Reports (2022)</li>
    </ul>
  </li>
  <li><a href="https://www.cambridge.org/core/journals/network-science/article/logic-and-learning-in-network-cascades/B89A3EB13FF6F1719482D38F11E37068">Logic and learning in network cascades</a>
    <ul>
      <li>Authors: Wilkerson, Galen J.; Moschoyiannis, Sotiris</li>
      <li>Published in Network Science (2021)</li>
    </ul>
  </li>
</ol>

<p><strong>Note</strong>: If you use this toolkit in your research, we kindly request that you cite the relevant research papers listed above to acknowledge the authors' contributions.</p>

<h2>Note</h2>
<p>Please be aware that this module contains a mix of vectorized and non-vectorized cascade algorithms, along with some experimental or incomplete functions. Before using it in a production environment, review the code carefully and exercise caution.</p>
<h2>LTM Animation Vectorized</h2>
<figure>
    <img src="zoo/ltm_simulation__1.gif" width="200" alt="LTM Simulation">
    <figcaption>Linear Threshold Model simulation. The grey node borders show their thresholds. Green nodes are seed nodes.</figcaption>
</figure>
<br><br>

<p>The <code>ltm_animation_vectorized.py</code> script generates an animation of the Linear Threshold Model (LTM) using a vectorized approach. This script allows for efficient simulation and visualization of LTM dynamics in network structures.</p>

<h3>Usage</h3>
<pre>
python ltm_animation_vectorized.py --h
usage: ltm_animation_vectorized.py [-h] [--num_nodes NUM_NODES] [--mean_degree MEAN_DEGREE] [--num_initial_active NUM_INITIAL_ACTIVE] [--max_steps MAX_STEPS]
                                   [--uniform_threshold UNIFORM_THRESHOLD]

Run Linear Threshold Model Simulation

options:
  -h, --help            show this help message and exit
  --num_nodes NUM_NODES
                        Number of nodes in the graph
  --mean_degree MEAN_DEGREE
                        Mean degree of the nodes in the graph
  --num_initial_active NUM_INITIAL_ACTIVE
                        Number of initially active nodes
  --max_steps MAX_STEPS
                        Maximum number of steps in the simulation
  --uniform_threshold UNIFORM_THRESHOLD
                        Uniform threshold for all nodes. If not set, thresholds are random.
</pre>


<h2>LTM Logic Animation Vectorized</h2>

<figure>
    <img src="zoo/ltm_logic.gif" width="300" alt="LTM Simulation">
    <figcaption> Linear Threshold Model logic simulation, showing activation patterns on the same network when running a cascade for each row of inputs.  The grey node borders show their thresholds.  Green nodes are input nodes.</figcaption>
</figure>
<br><br>

<p>The <code>ltm_logic_animation_vectorized.py</code> script offers a vectorized method to visualize the logic behind the Linear Threshold Model (LTM). It focuses on the logical aspects and decision-making processes within the LTM framework.</p>

<h3>Usage</h3>
<pre>
python ltm_logic_animation_vectorized.py --h
usage: ltm_logic_animation_vectorized.py [-h] [--num_nodes NUM_NODES] [--num_input_nodes NUM_INPUT_NODES] [--mean_degree MEAN_DEGREE] [--max_steps MAX_STEPS]

Run Linear Threshold Model Simulation

options:
  -h, --help            show this help message and exit
  --num_nodes NUM_NODES
                        Number of nodes in the graph
  --num_input_nodes NUM_INPUT_NODES
                        Number of input nodes
  --mean_degree MEAN_DEGREE
                        Mean degree of the nodes in the graph
  --max_steps MAX_STEPS
                        Maximum number of steps in the simulation
</pre>


<h2>Show an animation of the correpondence between percolation and cascade size.</h2>
<figure>
    <img src="zoo/gcc_ltm_animation.gif" width="700" alt="LTM Simulation">
    <figcaption> The correspondence between giant connected component size and mean cascade size, versus mean degree z.  Note how the size of the giant component in the figure to the right gets much larger as z reaches and exceeds z_critical = 1.</figcaption>
</figure>
<br><br>


<p>To run the <code>show_GCC_versus_cascade_size.py</code> script for simulating and visualizing the Linear Threshold Model, you can use the following command line options:</p>

<pre>
python gcc_versus_cascade_size_animation.py --h
usage: gcc_versus_cascade_size_animation.py [-h] [--n N] [--z_start Z_START] [--z_end Z_END] [--num_trials NUM_TRIALS] [--threshold THRESHOLD] [--num_simulations NUM_SIMULATIONS]
                                            [--plot_data {GCC,Cascade,Both}]

Run Linear Threshold Model Simulation

options:
  -h, --help            show this help message and exit
  --n N                 Number of nodes in the graph (default: 100)
  --z_start Z_START     Start of the mean degree range (default: 0)
  --z_end Z_END         End of the mean degree range (default: 5)
  --num_trials NUM_TRIALS
                        Number of z values to simulate (default: 20)
  --threshold THRESHOLD
                        Activation threshold for the nodes (default: 0.1)
  --num_simulations NUM_SIMULATIONS
                        Number of simulations per z value (default: 10)
  --plot_data {GCC,Cascade,Both}
                        Specify what to plot: GCC, Cascade, or Both (default: Both)
</pre>

<h2>Show the correpondence between percolation and cascade size.</h2>
<figure>
    <img src="zoo/GCC_cascade.png" width="500" alt="LTM Simulation">
    <figcaption> The correspondence between giant connected component size and mean cascade size, versus mean degree z.</figcaption>
</figure>
<br><br>


<p>To run the <code>show_GCC_versus_cascade_size.py</code> script for simulating and visualizing the Linear Threshold Model, you can use the following command line options:</p>

<pre>
python show_GCC_versus_cascade_size.py --h
usage: show_GCC_versus_cascade_size.py [-h] [--n N] [--z_start Z_START] [--z_end Z_END] [--num_z_steps NUM_Z_STEPS] [--threshold THRESHOLD] [--num_trials NUM_TRIALS]
                                       [--plot_data {GCC,Cascade,Both}]

Run Linear Threshold Model Simulation

options:
  -h, --help            show this help message and exit
  --n N                 Number of nodes in the graph (default: 100)
  --z_start Z_START     Start of the mean degree range (default: 0)
  --z_end Z_END         End of the mean degree range (default: 5)
  --num_z_steps NUM_Z_STEPS
                        Number of z values to simulate (default: 20)
  --threshold THRESHOLD
                        Activation threshold for the nodes (default: 0.1)
  --num_trials NUM_TRIALS
                        Number of simulations per z value (default: 10)
  --plot_data {GCC,Cascade,Both}
                        Specify what to plot: GCC, Cascade, or Both (default: Both)
</pre>

<h2>License</h2>
<p>This toolkit is provided under the <a href="LICENSE">MIT License</a>.</p>



 <div class="header">
        <h1>Linear Threshold Model Zoo</h1>
    </div>
    <div class="container">
        <!-- Add your images here -->
        <img src="zoo/ltm_simulation10.gif" alt="LTM Simulation 10">
        <img src="zoo/ltm_simulation11.gif" alt="LTM Simulation 11">
        <img src="zoo/ltm_simulation12.gif" alt="LTM Simulation 12">
        <img src="zoo/ltm_simulation13.gif" alt="LTM Simulation 13">
        <img src="zoo/ltm_simulation_1.gif" alt="LTM Simulation 1">
        <img src="zoo/ltm_simulation2.gif" alt="LTM Simulation 2">
        <img src="zoo/ltm_simulation3.gif" alt="LTM Simulation 3">
        <img src="zoo/ltm_simulation4.gif" alt="LTM Simulation 4">
        <img src="zoo/ltm_simulation5.gif" alt="LTM Simulation 5">
        <img src="zoo/ltm_simulation6.gif" alt="LTM Simulation 6">
        <img src="zoo/ltm_simulation7.gif" alt="LTM Simulation 7">
        <img src="zoo/ltm_simulation8.gif" alt="LTM Simulation 8">
        <img src="zoo/ltm_simulation9.gif" alt="LTM Simulation 9">
    </div>



</body>
</html>



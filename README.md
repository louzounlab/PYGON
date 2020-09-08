# PYGON

A Graph Neural Network Tool for Recovering Dense Sub-graphs in Random Dense Graphs.
 
### Installation

This code requires to install and run the [graph-measures](https://github.com/louzounlab/graph-measures) package. Currently, we have a copy of this package that is ready to use (in "graph_calculations" directory), but it is possible to remove its content, download the graph_measures repository and follow the instructions below to be able to run this code. The conda environment for this project (part 2 in the instructions) is still required. \
A detailed explanation for graph-measures package appears in a manual in graph-measures repository. Here we present short instructions:

1. Download the graph-measures project into "graph_calculations/graph_measures".
2. Create the anaconda environment for running this project by running _conda env create -f env.yml_ in the terminal.
3. Activate the new environment: _conda activate boost_.
4. Move into the directory "graph_calculations/graph_measures/features_algorithms/accelerated_graph_features/src".
5. Make the feature calculation files for motif calculations: _make -f Makefile-gpu_.
6. Great! Now one should be able to run PYGON end-to-end. Remember to work in boost environment when using this code.

Note that this code was tested only on Unix machines with GPUs. Some feature calculations might not work in other machines. \
Note also that the virtual environment we tried is anaconda-based.

### How to Use 

* The main code directory is "model". The other directory includes the code for feature calculations and will include 
saved pickle files of graphs and their features.

* For simply trying the PYGON model, one can run _python pygon_main.py_ in a terminal. This will run a simple training 
of PYGON on G(500, 0.5, 20) graphs (which will be built and dumped in "graph_calculations/pkl"). 
One can change the parameters or graph specifications appearing there to try PYGON on graphs of other sizes, 
edge probabilities, planted sub-graph sizes, planted sub-graph types or model hyper-parameters.

* More detailed performance tests can be found in _performance_testing.py_. 

* To run an [NNI](https://github.com/microsoft/nni) experiment on the performance of PYGON, move into "to_run_nni" and 
run the configuration the experiment as guided in NNI's documentation.

* The existing algorithms to which we compared our performance, as well as a faster version of PYGON 
(without dumping or printing anything), can be found in _other_algorithms.py_.

* The cleaning stage using the cleaning algorithm can be found in _second_stage.py_.


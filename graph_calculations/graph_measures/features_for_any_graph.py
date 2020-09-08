import datetime
import networkx as nx
import numpy as np
import os
import pickle
import sys
import torch
import logging

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_algorithms', 'accelerated_graph_features'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_algorithms', 'vertices'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_infra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'graph_infra'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_processor'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'features_meta'))
from loggers import PrintLogger, FileLogger, multi_logger
from graph_features import GraphFeatures


class FeatureCalculator:
    def __init__(self, edge_path, dir_path, features, acc=True, directed=False, gpu=False, device=2, verbose=True,
                 params=None):
        """
        :param edge_path: str
        Path to graph edges file (text-like file, e.g. txt or csv), from which the graph is built using networkx.
        The graph must be unweighted. If its vertices are not [0, 1, ..., n-1], they are mapped to become
        [0, 1, ..., n-1] and the mapping is saved.
        Every row in the edges file should include "source_id,distance_id", without a header row.
        :param dir_path: str
        Path to the directory in which the feature calculations will be (or already are) located.
        :param features: list of strings
        List of the names of each feature. Could be any name from features_meta.py or "additional_features".
        :param acc: bool
        Whether to run the accelerated features, assuming it is possible to do so.
        :param directed: bool
        Whether the built graph is directed.
        :param gpu: bool
        Whether to use GPUs, assuming it is possible to do so (i.e. the GPU exists and the CUDA matches).
        :param device: int
        If gpu is True, indicates on which GPU device to calculate. Will return error if the index doesn't match the
        available GPUs.
        :param verbose: bool
        Whether to print things indicating the phases of calculations.
        :param params: dict, or None
        For clique detection uses, this is a dictionary of the graph settings
        (size, directed, clique size, edge probability). Ignored for any other use.
        """

        self._dir_path = dir_path
        self._features = features  # By their name as appears in accelerated_features_meta
        self._gpu = gpu
        self._device = device
        self._verbose = verbose
        self._logger = multi_logger([PrintLogger("Logger", level=logging.DEBUG), FileLogger("FLogger", path=dir_path, level=logging.INFO)], name=None) if verbose else None
        self._params = params
        self._load_graph(edge_path, directed)
        self._get_feature_meta(features, acc)  # acc determines whether to use the accelerated features

        self._adj_matrix = None
        self._raw_features = None
        self._other_features = None

    def _load_graph(self, edge_path, directed=False):
        self._graph = nx.read_edgelist(edge_path, delimiter=',', create_using=nx.DiGraph() if directed else nx.Graph())
        vertices = np.array(self._graph.nodes)
        should_be_vertices = np.arange(len(vertices))
        if not np.array_equal(vertices, should_be_vertices):
            if self._verbose:
                self._logger.debug("Relabeling vertices to [0, 1, ..., n-1]")
            mapping = {i: v for i, v in enumerate(self._graph)}
            pickle.dump(mapping, open(os.path.join(self._dir_path, "vertices_mapping.pkl"), "wb"))
            self._graph = nx.convert_node_labels_to_integers(self._graph)
        if self._verbose:
            self._logger.info(str(datetime.datetime.now()) + " , Loaded graph")
            self._logger.debug("Graph Size: %d Nodes, %d Edges" % (len(self._graph), len(self._graph.edges))) 

    def _get_feature_meta(self, features, acc):
        if acc:
            from accelerated_features_meta import FeaturesMeta
        else:
            from features_meta import FeaturesMeta

        all_node_features = FeaturesMeta(gpu=self._gpu, device=self._device).NODE_LEVEL
        self._features = {}
        self._special_features = []
        for key in features:
            if key in ['degree', 'in_degree', 'out_degree', 'additional_features']:
                self._special_features.append(key)
            elif key not in all_node_features:
                if self._verbose:
                    self._logger.debug("Feature %s unknown, ignoring this feature" % key)
                features.remove(key)
                continue
            else:
                self._features[key] = all_node_features[key]

    def calculate_features(self):
        if not len(self._features) + len(self._special_features) and self._verbose:
            print("No features were chosen!")
        else:
            self._adj_matrix = nx.adjacency_matrix(self._graph)
            # self._adj_matrix = self._adj_matrix.toarray()
            self._raw_features = GraphFeatures(gnx=self._graph, features=self._features, dir_path=self._dir_path,
                                               logger=self._logger)
            self._raw_features.build(should_dump=True)  # The option of multiple workers in this function exists.
            self._other_features = OtherFeatures(self._graph, self._special_features, self._dir_path, self._params,
                                                 self._logger)
            self._other_features.build(should_dump=True)
            self._logger.info(str(datetime.datetime.now()) + " , Calculated features")

    @property
    def feature_matrix(self):
        return np.hstack((self._raw_features.to_matrix(mtype=np.array), self._other_features.feature_matrix))

    @property
    def adjacency_matrix(self):
        return self._adj_matrix


class OtherFeatures:
    def __init__(self, graph, features, dir_path, params=None, logger=None):
        self._graph = graph
        self._features = features
        self._dir_path = dir_path
        self._logger = logger
        self._params = params
        self._feat_string_to_function = {
            'degree': self._calc_degree,
            'in_degree': self._calc_in_degree,
            'out_degree': self._calc_out_degree,
            'additional_features': self._calc_additional_features
        }

        self._feature_matrix = None

    def _calc_degree(self):
        degrees = list(self._graph.degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_in_degree(self):
        degrees = list(self._graph.in_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_out_degree(self):
        degrees = list(self._graph.out_degree)
        return np.array([d[1] for d in degrees]).reshape(-1, 1)

    def _calc_additional_features(self):
        from additional_features import AdditionalFeatures
        if self._params is None:
            raise ValueError("params is None")
        if not os.path.exists(os.path.join(self._dir_path, "motif3.pkl")):
            raise FileNotFoundError("Motif 3 must be calculated")
        if not os.path.exists(os.path.join(self._dir_path, "motif4.pkl")):
            raise FileNotFoundError("Motif 4 must be calculated")

        motif_matrix = np.hstack((pickle.load(open(os.path.join(self._dir_path, "Motif_3.pkl"), "rb")),
                                  pickle.load(open(os.path.join(self._dir_path, "Motif_4.pkl"), "rb"))))
        add_ftrs = AdditionalFeatures(self._params, self._graph, self._dir_path, motif_matrix)
        return add_ftrs.calculate_extra_ftrs()

    def build(self, should_dump=False):
        self._feature_matrix = np.empty((len(self._graph), 0))
        for feat_str in self._features:
            if self._logger:
                start_time = datetime.datetime.now()
                self._logger.info("Start %s" % feat_str)
            if os.path.exists(os.path.join(self._dir_path, feat_str + '.pkl')) and feat_str != "additional_features":
                feat = pickle.load(open(os.path.join(self._dir_path, feat_str + ".pkl"), "rb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
                if self._logger:
                    cur_time = datetime.datetime.now()
                    self._logger.info("Finish %s at %s" % (feat_str, cur_time - start_time))
            else:
                feat = self._feat_string_to_function[feat_str]()
                if should_dump:
                    pickle.dump(feat, open(os.path.join(self._dir_path, feat_str + ".pkl"), "wb"))
                self._feature_matrix = np.hstack((self._feature_matrix, feat))
                if self._logger:
                    cur_time = datetime.datetime.now()
                    self._logger.info("Finish %s at %s" % (feat_str, cur_time - start_time))

    @property
    def feature_matrix(self):
        return self._feature_matrix


if __name__ == "__main__":
    head = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trial_inbar", "250mb_graph1")
    path = os.path.join(head, "random_edges1.txt")
    # feats = ["average_neighbor_degree", "betweenness_centrality", "bfs_moments",
    #          "closeness_centrality", "communicability_betweenness_centrality",
    #          "eccentricity", "fiedler_vector", "k_core", "load_centrality",
    #          "louvain", "motif3", "motif4", "degree", "additional_features"]
    feats = ["k_core", "louvain", "motif3", "motif4", "degree", "average_neighbor_degree", "louvain", "bfs_moments"]
    ftr_calc = FeatureCalculator(path, head, feats, acc=True, directed=False, gpu=True,
                                 device=3, verbose=True)
    ftr_calc.calculate_features()
    
    # head = os.path.join(os.path.dirname(__file__), "trial_inbar", "10mb_graph1")
    # path = os.path.join(head, "random_edges1.txt")
    # feats = ["average_neighbor_degree", "betweenness_centrality", "bfs_moments",
    #          "closeness_centrality", "communicability_betweenness_centrality",
    #          "eccentricity", "fiedler_vector", "k_core", "load_centrality",
    #          "louvain", "motif3", "motif4", "degree", "additional_features"]
    # ftr_calc = FeatureCalculator(path, head, feats, acc=True, directed=False, gpu=True,
    #                              device=0, verbose=True)
    # ftr_calc.calculate_features()
    #
    # head = os.path.join(os.path.dirname(__file__), "trial_inbar", "250mb_graph1")
    # path = os.path.join(head, "random_edges1.txt")
    # feats = ["average_neighbor_degree", "betweenness_centrality", "bfs_moments",
    #          "closeness_centrality", "communicability_betweenness_centrality",
    #          "eccentricity", "fiedler_vector", "k_core", "load_centrality",
    #          "louvain", "motif3", "motif4", "degree", "additional_features"]
    # feats = ["k_core", "motif3", "motif4", "degree", "average_neighbor_degree", "louvain", "degree", "bfs_moments"]
    # ftr_calc = FeatureCalculator(path, head, feats, acc=True, directed=False, gpu=True,
    #                              device=03, verbose=True)
    # ftr_calc.calculate_features()

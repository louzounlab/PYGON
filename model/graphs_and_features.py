import datetime
import networkx as nx
import itertools
import pickle
import torch
from __init__ import *
from utils import *
from graph_calculations import *
from betweenness_centrality import BetweennessCentralityCalculator
from accelerated_graph_features.bfs_moments import BfsMomentsCalculator
from feature_calculators import FeatureMeta
from graph_features import GraphFeatures
from accelerated_graph_features.motifs import nth_nodes_motif, MotifsNodeCalculator
from additional_features import AdditionalFeatures, MotifProbability


class ER:
    def __init__(self, params):
        self._nodes = params['vertices']
        self._p = params['probability']
        self._is_directed = params['directed']

        self._graph = self.build()

    def build(self):
        return nx.gnp_random_graph(self._nodes, self._p, directed=self._is_directed)

    def graph(self):
        return self._graph


class PlantSubgraph:
    def __init__(self, graph, params):
        self._graph = graph
        self._subgraph_size = params['subgraph_size']
        self._graph_size = params['vertices']
        self._is_directed = params['directed']
        self._subgraph_vertices = []
        self.plant(params['subgraph'])

    def _is_valid(self, subgraph):
        assert self._subgraph_size < self._graph_size, \
            "The subgraph size is larger than the graph size"
        assert subgraph != 'dag-clique' or self._is_directed, \
            "For planting a DAG-clique, the graph must be directed"
        assert subgraph != 'k-plex' or not self._is_directed, \
            "For planting a k-plex, the graph must be undirected"

    def plant(self, subgraph):
        self._is_valid(subgraph)
        self._subgraph_vertices = np.random.choice(self._graph.nodes(), self._subgraph_size,
                                                   replace=False).tolist()
        ebunch = itertools.permutations(self._subgraph_vertices, 2) if self._is_directed else \
            itertools.combinations(self._subgraph_vertices, 2)
        self._graph.remove_edges_from(ebunch)
        if subgraph == "clique":
            self._plant_clique()
        elif subgraph == "dag-clique":
            self._plant_dag_clique()
        elif subgraph == "k-plex":
            self._plant_k_plex()
        elif subgraph == "biclique":
            self._plant_biclique()
        elif subgraph.startswith("G(k,"):
            self._plant_gkp(float(subgraph.split(" ")[-1][:-1]))
        else:
            raise ValueError("Wrong choice of subgraph")

    def _plant_clique(self):
        ebunch = itertools.permutations(self._subgraph_vertices, 2) if self._is_directed else \
            itertools.combinations(self._subgraph_vertices, 2)
        self._graph.add_edges_from(ebunch)

    def _plant_dag_clique(self):
        # A clique in the undirected version of the graph that is a directed acyclic graph in the directed version.
        # The choice of vertices defines an order over the vertices, so we will draw edges from a vertex to all the
        # following vertices by the list.
        for i, j in itertools.combinations(self._subgraph_vertices, 2):
            self._graph.add_edge(i, j)

    def _plant_k_plex(self):
        # A k-plex is a graph H in which each vertex v has at least |H| - k neighbors.
        # Clique is a special case of k-plex when k = 1.
        # Here we will take a k-plex such that if |H| is even then each vertex will miss one neighbor,
        # and if |H| is odd then one vertex will be connected to all vertices and the rest miss one.
        # Maybe later k should be an input parameter and the building should be more random.
        pairs_of_missing_vertices = [self._subgraph_vertices[i:i+2] for i in range(0, len(self._subgraph_vertices), 2)]
        self._plant_clique()
        for pair in pairs_of_missing_vertices:
            if len(pair) == 2:
                self._graph.remove_edge(*pair)

    def _plant_biclique(self):
        first, second = np.array_split(self._subgraph_vertices, 2)
        ebunch = [(int(one), int(two)) for one, two in itertools.product(first, second)]
        self._graph.add_edges_from(ebunch)

    def _plant_gkp(self, q):
        # Create a random G(k, q) graph and plant a copy of it in our graph.
        ebunch = nx.gnp_random_graph(n=self._subgraph_size, p=q, directed=self._is_directed).edges
        conversion = {v1: v2 for v1, v2 in zip(range(self._subgraph_size), self._subgraph_vertices)}
        self._graph.add_edges_from((conversion[u], conversion[v]) for u, v in ebunch)

    def graph_sg(self):
        return self._graph

    def subgraph_vertices(self):
        return self._subgraph_vertices


class GraphBuilder:
    def __init__(self, params, dir_path):
        self._params = params
        self._dir_path = dir_path
        self._gnx = None
        self._labels = []
        self._build_er_and_subgraph()
        self._label_graph()

    def _build_er_and_subgraph(self):
        if os.path.exists(os.path.join(self._dir_path, 'gnx.pkl')):
            self._gnx = pickle.load(open(os.path.join(self._dir_path, 'gnx.pkl'), 'rb'))
        else:
            check_make_dir(self._dir_path)
            graph = ER(self._params).graph()
            ps = PlantSubgraph(graph, self._params)
            self._gnx = ps.graph_sg()
            self._subgraph_vertices = ps.subgraph_vertices()
            pickle.dump(self._gnx, open(os.path.join(self._dir_path, 'gnx.pkl'), "wb"))

    def _label_graph(self):
        if os.path.exists(os.path.join(self._dir_path, 'labels.pkl')):
            self._labels = pickle.load(open(os.path.join(self._dir_path, 'labels.pkl'), "rb"))
        else:
            labels = []
            for v in self.vertices():
                labels.append(0 if v not in self._subgraph_vertices else 1)
            self._labels = labels
            pickle.dump(self._labels, open(os.path.join(self._dir_path, 'labels.pkl'), "wb"))
        print(str(datetime.datetime.now()) + " , Built a graph")

    def vertices(self):
        return self._gnx.nodes

    def graph(self):
        return self._gnx

    def labels(self):
        return self._labels


class FeatureCalculator:
    def __init__(self, params, graph, dir_path, features, dump=True, gpu=False, device=2):
        self._params = params
        self._graph = graph
        self._dir_path = dir_path
        self._features = features
        self._feat_string_to_function = {
            'Degree': self._calc_degree,
            'In-Degree': self._calc_in_degree,
            'Out-Degree': self._calc_out_degree,
            'Betweenness': self._calc_betweenness,
            'BFS': self._calc_bfs,
            'Motif_3': self._calc_motif3,
            'additional_features': self._calc_additional_features
        }
        self._dump = dump
        self._gpu = gpu
        self._device = device
        self._calculate_features()

    def _calculate_features(self):
        # Features are after taking log(feat + small_epsilon).
        # Currently, no external features are possible.
        if not len(self._features):
            self._feature_matrix = np.identity(len(self._graph))
        else:
            self._feature_matrix = np.empty((len(self._graph), 0))
        self._adj_matrix = nx.adjacency_matrix(self._graph).toarray()
        for feat_str in self._features:
            feat = self._feat_string_to_function[feat_str]()
            if self._dump:
                pickle.dump(feat, open(os.path.join(self._dir_path, feat_str + ".pkl"), "wb"))
            self._feature_matrix = np.hstack((self._feature_matrix, feat))
        print(str(datetime.datetime.now()) + " , Calculated features")

    def _calc_degree(self):
        degrees = list(self._graph.degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_in_degree(self):
        degrees = list(self._graph.in_degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_out_degree(self):
        degrees = list(self._graph.out_degree)
        return self._log_norm(np.array([deg[1] for deg in degrees]).reshape(-1, 1))

    def _calc_betweenness(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"betweenness": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=self._dump)
        feature_dict = raw_ftr["betweenness"]._features
        feature_mx = np.zeros((len(feature_dict), 1))
        for i in feature_dict.keys():
            feature_mx[i] = feature_dict[i]
        return self._log_norm(feature_mx)

    def _calc_bfs(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=self._dump)
        feat = raw_ftr["bfs_moments"]._features
        if type(feat) == list:
            feature_mx = np.array(feat)
        else:
            feature_mx = np.zeros((len(feat), len(list(feat.values())[0][0])))
            for i in feat.keys():
                for j in range(len(feat[i][0])):
                    feature_mx[i, j] = feat[i][0][j]
        return self._log_norm(feature_mx)

    def _calc_motif3(self):
        raw_ftr = GraphFeatures(self._graph,
                                {"motif3": FeatureMeta(nth_nodes_motif(3, gpu=self._gpu, device=self._device), {"m3"})},
                                dir_path=self._dir_path)
        raw_ftr.build(should_dump=self._dump)
        feature = raw_ftr['motif3']._features
        if type(feature) == dict:
            motif_matrix = self._to_matrix(feature)
        else:
            motif_matrix = feature
        return self._log_norm(motif_matrix)

    def _calc_additional_features(self):
        # MUST BE AFTER CALCULATING MOTIFS
        assert "Motif_3" is self._features, "The additional features must be computed only with the motifs"
        motif_matrix = pickle.load(open(os.path.join(self._dir_path, "Motif_3.pkl"), "rb"))
        add_ftrs = AdditionalFeatures(self._params, self._graph, motif_matrix)
        return self._log_norm(add_ftrs.calculate_extra_ftrs())

    @staticmethod
    def _to_matrix(motif_features):
        rows = len(motif_features.keys())
        columns = len(motif_features[0])
        final_mat = np.zeros((rows, columns))
        for i in range(rows):
            for j in range(columns):
                final_mat[i, j] = motif_features[i][j]
        return final_mat

    @staticmethod
    def _log_norm(feature_matrix):
        if type(feature_matrix) == list:
            feature_matrix = np.array(feature_matrix)
        feature_matrix[np.isnan(feature_matrix)] = 1e-10
        not_log_normed = np.abs(feature_matrix)
        not_log_normed[not_log_normed < 1e-10] = 1e-10
        return np.log(not_log_normed)

    @property
    def feature_matrix(self):
        return self._feature_matrix

    @property
    def adjacency_matrix(self):
        return self._adj_matrix


def feature_calculation(v, prob, subgraph, sg, di, features, new_runs):
    param_dict = {'vertices': v,
                  'probability': prob,
                  'subgraph': subgraph,  # 'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
                  'subgraph_size': sg,
                  'directed': di,
                  'features': features
                  }
    key_name = (subgraph, f"n_{v}_p_{prob}_size_{sg}_{'d' if di else 'ud'}")
    head_path = os.path.join(os.path.dirname(__file__), '..', 'graph_calculations', 'pkl', subgraph, key_name[1] + '_runs')
    if not os.path.exists(os.path.dirname(head_path)):  # If the directory of the subgraph does not exist, create it.
        os.mkdir(os.path.dirname(head_path))
    if not os.path.exists(head_path):
        os.mkdir(head_path)
        print("Made new directory")
    graph_ids = os.listdir(head_path)
    if len(graph_ids) == 0 and new_runs == 0:
        raise ValueError('No runs here!')

    print(f"Graph count: {len(graph_ids)}")
    for run in range(len(graph_ids) + new_runs):
        dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(run))
        data = GraphBuilder(param_dict, dir_path)
        gnx = data.graph()
        _ = FeatureCalculator(param_dict, gnx, dir_path, param_dict['features'], gpu=True, device=0)

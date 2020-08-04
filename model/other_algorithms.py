"""
This file includes several functions used to create figures in PYGON paper,
including implementations of existing algorithms and an efficient implementation of PYGON algorithm.
"""
import time
from itertools import permutations, product
from functools import partial
import numpy as np
from scipy.linalg import eig, eigh
from scipy.stats import norm
import torch
import networkx as nx
from sklearn.preprocessing import StandardScaler
from __init__ import *
from graph_calculations import *
from graphs_and_features import GraphBuilder, FeatureCalculator
from pygon_model import PYGONModel
from utils import check_make_dir


def graphs_loader(sz, p, sg_sz, subgraph):
    """
    Load or build the graphs without features, for AKS, DGP and DM, and for run time check of PYGON.
    """
    graph_params = {
        'vertices': sz,
        'probability': p,
        'subgraph': subgraph,
        # 'clique', 'dag-clique', 'k-plex', 'biclique' or 'G(k, q)' for G(k, q) with probability q (e.g. 'G(k, 0.9)').
        'subgraph_size': sg_sz,
        'directed': True if subgraph == "dag-clique" else False
    }
    key_name = (subgraph, f"n_{sz}_p_{p}_size_{sg_sz}_{'d' if graph_params['directed'] else 'ud'}")
    head_path = os.path.join(os.path.dirname(__file__), "..", 'graph_calculations', 'pkl', subgraph,
                             key_name[1] + '_runs')
    check_make_dir(head_path)
    graphs, labels = [], []
    for run in range(20):
        dir_path = os.path.join(head_path, key_name[1] + "_run_" + str(run))
        data = GraphBuilder(graph_params, dir_path)
        graphs.append(data.graph())
        lbs = data.labels()

        if type(lbs) == dict:
            new_lbs = [[y for x, y in lbs.items()]]
            labels += new_lbs
        else:
            labels += [lbs]
    return graphs, labels


def dgp_phi_bar(x):
    return 1 - norm.cdf(x)


def dgp_gamma(alpha, eta):
    return alpha * dgp_phi_bar(eta)


def dgp_delta(alpha, eta, c):
    return alpha * dgp_phi_bar(eta - max(c, 1.261) * np.sqrt(alpha))


def dgp_tau(alpha, beta):
    return (1 - alpha) * dgp_phi_bar(beta)


def dgp_rho(alpha, beta, eta, c):
    return (1 - alpha) * dgp_phi_bar(beta - max(c, 1.261) * dgp_delta(alpha, eta, c) / np.sqrt(dgp_gamma(alpha, eta)))


def dgp_choose_s_i(v, alpha):
    out = []
    n = np.random.random_sample((len(v),))
    for i, vert in enumerate(v):
        if n[i] < alpha:
            out.append(vert)
    return out


def dgp_get_si_tilde(graph, si, eta):
    out = []
    for v in si:
        neighbors = set(graph.neighbors(v))
        if len(set(si).intersection(neighbors)) >= 0.5 * len(si) + 0.5 * eta * np.sqrt(len(si)):
            out.append(v)
    return out


def dgp_get_vi(graph, vi_before, si, si_tilde, beta):
    out = []
    for v in set(vi_before).difference(si):
        neighbors = set(graph.neighbors(v))
        if len(set(si_tilde).intersection(neighbors)) >= 0.5 * len(si_tilde) + 0.5 * beta * np.sqrt(len(si_tilde)):
            out.append(v)
    return out


def dgp_get_k_tilde(g_t, alpha, beta, eta, t, c, k):
    out = []
    k_t = np.power(dgp_rho(alpha, beta, eta, c), t) * k
    for v in g_t:
        if g_t.degree(v) >= 0.5 * len(g_t) + 0.75 * k_t:
            out.append(v)
    return out


def dgp_get_k_tag(k_tilde, graph):
    second_set = []
    for v in graph:
        neighbors = set(graph.neighbors(v))
        if len(set(k_tilde).intersection(neighbors)) >= 0.75 * len(k_tilde):
            second_set.append(v)
    return list(set(k_tilde).union(second_set))


def dgp_get_k_star(k_tag, graph, k):
    g_k_tag = nx.induced_subgraph(graph, k_tag)
    vertices = [v for v in g_k_tag]
    degrees = [g_k_tag.degree(v) for v in vertices]
    vertices_order = [vertices[v] for v in np.argsort(degrees)]
    return vertices_order[-2 * k:]


def run_aks(sz, p, sg_sz, subgraph, write=False, writer=None):
    """
    An implementation of the algorithm of Alon, Krivelevich and Sudakov for clique recovery
    using a spectral technique.
    """
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        w = nx.to_numpy_array(graph)
        if np.allclose(w, w.T):
            _, eigvec = eigh(w, eigvals=(sz - 2, sz - 2))
        else:
            eigvec = eig(w, left=False, right=True)[1][:, 1]
        indices_order = np.argsort(np.abs(eigvec).ravel()).tolist()
        subset = indices_order[-2 * sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in subset if labels[v]]))
        # Without the cleaning stage of choosing the vertices connected to at least 3/4 of this subset
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


def run_dgp(sz, p, sg_sz, subgraph, write=False, writer=None):
    """
    An implementation of the algorithm of Dekel, Gurel-Gurevich and Peres for clique recovery
    using degree-based measurements.
    """
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        alpha, beta, eta = 0.8, 2.3, 1.2
        eps_4 = 1. / alpha - 1e-8
        t = eps_4 * np.log(sz) / np.log(
            np.power(dgp_rho(alpha, beta, eta, sg_sz / np.sqrt(sz)), 2) / dgp_tau(alpha, beta))
        t = np.floor(t)
        v_i = [v for v in range(len(labels))]
        # First Stage #
        for _ in range(int(t)):
            s_i = dgp_choose_s_i(v_i, alpha)
            s_i_tilde = dgp_get_si_tilde(graph, s_i, eta)
            new_vi = dgp_get_vi(graph, v_i, s_i, s_i_tilde, beta)
            v_i = new_vi
        # Second Stage #
        g_t = nx.induced_subgraph(graph, v_i)
        k_tilde = dgp_get_k_tilde(g_t, alpha, beta, eta, t, sg_sz / np.sqrt(sz), sg_sz)
        # INCLUDING the third, extension stage #
        k_tag = dgp_get_k_tag(k_tilde, graph)
        k_star = dgp_get_k_star(k_tag, graph, sg_sz)
        remaining_subgraph_vertices.append(len([v for v in k_star if labels[v]]))
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


def run_dm(sz, p, sg_sz, subgraph, write=False, writer=None):
    """
    An implementation of the algorithm of Deshpande and Montanari for clique recovery
    using approximate message passing (i.e. belief propagation)
    """
    graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []

    start_time = time.time()
    for graph, labels in zip(graphs, all_labels):
        w = nx.to_numpy_array(graph)
        for i, j in permutations(range(w.shape[0]), 2):
            if i != j and w[i, j] == 0:
                w[i, j] = -1
            elif w[i, j] == 1:
                w[i, j] = (1 - p) / p  # DM is adjusted to match its requirement that E[W] = 0.
        kappa = sg_sz / np.sqrt(sz)
        gamma_vectors = [np.ones((sz,))]
        gamma_matrices = [np.subtract(np.ones((sz, sz)), np.eye(sz))]
        t_star = 100
        # Belief Propagation iterations. The code here uses some matrices to shorten the time comparing to for loops #
        for t in range(t_star):
            helping_matrix = np.exp(gamma_matrices[t]) / np.sqrt(sz)
            log_numerator = np.log(1 + np.multiply(1 + w, helping_matrix))
            log_denominator = np.log(1 + helping_matrix)
            helping_for_vec = log_numerator - log_denominator
            gamma_vec = np.log(kappa) + np.sum(helping_for_vec, axis=1) - np.diag(helping_for_vec)
            gamma_mat = np.tile(gamma_vec, (sz, 1)) - helping_for_vec.transpose()
            gamma_vectors.append(gamma_vec)
            gamma_matrices.append(gamma_mat)
        sorted_vertices = np.argsort(gamma_vectors[t_star])
        c_n_hat = sorted_vertices[-2 * sg_sz:]
        # Without the cleaning stage which is similar to ours.
        remaining_subgraph_vertices.append(len([v for v in c_n_hat if labels[v]]))
    total_time = time.time() - start_time
    if write:
        assert writer is not None
        writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4)]])
    return total_time, np.mean(remaining_subgraph_vertices)


def calculate_features(graphs, params, graph_params):
    adjacency_matrices, feature_matrices = [], []
    for graph in graphs:
        fc = FeatureCalculator(graph_params, graph, "", params['features'], dump=False, gpu=True, device=0)
        adjacency_matrices.append(fc.adjacency_matrix)
        feature_matrices.append(fc.feature_matrix)

    # Normalizing the features by z-score (i.e. standard scaler). Having all the graphs regardless whether they are
    # training, eval of test, we can scale based on all of them together. Scaling based on the training and eval only
    # shows similar performance.
    scaler = StandardScaler()
    all_matrix = np.vstack(feature_matrices)
    scaler.fit(all_matrix)
    for i in range(len(feature_matrices)):
        feature_matrices[i] = scaler.transform(feature_matrices[i].astype('float64'))
    return adjacency_matrices, feature_matrices


def split_into_folds(adj_matrices, feature_matrices, labels):
    runs = []
    all_indices = np.arange(len(labels))
    np.random.shuffle(all_indices)
    folds = np.array_split(all_indices, 5)
    # for it in range(2):
    for it in range(len(folds)):
        test_fold = folds[it]
        eval_fold = folds[(it + 1) % 5]
        train_indices = np.hstack([folds[(it + 2 + j) % 5] for j in range(3)])
        training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels, \
            test_features, test_adj, test_labels = map(lambda x: [x[1][j] for j in x[0]],
                                                       product([train_indices, eval_fold, test_fold],
                                                               [feature_matrices, adj_matrices, labels]))

        runs.append((training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels,
                     test_features, test_adj, test_labels))
    return runs


def weighted_mse(outputs, labels, weights_tensor):
    return torch.mean(weights_tensor * torch.pow(outputs - labels, torch.tensor([2], device=labels.device)))


def build_weighted_loss(unary, class_weights, labels):
    weights_list = []
    for i in range(labels.shape[0]):
        weights_list.append(class_weights[labels[i].data.item()])
    weights_tensor = torch.tensor(weights_list, dtype=torch.double, device=labels.device)
    if unary == "bce":
        return torch.nn.BCELoss(weight=weights_tensor).to(labels.device)
    else:
        return partial(weighted_mse, weights_tensor=weights_tensor)


def pairwise_loss(flat_x, flat_adj):
    return - torch.mean((1 - flat_adj) * torch.log(
        torch.where(1 - flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device), 1 - flat_x)) +
                        flat_adj * torch.log(
        torch.where(flat_x <= 1e-8, torch.tensor([1e-8], dtype=torch.double, device=flat_x.device), flat_x)))


def binomial_reg(y_hat, graph_params):
    return - torch.mean(y_hat * np.log(graph_params["subgraph_size"] / graph_params["vertices"]) +
                        (1 - y_hat) * np.log(1 - graph_params["subgraph_size"] / graph_params["vertices"]))


def train_pygon(training_features, training_adjs, training_labels, eval_features, eval_adjs, eval_labels,
                params, class_weights, activations, unary, coeffs, graph_params):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    pygon = PYGONModel(n_features=training_features[0].shape[1], hidden_layers=params["hidden_layers"],
                       dropout=params["dropout"],
                       activations=activations, p=graph_params["probability"], normalization=params["edge_normalization"])
    pygon.to(device)
    opt = params["optimizer"](pygon.parameters(), lr=params["lr"], weight_decay=params["regularization"])

    n_training_graphs = len(training_labels)
    graph_size = graph_params["vertices"]
    n_eval_graphs = len(eval_labels)

    counter = 0  # For early stopping
    min_loss = None
    for epoch in range(params["epochs"]):
        # -------------------------- TRAINING --------------------------
        training_graphs_order = np.arange(n_training_graphs)
        np.random.shuffle(training_graphs_order)
        for i, idx in enumerate(training_graphs_order):
            training_mat = torch.tensor(training_features[idx], device=device)
            training_adj, training_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                             [training_adjs, training_labels])
            pygon.train()
            opt.zero_grad()
            output_train = pygon(training_mat, training_adj)
            output_matrix_flat = (torch.mm(output_train, output_train.transpose(0, 1)) + 1 / 2).flatten()
            training_criterion = build_weighted_loss(unary, class_weights, training_lbs)
            loss_train = coeffs[0] * training_criterion(output_train.view(output_train.shape[0]), training_lbs) + \
                coeffs[1] * pairwise_loss(output_matrix_flat, training_adj.flatten()) + \
                coeffs[2] * binomial_reg(output_train, graph_params)
            loss_train.backward()
            opt.step()

        # -------------------------- EVALUATION --------------------------
        graphs_order = np.arange(n_eval_graphs)
        np.random.shuffle(graphs_order)
        outputs = torch.zeros(graph_size * n_eval_graphs, dtype=torch.double)
        output_xs = torch.zeros(graph_size ** 2 * n_eval_graphs, dtype=torch.double)
        adj_flattened = torch.tensor(np.hstack([eval_adjs[idx].flatten() for idx in graphs_order]))
        for i, idx in enumerate(graphs_order):
            eval_mat = torch.tensor(eval_features[idx], device=device)
            eval_adj, eval_lbs = map(lambda x: torch.tensor(data=x[idx], dtype=torch.double, device=device),
                                     [eval_adjs, eval_labels])
            pygon.eval()
            output_eval = pygon(eval_mat, eval_adj)
            output_matrix_flat = (torch.mm(output_eval, output_eval.transpose(0, 1)) + 1 / 2).flatten()
            output_xs[i * graph_size ** 2:(i + 1) * graph_size ** 2] = output_matrix_flat.cpu()
            outputs[i * graph_size:(i + 1) * graph_size] = output_eval.view(output_eval.shape[0]).cpu()
        all_eval_labels = torch.tensor(np.hstack([eval_labels[idx] for idx in graphs_order]), dtype=torch.double)
        eval_criterion = build_weighted_loss(unary, class_weights, all_eval_labels)
        loss_eval = (coeffs[0] * eval_criterion(outputs, all_eval_labels) +
                     coeffs[1] * pairwise_loss(output_xs, adj_flattened) +
                     coeffs[2] * binomial_reg(outputs, graph_params)).item()

        if min_loss is None:
            current_min_loss = loss_eval
        else:
            current_min_loss = min(min_loss, loss_eval)

        if epoch >= 10 and params["early_stop"]:  # Check for early stopping during training.
            if min_loss is None:
                min_loss = current_min_loss
                torch.save(pygon.state_dict(), "tmp_time.pt")  # Save the best state.
            elif loss_eval < min_loss:
                min_loss = current_min_loss
                torch.save(pygon.state_dict(), "tmp_time.pt")  # Save the best state.
                counter = 0
            else:
                counter += 1
                if counter >= 40:  # Patience for learning
                    break
    # After stopping early, our model is the one with the best eval loss.
    pygon.load_state_dict(torch.load("tmp_time.pt"))
    os.remove("tmp_time.pt")
    return pygon


def test_pygon(model, test_features, test_adjs, graph_params):
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    graph_size = graph_params['vertices']
    n_graphs = len(test_adjs)
    graphs_order = np.arange(n_graphs)
    np.random.shuffle(graphs_order)
    outputs = torch.zeros(graph_size * n_graphs, dtype=torch.double)
    for i, idx in enumerate(graphs_order):
        test_mat = torch.tensor(test_features[idx], device=device)
        test_adj = torch.tensor(data=test_adjs[idx], dtype=torch.double, device=device)
        model.eval()
        output_test = model(test_mat, test_adj)
        outputs[i * graph_size:(i + 1) * graph_size] = output_test.view(output_test.shape[0]).cpu()
    return outputs.tolist()


def run_pygon_time(size, p, subgraph_size, subgraph, params, other_params):
    """
    An implementation of PYGON without time-wasting actions such as saving the graph and features as pickle files.
    """
    graphs, all_labels = graphs_loader(size, p, subgraph_size, subgraph)

    graph_params = {'vertices': size, 'probability': p, 'subgraph_size': subgraph_size,
                    'directed': True if subgraph == "dag-clique" else False}
    if other_params is None:
        unary = "bce"
        coeffs = [1., 0., 0.]
    else:
        if "unary" in other_params:
            unary = other_params["unary"]
        else:
            unary = "bce"
        if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
            if other_params["c2"] == "k":
                c2 = 1. / subgraph_size
            elif other_params["c2"] == "sqk":
                c2 = 1. / np.sqrt(subgraph_size)
            else:
                c2 = other_params["c2"]
            coeffs = [other_params["c1"], c2, other_params["c3"]]
        else:
            coeffs = [1., 0., 0.]

    # Preprocessing - feature calculations
    start_time = time.time()
    adj_matrices, feature_matrices = calculate_features(graphs, params, graph_params)
    feature_calc_time = time.time() - start_time

    runs = split_into_folds(adj_matrices, feature_matrices, all_labels)
    class_weights = {0: (float(size) / (size - subgraph_size)), 1: (float(size) / subgraph_size)}
    activations = [params['activation']] * (len(params['hidden_layers']) + 1)

    remaining_subgraph_vertices = []
    total_training_time = 0
    total_test_time = 0
    for fold in runs:
        training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels, \
            test_features, test_adj, test_labels = fold
        # Training
        training_start_time = time.time()
        model = train_pygon(training_features, training_adj, training_labels, eval_features, eval_adj, eval_labels,
                            params, class_weights, activations, unary, coeffs, graph_params)
        one_fold_training_time = time.time() - training_start_time
        total_training_time += one_fold_training_time

        # Testing
        test_start_time = time.time()
        test_scores = test_pygon(model, test_features, test_adj, graph_params)
        for r in range(len(test_labels) // size):
            ranks_by_run = test_scores[r * size:(r + 1) * size]
            labels_by_run = test_labels[r * size:(r + 1) * size]
            sorted_vertices_by_run = np.argsort(ranks_by_run)
            c_n_hat_by_run = sorted_vertices_by_run[-2 * subgraph_size:]
            remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
        one_fold_test_time = time.time() - test_start_time
        total_test_time += one_fold_test_time

    return feature_calc_time, total_training_time, total_test_time, np.mean(remaining_subgraph_vertices)

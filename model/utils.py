import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
matplotlib.use('agg')


def update_params(params, sg_sz, other_params=None):
    """
    Update the params dict with the possible change of loss.
    :param params: The params dict - a dictionary of the hyper-parameters for PYGON.
    :param other_params: The dictionary including keys 'c1', 'c2', 'c3' and 'unary',
                         corresponding to the loss coefficients and type of loss
                         (c1*[WBCE if 'unary' is 'bce' else WMSE] + c2*Pairwise + c3*Binomial).
    :param sg_sz: int. Size of planted sub-graph.
    :return:
    """
    if other_params is None:
        upd_params = {"coeffs": [1., 0., 0.], "unary": "bce"}
    else:
        upd_params = {}
        if "unary" in other_params:
            upd_params["unary"] = other_params["unary"]
        if all(["c1" in other_params, "c2" in other_params, "c3" in other_params]):
            if other_params["c2"] == "k":
                c2 = 1. / sg_sz
            elif other_params["c2"] == "sqk":
                c2 = 1. / np.sqrt(sg_sz)
            else:
                c2 = other_params["c2"]
            upd_params["coeffs"] = [other_params["c1"], c2, other_params["c3"]]
    params.update(upd_params)


def check_make_dir(path_check, path_make=None):
    if path_make is None:
        path_make = path_check
    if not os.path.exists(path_check):
        os.makedirs(path_make)


def write_to_csv(writer, results, sz, sg_sz, header=None, per_graph=False):
    scores, lbs = results
    auc = []
    remaining_subgraph_vertices = []
    for r in range(len(lbs) // sz):
        ranks_by_run = scores[r*sz:(r+1)*sz]
        labels_by_run = lbs[r*sz:(r+1)*sz]
        auc.append(roc_auc_score(labels_by_run, ranks_by_run))
        sorted_vertices_by_run = np.argsort(ranks_by_run)
        c_n_hat_by_run = sorted_vertices_by_run[-2*sg_sz:]
        remaining_subgraph_vertices.append(len([v for v in c_n_hat_by_run if labels_by_run[v]]))
    if not per_graph:
        if header is None:
            writer.writerow([str(val) for val in [sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
        else:
            writer.writerow([str(val) for val in [header, sz, sg_sz, np.round(np.mean(remaining_subgraph_vertices), 4),
                                                  np.round(np.mean(auc), 4)]])
    else:
        for i in range(len(auc)):
            writer.writerow([str(val) for val in [sz, sg_sz, i,
                                                  np.round(remaining_subgraph_vertices[i], 4),
                                                  np.round(auc[i], 4)]])
        writer.writerow([str(val) for val in [sz, sg_sz, 'Mean',
                                              np.round(np.mean(remaining_subgraph_vertices), 4),
                                              np.round(np.mean(auc), 4)]])


def plot_losses(train_losses, eval_losses, test_losses, sz, sg_sz, subgraph):
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    fig.suptitle(t=f'Loss by iteration - size {sz}, subgraph size {sg_sz}', y=1, fontsize='x-large')
    for r in range(len(train_losses)):
        train_loss = train_losses[r]
        eval_loss = eval_losses[r]
        test_loss = test_losses[r]
        train_iterations = np.arange(1, len(train_loss) + 1)
        eval_iterations = np.arange(1, len(eval_loss) + 1)
        test_iterations = np.arange(1, 5*len(test_loss) + 1, 5)
        ax[0].plot(train_iterations, train_loss)
        ax[0].set_title(f'Training loss for each run, '
                        f'mean final loss: {np.mean([train_losses[i][-1] for i in range(len(train_losses))]):.4f}')
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel('loss')
        ax[1].plot(eval_iterations, eval_loss)
        ax[1].set_title(f'Eval loss for each run, '
                        f'mean final loss: {np.mean([eval_losses[i][-1] for i in range(len(eval_losses))]):.4f}')
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel('loss')
        ax[2].plot(test_iterations, test_loss)
        ax[2].set_title(f'Test loss for each run, '
                        f'mean loss: {np.mean([test_losses[i][-1] for i in range(len(test_losses))]):.4f}')
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel('loss')
    ax[0].legend([f'Loss = {train_losses[run][-1]}' for run in range(len(train_losses))])
    ax[1].legend([f'Loss = {eval_losses[run][-1]:.4f}' for run in range(len(eval_losses))])
    ax[2].legend([f'Loss = {test_losses[run][-1]:.4f}' for run in range(len(test_losses))])
    check_make_dir(os.path.join(os.path.dirname(__file__), 'figures', subgraph))
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', subgraph, f'Loss_by_iteration_{sz}_{sg_sz}.png'))


def dump_results(filename, sz, sg_sz, subgraph, train_res, eval_res, test_res):
    dir_path = os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph, filename)
    check_make_dir(os.path.join(dir_path, f"{sz}_{sg_sz}"))
    for (scores, lbs), what_set in zip([train_res, eval_res, test_res], ["train", "eval", "test"]):
        res_by_run_df = pd.DataFrame({"scores": scores, "labels": lbs})
        res_by_run_df.to_csv(os.path.join(dir_path, f"{sz}_{sg_sz}", f"{what_set}_results.csv"))

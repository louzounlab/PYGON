"""
Performance testing file of the first stage of subgraph recovering, i.e. PYGON.
We assume here that the graphs on which the tests run have already been created. If not, one may add the variable
new_runs=(number of new runs to build) in PYGONMain class, or use feature_calculation function in graphs_and_features.py.
"""
import csv
from torch import relu
from torch.optim import Adam
from sklearn.metrics import roc_curve
from utils import *
from pygon_main import PYGONMain


PARAMS = {  # Best parameters found. Tested on several sizes, subgraphs and edge probabilities.
    "features": ['Motif_3'],
    "hidden_layers": [225, 175, 400, 150],
    "epochs": 1000,
    "dropout": 0.4,
    "lr": 0.005,
    "regularization": 0.0005,
    "optimizer": Adam,
    "activation": relu,
    "early_stop": True,
    "edge_normalization": "correct"
}


def performance_test_pygon(filename, sizes, subgraph, other_params=None, dump=False, p=0.5):
    """
    Create a csv for each set (training, eval, test).
    Calculate mean recovered subgraph vertices in TOP 2*SUBGRAPH SIZE and mean AUC.
    """
    check_make_dir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    check_make_dir(os.path.join(os.path.dirname(__file__), 'model_outputs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            update_params(params, sg_sz, other_params)
            pygon = PYGONMain(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                _, _, _, _, _, _, _, _, _, _, _, _ = pygon.train_performance(input_params=params, check='CV')

            if dump:
                dump_results(filename, sz, sg_sz, subgraph, (train_scores, train_lbs), (eval_scores, eval_lbs),
                             (eval_scores, eval_lbs))

            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)


def performance_test_per_graph(filename, sizes, subgraph, other_params=None, p=0.5):
    check_make_dir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Graph Index', 'Mean recovered subgraph vertices',
                        s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            update_params(params, sg_sz, other_params)
            pygon = PYGONMain(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, \
                _, _, _, _, _, _, _, _, _, _, _, _ = pygon.train_performance(input_params=params, check='CV')
            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz, per_graph=True)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz, per_graph=True)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz, per_graph=True)


def loss_by_iteration(filename, sizes, other_params=None, subgraph='clique', p=0.5):
    """
    Plot the losses (three subplots for the total loss by iteration - training, eval and test),
    save a figure for each pair (size, subgraph_size) in sizes.
    """
    check_make_dir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    if p != 0.5:
        filename += f"_p_{p}"
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for sz, sg_sz in sizes:
            print(str(sz) + ",", sg_sz)
            params = PARAMS.copy()
            update_params(params, sg_sz, other_params)
            pygon = PYGONMain(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params["features"], subgraph=subgraph)
            test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_losses, \
                eval_losses, test_losses, _, _, _, _, _, _, _, _, _ = pygon.train_performance(input_params=params, check='CV')

            # ANALYZING - Losses and AUCs.
            plot_losses(train_losses, eval_losses, test_losses, sz, sg_sz, subgraph)

            write_to_csv(wr, (test_scores, test_lbs), sz, sg_sz)
            write_to_csv(gwr, (train_scores, train_lbs), sz, sg_sz)
            write_to_csv(hwr, (eval_scores, eval_lbs), sz, sg_sz)


def loss_and_roc(sizes, midname="", other_params=None, subgraph='clique', dump=False, p=0.5):
    """
    Plot losses by iterations as above, this time in four rows (one for total loss, one for unary loss,
    one for pairwise loss and one for binomial reguarization), as well as the final ROC curves, all in one figure.
    This is done again once for each pair (size, subgraph_size) in sizes.
    """
    for folder in ['model_outputs', 'figures']:
        check_make_dir(os.path.join(os.path.dirname(__file__), folder, subgraph))
    if p != 0.5:
        midname += f"_p_{p}"
    for sz, sg_sz in sizes:
        print(str(sz) + ",", sg_sz)
        params = PARAMS.copy()
        update_params(params, sg_sz, other_params)
        pygon = PYGONMain(sz, p, sg_sz, True if subgraph == "dag-clique" else False, features=params['features'], subgraph=subgraph)
        test_scores, test_lbs, eval_scores, eval_lbs, train_scores, train_lbs, train_all_losses, eval_all_losses, \
            test_all_losses, train_unary_losses, eval_unary_losses, test_unary_losses, \
            train_pairwise_losses, eval_pairwise_losses, test_pairwise_losses,\
            train_binom_regs, eval_binom_regs, test_binom_regs = pygon.train_performance(input_params=params, check='CV')

        if dump:
            dump_results(midname, sz, sg_sz, subgraph, (train_scores, train_lbs), (eval_scores, eval_lbs),
                         (eval_scores, eval_lbs))

        # ANALYZING - Losses and ROCs.
        fig, ax = plt.subplots(5, 3, figsize=(16, 9))
        fig.suptitle(t=f"size {sz}, clique size {sg_sz}, "
                       f"coeffs: {params['coeffs'][0]:.3f}, {params['coeffs'][1]:.3f}, {params['coeffs'][2]:.3f}",
                     y=1, fontsize='x-large')
        part = ["total loss", "unary loss", "pairwise loss", "binomial reg."]
        for i, (train_losses, eval_losses, test_losses) in enumerate(
                [(train_all_losses, eval_all_losses, test_all_losses), (train_unary_losses, eval_unary_losses, test_unary_losses),
                 (train_pairwise_losses, eval_pairwise_losses, test_pairwise_losses),
                 (train_binom_regs, eval_binom_regs, test_binom_regs)]):
            for r in range(len(train_losses)):
                train_loss, eval_loss, test_loss = train_losses[r], eval_losses[r], test_losses[r]
                train_iterations, eval_iterations, test_iterations = \
                    np.arange(1, len(train_loss) + 1), np.arange(1, len(eval_loss) + 1), np.arange(1, 5*len(test_loss) + 1, 5)
                ax[i, 0].plot(train_iterations, train_loss)
                ax[i, 1].plot(eval_iterations, eval_loss)
                ax[i, 2].plot(test_iterations, test_loss)
            for j, (set_name, losses) in enumerate(zip(['Training', 'Eval', 'Test'], [train_losses, eval_losses, test_losses])):
                ax[i, j].set_title(f"{set_name} {part[i]} for each run, "
                                   f"mean final loss: {np.mean([losses[i][-1] for i in range(len(losses))]):.4f}")
                ax[i, j].set_xlabel('iteration')
                ax[i, j].set_ylabel(part[i])
                ax[i, j].legend([f'Final = {losses[run][-1]:.4f}' for run in range(len(losses))])
        for k, (scores, lbs, set_name) in enumerate(
                zip([train_scores, eval_scores, test_scores], [train_lbs, eval_lbs, test_lbs],
                    ["Training", "Eval", "Test"])):
            auc = []
            for r in range(len(test_lbs) // sz):
                ranks_by_run = scores[r * sz:(r + 1) * sz]
                labels_by_run = lbs[r * sz:(r + 1) * sz]
                auc.append(roc_auc_score(labels_by_run, ranks_by_run))
                fpr, tpr, _ = roc_curve(labels_by_run, ranks_by_run)
                ax[4, k].plot(fpr, tpr)
            ax[4, k].set_title(f"{set_name} AUC for each run, Mean AUC: {np.mean(auc):.4f}")
            ax[4, k].set_xlabel('FPR')
            ax[4, k].set_ylabel('TPR')
        plt.tight_layout(h_pad=0.2)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', subgraph, f'loss_roc_{midname}_{sz}_{sg_sz}.png'))


def dumped_to_performance(dumping_path, subgraph, filename, p=0.5):
    """
    From the dumped output-labels CSVs do as "performance_test_pygon" does.
    """
    if p != 0.5:
        filename += f"_p_{p}"
    check_make_dir(os.path.join(os.path.dirname(__file__), 'csvs', subgraph))
    with open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_test.csv'), 'w') as f, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_train.csv'), 'w') as g, \
            open(os.path.join(os.path.dirname(__file__), 'csvs', subgraph, filename + '_eval.csv'), 'w') as h:
        wr, gwr, hwr = map(csv.writer, [f, g, h])
        for w, s in zip([wr, gwr, hwr], ['test', 'training', 'eval']):
            w.writerow(['Graph Size', 'Subgraph Size', 'Mean recovered subgraph vertices', s + ' AUC on all runs'])
        for res_dir in sorted(os.listdir(dumping_path)):
            sz, sg_sz = map(int, res_dir.split("_"))
            for w, res in zip([wr, gwr, hwr],
                              map(lambda x: pd.read_csv(os.path.join(dumping_path, res_dir, x + "_results.csv")),
                                  ["test", "eval", "train"])):
                scores = res.scores.values
                lbs = res.labels.values
                write_to_csv(w, (scores, lbs), sz, sg_sz)

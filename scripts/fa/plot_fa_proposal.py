from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.fa.fa_base import setup_argparse
from scripts.fa.fa_vinfs import get_normalizing_flows
from src.algorithms import get_algorithm
from src.problems import get_problem
from src.samplers import PowerPosteriorParticleDensity, SingleModelMPD
from src.utils.tools import kde_joint
from src.variables import ParametricModelSpace


class RJPropTest:
    def __init__(self, parametric_model, target_draws):
        """
        target_draws should be an enumerated model list perhaps. TODO determine this.
        This class uses equation (16) from Bartolucci et al (2006)
        """
        self.pmodel = parametric_model
        self.nmutations = 0

        assert isinstance(self.pmodel, ParametricModelSpace)

        self.init_more()
        self.target_draws = target_draws

        self.pmodel.setStartingDistribution(self.pmodel)

        # create the MMMPD
        mmmpd = SingleModelMPD(self.pmodel)

        # add prior draws to avoid an exception
        init_draws = self.pmodel.draw(target_draws.shape[0])
        init_llh = self.pmodel.compute_llh(init_draws)
        init_log_prior = self.pmodel.compute_prior(init_draws)
        init_pppd = PowerPosteriorParticleDensity(
            self.pmodel,
            None,
            init_llh,
            init_log_prior,
            init_draws,
            0.0,
            np.log(np.ones_like(init_llh) * 1.0 / init_draws.shape[0]),
        )
        mmmpd.addComponent(init_pppd)

        # add target draws
        llh = self.pmodel.compute_llh(target_draws)
        log_prior = self.pmodel.compute_prior(target_draws)
        pppd = PowerPosteriorParticleDensity(
            self.pmodel,
            None,
            llh,
            log_prior,
            target_draws,
            1.0,
            np.log(np.ones_like(llh) * 1.0 / target_draws.shape[0]),
        )
        mmmpd.addComponent(pppd)
        self.pmodel.calibrateProposalsMMMPD(mmmpd, target_draws.shape[0], 1)

    def init_more(self):
        pass

    def propose(self, target_draws, blocksize=None):
        theta = target_draws
        N = theta.shape[0]
        if blocksize is None:
            blocksize = N
        llh = np.zeros(N)  # log likelihood
        cur_prior = np.zeros(N)  # log prior
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros(N)
        prop_llh = np.full(N, np.NINF)
        prop_prior = np.zeros(N)
        prop_id = np.zeros(N)
        prop_lpqratio = np.zeros(N)

        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)

        # get indices for computation blocks
        nblocks = int(np.ceil((1.0 * N) / blocksize))
        blocks = [np.arange(i * blocksize, min(N, (i + 1) * blocksize)) for i in range(nblocks)]

        # TODO reuse this computation from constructor for mmmpd
        for bidx in blocks:
            llh[bidx] = self.pmodel.compute_llh(theta[bidx])
            cur_prior[bidx] = self.pmodel.compute_prior(theta[bidx])
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(theta[bidx], N)
            prop_prior[bidx] = self.pmodel.compute_prior(prop_theta[bidx])

        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)

        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(np.isfinite(prop_prior), np.isfinite(prop_lpqratio))
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta, prop_theta, prop_lpqratio, prop_llh, llh, cur_prior, prop_prior, 1
        )  # TODO implement
        return prop_theta


def generate_theta_mode_train_theta_test_theta(data_folder):
    mk_theta_train = {}
    mk_theta_test = {}
    mk_theta_mode = {}
    for k in range(2, 4):
        mk_theta = np.load(str(data_folder / f"gold_m{k}.npy"))
        train_idx = np.zeros(mk_theta.shape[0]).astype(bool)
        train_idx[::2] = True
        test_idx = ~train_idx
        mk_theta_train[k] = mk_theta[train_idx]
        mk_theta_test[k] = mk_theta[test_idx]
        mk_theta_mode[k] = mk_theta_test[k][0::100]

    train_theta = np.vstack([th for mk, th in mk_theta_train.items()])
    test_theta = np.vstack([th for mk, th in mk_theta_test.items()])

    mktt2 = mk_theta_test[2]
    mktt3 = mk_theta_test[3]

    return mktt2, mktt3, mk_theta_mode, train_theta, test_theta


def generate_prop_theta(algorithms, y_data, mk_theta_mode, train_theta):
    prop_theta = {}
    for algo in algorithms:
        problem = get_problem("FA")
        model = get_algorithm(
            algo,
            problem=problem,
            y_data=y_data,
            k_min=1,
            k_max=2,
            normalizing_flows=get_normalizing_flows,
            # save_flows_dir=str(Path("data") / "flows"),
        )

        proptest = RJPropTest(model, train_theta[::10])
        for k in range(2, 4):
            prop_theta.setdefault(algo, {}).update({k: proptest.propose(mk_theta_mode[k])})
    return prop_theta


def plot(figures_folder, prob, algorithms, mktt2, mktt3, mk_theta_mode, prop_theta, kdegriddim, exp):
    plt.close()
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(0.9 * 6.0, 0.9 * 7.0))
    ax = ax.flatten()

    kde_joint(ax[0], mktt2[:, [9, 10]], cmap="Blues", alpha=1, bw=0.05, n_grid_points=kdegriddim)
    for i, algo in enumerate(algorithms):
        ax[i + 1].set_xlim([-1, 1])
        ax[i + 1].set_ylim([-1, 1])

        kde_joint(ax[i + 1], mktt3[:, [9, 13]], cmap="Blues", alpha=1, bw=0.05, n_grid_points=kdegriddim)
        ax[i + 1].scatter(prop_theta[algo][2][:, 9], prop_theta[algo][2][:, 13], color="red", s=1, alpha=0.7)

    ax[0].scatter(mk_theta_mode[2][:, 9], mk_theta_mode[2][:, 10], color="red", s=0.3, alpha=0.5)
    ax[0].title.set_text("Source\n2-Factor Model")
    ax[0].set_xlabel(r"$\beta_{2,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[0].set_ylabel(r"$\beta_{2,5}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[1].title.set_text("Target\n3-Factor Model, L&W")
    ax[1].set_xlabel(r"$\beta_{2,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[1].set_ylabel(r"$\beta_{3,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[2].title.set_text("Target\n3-Factor Model, Affine")
    ax[2].set_xlabel(r"$\beta_{2,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[2].set_ylabel(r"$\beta_{3,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[3].title.set_text("Target\n3-Factor Model, RQMA")
    ax[3].set_xlabel(r"$\beta_{2,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[3].set_ylabel(r"$\beta_{3,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[4].title.set_text("Target\n3-Factor Model, VINF")
    ax[4].set_xlabel(r"$\beta_{2,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[4].set_ylabel(r"$\beta_{3,4}$")  # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    for i in range(len(algorithms)):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

    plt.tight_layout()
    fig.savefig(str(figures_folder / f"{prob}_proposal_ngrid{kdegriddim}_Exp{exp}.pdf"))
    # plt.show()


if __name__ == "__main__":
    data_folder = Path("data") / "core"
    figures_folder = Path("docs") / "figures"
    figures_folder.mkdir(parents=True, exist_ok=True)

    Y = np.load(str(data_folder / "FA_data.npy"))

    PROBLEMS = ["FA"]
    ALGORITHMS = [f"FactorAnalysisModel{algo}" for algo in ["LW", "AF", "NF", "VINF"]]

    mktt2, mktt3, mk_theta_mode, train_theta, _ = generate_theta_mode_train_theta_test_theta(data_folder)

    args = setup_argparse()

    for exp in range(args.start, args.end + 1):
        prop_theta = generate_prop_theta(ALGORITHMS, Y, mk_theta_mode, train_theta)

        for prob in PROBLEMS:
            for kdegriddim in [128, 256]:
                plot(figures_folder, prob, ALGORITHMS, mktt2, mktt3, mk_theta_mode, prop_theta, kdegriddim, exp)

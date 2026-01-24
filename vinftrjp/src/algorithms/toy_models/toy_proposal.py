import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from src.proposals import Proposal


class RJToyModelProposal(Proposal):
    def __init__(
        self,
        problem,
        model_id_name="k",
        t1_name="t1",
        t2_name="t2",
        within_model_proposal=None,
        **kwargs,
    ):
        self.problem = problem
        self.k_name = model_id_name
        self.t1_name = t1_name
        self.t2_name = t2_name

        self.verbose = kwargs.get("verbose", False)
        self.run_index = kwargs.get("run_index", None)

        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.rv_names = [self.k_name, self.t1_name, self.t2_name]

        super().__init__([*self.rv_names, self.within_model_proposal])

        self.exclude_concat = [self.k_name]

    def set_run_index(self, run_index):
        self.run_index = run_index

    @staticmethod
    def make_mk2cov_t(t, var_h, mk2cov):
        var_h_inv = 1.0 / var_h
        mk2L, mk2Q = torch.linalg.eigh(mk2cov)
        return torch.mm(mk2Q, torch.mm(torch.diag(1.0 / (var_h_inv + t * ((1.0 / mk2L) - var_h_inv))), mk2Q.T))

    def calibratemmmpd(self, mmmpd, size, t):
        """
        This version computes the exact model probabilities for the 1D or 2D gaussian mixture example.
        """
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        mks = self.pmodel.getModelKeys()  # get all keys

        # set up transforms
        self.flows = {}
        self.mk_logZhat = {}

        var_h = 25.0
        var_h_inv = 1.0 / var_h
        mk1sig = 1
        log_mk1sig = 0
        log_mk1cov_t = -np.log((1.0 / mk1sig) * t + var_h_inv * (1 - t))

        mk2cov = torch.tensor([[1.0, 0.99], [0.99, 1.0]])
        mk2cov_t = self.make_mk2cov_t(t, var_h, mk2cov)
        if self.verbose:
            print("mk2cov_t", mk2cov_t)
            print("test t=1 mk2cov", self.make_mk2cov_t(0, var_h, mk2cov))

        # compute normalising constants.
        log_mk1sig_t = 0.5 * log_mk1cov_t
        _, log_mk2detcov_t = torch.linalg.slogdet(mk2cov_t)
        log_mk1sig_h_1t = np.log(var_h) * (1 - t) * 0.5
        log_mk2detcov_h_1t = (1 - t) * 2 * np.log(var_h)
        _, log_mk2detcov = torch.linalg.slogdet(mk2cov)

        # inverse normalising constant I think
        log_mk1_inc = np.log(self.problem.m1prob) * t + log_mk1sig_t - (log_mk1sig * t + log_mk1sig_h_1t)
        log_mk2_inc = np.log(1 - self.problem.m1prob) * t + 0.5 * (
            log_mk2detcov_t - (log_mk2detcov * t + log_mk2detcov_h_1t)
        )

        self.mk_logZhat[(0,)] = log_mk1_inc - torch.log(np.exp(log_mk1_inc) + torch.exp(log_mk2_inc))
        self.mk_logZhat[(1,)] = log_mk2_inc - torch.log(np.exp(log_mk1_inc) + torch.exp(log_mk2_inc))

        if self.run_index is not None:
            self.mk_logZhat[(0,)] = -np.inf
            self.mk_logZhat[(1,)] = 1.0

        if self.verbose:
            print("Z = ", {mk: np.exp(lz) for mk, lz in self.mk_logZhat.items()})

        self._calibratemmmpd(mks, mmmpd, t)

    def _calibratemmmpd(self, mks, mmmpd, t) -> None:
        raise NotImplementedError

    def extractModelConcatCols(self, X, mk):
        if self.getModelInt(mk) == 0:
            return X[:, 0].reshape((X.shape[0], 1))
        elif self.getModelInt(mk) == 1:
            return X[:, :2]

    def returnModelConcatCols(self, XX, mk):
        if self.getModelInt(mk) == 0:
            return np.column_stack([XX, np.zeros_like(XX)])
        elif self.getModelInt(mk) == 1:
            return XX

    def transformToBase(self, inputs, mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs, mk), mk)
        XX, logdet = self.flows[mk]._transform.forward(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(
            self.returnModelConcatCols(XX.detach().numpy(), mk), inputs, mk
        ), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs, mk), mk)
        XX, logdet = self.flows[mk]._transform.inverse(torch.tensor(X, dtype=torch.float32))
        return self.deconcatParameters(
            self.returnModelConcatCols(XX.detach().numpy(), mk), inputs, mk
        ), logdet.detach().numpy()

    def getModelInt(self, mk):
        if isinstance(mk, list):
            return np.array([m[0] for m in mk])
        else:
            return mk[0]

    def draw(self, theta, size=1):
        logpqratio = np.zeros(theta.shape[0])
        prop_theta = theta.copy()
        prop_ids = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}

        sigma_u = 1  # std dev of auxiliary u dist

        model_enumeration, _ = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)

            lpq_mk[mk] = lpq1
            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)
            t1 = proposed[mk][self.t1_name]  # .flatten()
            t2 = proposed[mk][self.t2_name]  # .flatten()

            mk_n = t1.shape[0]

            mprobkeys = list(self.mk_logZhat.keys())

            totallogZ = logsumexp(np.array([z for mkz, z in self.mk_logZhat.items()]))
            mprobs = np.exp(np.array([self.mk_logZhat[mkz] - totallogZ for mkz in mprobkeys]))
            mprobsdict = {mk: mprobs[i] for i, mk in enumerate(mprobkeys)}

            mpropidx = np.random.choice(len(mprobkeys), size=mk_n, p=mprobs)
            mprop = [mprobkeys[i] for i in mpropidx]

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            if self.getModelInt(mk) == 0:
                mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta = (
                    self._draw_0(
                        mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta
                    )
                )

            elif self.getModelInt(mk) == 1:
                mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta = (
                    self._draw_1(
                        mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta
                    )
                )

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids

        return prop_theta, logpqratio, prop_ids

    def _draw_0(self, mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta):
        jump_idx = self.getModelInt(mprop) == 1
        within_idx = self.getModelInt(mprop) == 0

        # within model move
        prop_mk_theta[within_idx], lpq_mk[mk][within_idx], mk_prop_ids[within_idx] = self.within_model_proposal.draw(
            mk_theta[within_idx], within_idx.sum()
        )

        # switch
        if jump_idx.sum() > 0:
            # draw the aux var.
            if self.run_index is None:
                u = norm(0, sigma_u).rvs(jump_idx.sum())
            else:
                u_unif = np.linspace(1e-5, 1 - 1e-5, jump_idx.sum())[self.run_index]
                u = norm(0, sigma_u).ppf(u_unif)

            log_u_eval = norm(0, sigma_u).logpdf(u.flatten())

            lpq_mk[mk][jump_idx] += -log_u_eval + np.log(mprobsdict[(0,)]) - np.log(mprobsdict[(1,)])

            # for naive, update with random walk using old u
            Tmktheta[jump_idx] = self.setVariable(Tmktheta[jump_idx], self.t2_name, u)

            # for naive, update with random walk using old u
            Tmktheta[jump_idx] = self.setVariable(Tmktheta[jump_idx], self.k_name, 1)

            prop_mk_theta[jump_idx], lpq2 = self.transformFromBase(Tmktheta[jump_idx], (1,))
            lpq_mk[mk][jump_idx] += lpq2
            mk_prop_ids[jump_idx] = np.full(jump_idx.sum(), id(self))

        return mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta

    def _draw_1(self, mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta):
        jump_idx = self.getModelInt(mprop) == 0
        within_idx = self.getModelInt(mprop) == 1

        # within model move
        prop_mk_theta[within_idx], lpq_mk[mk][within_idx], mk_prop_ids[within_idx] = self.within_model_proposal.draw(
            mk_theta[within_idx], within_idx.sum()
        )

        # switch
        if jump_idx.sum() > 0:
            u = t2[jump_idx]

            log_u_eval = norm(0, sigma_u).logpdf(u.flatten())

            lpq_mk[mk][jump_idx] += log_u_eval - np.log(mprobsdict[(0,)]) + np.log(mprobsdict[(1,)])

            Tmktheta[jump_idx] = self.setVariable(Tmktheta[jump_idx], self.k_name, 0)
            Tmktheta[jump_idx] = self.setVariable(Tmktheta[jump_idx], self.t2_name, 0)

            # transform back
            prop_mk_theta[jump_idx], lpq2 = self.transformFromBase(Tmktheta[jump_idx], (0,))
            lpq_mk[mk][jump_idx] += lpq2
            mk_prop_ids[jump_idx] = np.full(jump_idx.sum(), id(self))

        return mprop, mk_theta, mk, prop_mk_theta, lpq_mk, mk_prop_ids, t2, sigma_u, mprobsdict, Tmktheta

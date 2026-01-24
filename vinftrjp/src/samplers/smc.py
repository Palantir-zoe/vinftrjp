import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import uniform

from src.proposals import Proposal
from src.utils.tools import progress
from src.variables import ParametricModelSpace


def VERBOSITY_HIGH():
    # return True # set when debugging
    return False


TESTBIAS = False


class SMCQuantities:
    def __init__(self, m_indices, llh, log_prior, theta, pmodel, mk):
        self.pmodel = pmodel  # reference
        self.mk = mk
        self.indices = m_indices
        self.N = m_indices.shape[0]
        self.theta = theta.copy()
        self.log_w = np.full(self.N, -np.log(self.N))
        self.log_w_norm = self.log_w.copy()  # log(1/N)
        self.log_lh = llh.copy()  # np.zeros(self.N)
        self.log_prior = log_prior.copy()  # np.zeros(self.N)
        self.log_Zt = 0  # running total estimate of marginal likelihood
        self.log_ESS = np.log(self.N)  # running total estimate of marginal likelihood
        self.log_CESS = np.log(self.N)  # running total estimate of marginal likelihood

    def updateAfterMutate(self, m_indices, llh, log_prior, theta):
        self.indices = m_indices
        self.N = m_indices.shape[0]
        self.theta = theta.copy()
        self.log_lh = llh.copy()
        self.log_prior = log_prior.copy()
        self.log_w = np.full(self.N, -np.log(self.N))
        self.log_w_norm = self.log_w.copy()  # log(1/N)
        self.log_ESS = np.log(self.N)  # running total estimate of marginal likelihood
        self.log_CESS = np.log(self.N)  # running total estimate of marginal likelihood

    def updateAcceptanceRate(self, a):
        self.ar = a

    def getAcceptanceRate(self):
        """
        for now get a summary acceptance rate
        """
        return self.ar

    def makePPPD(self, gamma_t):
        return PowerPosteriorParticleDensity(
            self.pmodel,
            self.mk,
            self.log_lh,
            self.log_prior,
            self.theta,
            gamma_t,
            self.log_w,
        )

    def __str__(self):
        return "log_Zt: {}\nindices: {}\nN: {}\nlog_w_norm: {}\nlog_w: {}\nlog_ESS: {}".format(
            self.log_Zt, self.indices, self.N, self.log_w_norm, self.log_w, self.log_ESS
        )

    def __repr__(self):
        return self.__str__()
        # return self.__dict__


class MixtureParticleDensity:
    def __init__(self, mk_id=None):
        self.mixture_components = {}
        self.target_weights = {}
        self.target_log_unnorm_weights = {}
        self.mk_id = mk_id

    def getComponentKeys(self):
        return self.mixture_components.keys()

    def log_Zt(self, gamma_t=1):
        """
        Returns estimate of log_Zt at returned last_gamma_t
        input arg gamma_t is default 1, used as cutoff for density used in estimation.
        """
        last_gamma_t = 0
        n_components = len(self.mixture_components)
        log_Zt_Zt1 = np.zeros(n_components)
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if c.gamma_t > gamma_t:
                continue
            log_Zt_Zt1[i] = c.log_Zt_Zt1(last_gamma_t)
            last_gamma_t = c.gamma_t
        log_Zt = np.cumsum(log_Zt_Zt1)
        return log_Zt, last_gamma_t

    def log_Zt_k(self, mk, gamma_t=1):
        """
        TODO Deprecate,
        Returns estimate of log_Zt at returned last_gamma_t
        input arg gamma_t is default 1, used as cutoff for density used in estimation.
        """
        last_gamma_t = 0
        n_components = len(self.mixture_components)
        log_Zt_Zt1_k = np.zeros(n_components)  # don't need +1 because we pushed the gamma_0 state
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if c.gamma_t > gamma_t:
                break
            log_Zt_Zt1_k[i] = c.log_Zt_Zt1_k(mk, last_gamma_t)
            last_gamma_t = c.gamma_t
        log_Zt_k = np.cumsum(log_Zt_Zt1_k)
        return log_Zt_k, last_gamma_t

    def nComponentsTo(self, gamma_t):
        n = 0
        for gt in self.mixture_components.keys():
            if gt < gamma_t:
                n += 1
        return n

    def addComponent(self, pppd):
        # TODO either move the pmodel ref from pppd to this class or assert it matches all others in the list
        self.mixture_components[pppd.gamma_t] = pppd  # replaces any pppd at this temperature
        self._recomputeWeights(pppd.gamma_t)

    def deleteComponentAt(self, gamma_t):
        assert gamma_t in self.mixture_components.keys()
        # TODO either move the pmodel ref from pppd to this class or assert it matches all others in the list
        del self.mixture_components[gamma_t]  # replaces any pppd at this temperature
        # recompute at last weight
        if len(self.mixture_components.keys()) > 0:
            gamma_list = sorted(self.mixture_components.keys())
            self._recomputeWeights(gamma_list[-1])

    def _recomputeWeights(self, gamma_t):
        assert gamma_t in self.mixture_components.keys()
        # print(self.mixture_components.keys())
        assert 0 in self.mixture_components.keys()
        # get gamma index
        gamma_list = sorted(self.mixture_components.keys())
        # print("gamma list",gamma_list)
        nth_component = gamma_list.index(gamma_t)
        # print("nth component",nth_component)
        n_components = nth_component + 1
        # re-compute weights
        log_w = []  # we use a list instead of a 2D array because each target will have a different number of particles
        w = []
        log_w = []
        log_Zt, last_gamma_t = self.log_Zt()
        if VERBOSITY_HIGH():
            print("_recomputeWeights log_Z[{}]={}".format(gamma_t, log_Zt))
        extradensity = {}
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            extradensity[i] = np.zeros((c.size(), n_components))
            for j in range(n_components):
                k2 = gamma_list[j]
                c2 = self.mixture_components[k2]
                extradensity[i][:, j] = (
                    c2.gamma_t * (c.llh + c.log_prior)
                    + (1 - c2.gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                    - log_Zt[j]
                )
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            log_w_t = (
                gamma_t * (c.llh + c.log_prior)
                + (1 - gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                - (-np.log(n_components) + logsumexp(extradensity[i], axis=1))
            )

            log_w.append(log_w_t)
        denom = logsumexp(np.concatenate(log_w))
        for i, lw in enumerate(log_w):
            w.append(np.exp(lw - denom))
        self.target_weights[gamma_t] = w
        self.target_log_unnorm_weights[gamma_t] = log_w

    def getOriginalParticleDensityForTemperature(self, gamma_t, normalise_weights=True):
        """returns weighted particle density"""
        c = self.mixture_components[gamma_t]
        if normalise_weights:
            return c.theta, c.original_log_w - logsumexp(c.original_log_w)
        else:
            return c.theta, c.original_log_w

    def getParticleDensityForTemperature(self, gamma_t, normalise_weights=True, return_logpdfs=False):
        """returns weighted particle density"""
        self._recomputeWeights(gamma_t)
        size = 0
        firstkey = list(self.mixture_components.keys())[0]
        dim = self.mixture_components[firstkey].dim()
        if normalise_weights:
            target_weights = self.target_weights
        else:
            target_weights = self.target_log_unnorm_weights
        for i, c in enumerate(target_weights[gamma_t]):
            size += c.shape[0]
        target = np.zeros((size, dim))
        target_w = np.zeros(size)
        if return_logpdfs:
            target_llh = np.zeros(size)
            target_log_prior = np.zeros(size)
        offset = 0
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if i >= len(target_weights[gamma_t]):
                break
            target[offset : offset + c.size(), :] = c.theta
            target_w[offset : offset + c.size()] = target_weights[gamma_t][i]
            if return_logpdfs:
                target_llh[offset : offset + c.size()] = c.llh
                target_log_prior[offset : offset + c.size()] = c.log_prior
            offset += c.size()
        if return_logpdfs:
            return target, target_w, target_llh, target_log_prior
        else:
            return target, target_w


class MultiModelMPD:
    def __init__(self):
        self.densities = {}

    def getTemperatures(self):
        dk = list(self.densities.keys())
        return list(self.densities[dk[0]].getComponentKeys())

    def addComponent(self, k, pppd):
        # TODO assert(k is a model key)
        if k not in self.densities:
            self.densities[k] = MixtureParticleDensity(mk_id=k)
        self.densities[k].addComponent(pppd)

    def getModelKeys(self):
        """
        Just return keys of densities
        """
        return self.densities.keys()

    def resample(self, weights, n):
        indices = np.zeros(n, dtype=np.int32)
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
            indices[i] = j
        return indices

    def getlogZForModelAndTemperature_old(self, k, gamma_t):
        target, targetw = self.densities[k].getParticleDensityForTemperature(gamma_t, normalise_weights=False)
        return logsumexp(targetw) - np.log(targetw.shape[0])

    def getlogZForModelAndTemperature(self, k, gamma_t):
        logZt, last_gamma_t = self.densities[k].log_Zt(gamma_t)
        return logZt[self.densities[k].nComponentsTo(gamma_t)]

    def getParticleDensityForModelAndTemperature(self, k, gamma_t, resample=False, resample_max_size=2000):
        """
        TODO make more Pythonic as an iterator
        """
        if k not in self.densities:
            raise "Unknown model key {}".format(k)
        target_list = []
        target_w_list = []

        target, targetw = self.densities[k].getParticleDensityForTemperature(gamma_t, normalise_weights=False)
        targetw_norm = np.exp(targetw - logsumexp(targetw))
        if resample:
            # print("weights norm sum",targetw_norm.sum())
            resample_size = min(resample_max_size, targetw_norm.shape[0] * 2)
            target_list.append(target[self.resample(targetw_norm, resample_size)])
        else:
            target_list.append(target)
        target_w_list.append(targetw_norm)

        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getOriginalParticleDensityForTemperature(self, gamma_t, resample=False, resample_max_size=2000):
        target_list = []
        target_w_list = []
        n = 0
        for k, density in self.densities.items():
            target, targetw = density.getOriginalParticleDensityForTemperature(gamma_t, normalise_weights=True)
            target_list.append(target)
            target_w_list.append(targetw)
            n += target.shape[0]
        if resample:
            for i, (t, tw) in enumerate(zip(target_list, target_w_list)):
                targetw_norm = np.exp(tw - logsumexp(tw))
                resample_size = int(t.shape[0] * 1.0 / n * resample_max_size)
                target_list[i] = t[self.resample(targetw_norm, resample_size)]
                target_w_list[i] = tw[self.resample(targetw_norm, resample_size)]
        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if VERBOSITY_HIGH():
            print("whole orig pd size", n)
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getParticleDensityForTemperature_broken(self, gamma_t, resample=False, resample_max_size=2000):
        """
        TODO this implementation is broken.
        """
        target_list = []
        target_w_list = []
        for k, density in self.densities.items():
            target, targetw = density.getParticleDensityForTemperature(gamma_t)  # normalises weights
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                target_list.append(target[self.resample(targetw, resample_size)])
            else:
                target_list.append(target)
            target_w_list.append(targetw)
        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack


# Wrapper obect for a single model MPD that behaves as a MultiModelMPD
class SingleModelMPD(MultiModelMPD):
    def __init__(self, pmodel):
        self.pmodel = pmodel
        self.density = MixtureParticleDensity()

    def getTemperatures(self):
        return list(self.density.getComponentKeys())

    def addComponent(self, pppd):
        # TODO assert(k is a model key)
        self.density.addComponent(pppd)

    def deleteComponentAt(self, gamma_t):
        self.density.deleteComponentAt(gamma_t)

    def getModelKeys(self):
        """
        Just return keys of densities
        """
        return self.pmodel.getModelKeys()

    def getlogZForModelAndTemperature(self, k, gamma_t):
        target, targetw = self.density.getParticleDensityForTemperature(gamma_t, normalise_weights=False)
        # get model k from the target
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(target)
        if k not in model_key_dict.keys():
            raise "Model {} not found in recycled target density.".format(k)
        mk_targetw = targetw[model_key_dict[k]]
        mk_logZOPR = logsumexp(mk_targetw) - np.log(mk_targetw.shape[0])
        total_logZOPR = logsumexp(targetw) - np.log(targetw.shape[0])
        total_logZOPR_list = []
        for mk, idx in model_key_dict.items():
            total_logZOPR_list.append(logsumexp(targetw[idx]) - np.log(targetw[idx].shape[0]))
        total_logZOPR_2 = logsumexp(np.array(total_logZOPR_list))
        prob_mk_lZOPR = np.exp(mk_logZOPR - total_logZOPR_2)
        return mk_logZOPR

    def getSMCLogZForModelAndTemperature(self, k, gamma_t):
        """
        TODO deprecate.
        """
        logZtk, last_gamma_t = self.density.log_Zt_k(k, gamma_t)
        return logZtk[self.density.nComponentsTo(gamma_t)]

    def getSMCLogZForTemperature(self, gamma_t):
        if VERBOSITY_HIGH():
            print("getSMCLogZTemperature gamma_t", gamma_t)
        logZt, last_gamma_t = self.density.log_Zt(gamma_t)
        if VERBOSITY_HIGH():
            print("logZt is ", logZt)
        return logZt[self.density.nComponentsTo(gamma_t)]

    def getParticleDensityForModelAndTemperature(self, k, gamma_t, resample=False, resample_max_size=2000):
        """
        TODO make more Pythonic as an iterator
        """
        target_list = []
        target_w_list = []

        target, targetw = self.density.getParticleDensityForTemperature(gamma_t, normalise_weights=False)
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(target)
        if k not in model_key_dict.keys():
            raise BaseException("Model {} not found in recycled target density.".format(k))
        mk_target = target[model_key_dict[k]]
        mk_targetw = targetw[model_key_dict[k]]
        targetw_norm = np.exp(mk_targetw - logsumexp(mk_targetw))
        if resample:
            resample_size = min(resample_max_size, targetw_norm.shape[0] * 2)
            resample_idx = self.resample(targetw_norm, resample_size)
            target_list.append(mk_target[resample_idx])
            target_w_list.append(targetw_norm[resample_idx])
        else:
            target_list.append(mk_target)
            target_w_list.append(targetw_norm)

        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getParticleDensityForTemperature(self, gamma_t, resample=False, resample_max_size=2000, return_logpdfs=False):
        if return_logpdfs:
            (
                target,
                targetw,
                targetllh,
                targetlogprior,
            ) = self.density.getParticleDensityForTemperature(gamma_t, normalise_weights=False, return_logpdfs=True)
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                targetw_norm = np.exp(targetw - logsumexp(targetw))
                resample_indices = self.resample(targetw_norm, resample_size)
                return (
                    target[resample_indices],
                    np.full(resample_size, 1.0 / resample_size),
                    targetllh[resample_indices],
                    targetlogprior[resample_indices],
                )
            else:
                return target, targetw, targetllh, targetlogprior
        else:
            target, targetw = self.density.getParticleDensityForTemperature(
                gamma_t, normalise_weights=False, return_logpdfs=False
            )
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                targetw_norm = np.exp(targetw - logsumexp(targetw))
                resample_indices = self.resample(targetw_norm, resample_size)
                return target[resample_indices], np.full(resample_size, 1.0 / resample_size)
            else:
                return target, targetw

    def getOriginalParticleDensityForTemperature(self, gamma_t, resample=False, resample_max_size=2000):
        target_list = []
        target_w_list = []
        target, targetw = self.density.getOriginalParticleDensityForTemperature(gamma_t, normalise_weights=True)
        if resample:
            resample_size = min(resample_max_size, targetw.shape[0] * 2)
            targetw_norm = np.exp(targetw - logsumexp(targetw))
            resample_indices = self.resample(targetw_norm, resample_size)
            return target[resample_indices], np.full(resample_size, 1.0 / resample_size)
        else:
            return target, targetw


class PowerPosteriorParticleDensity:
    def __init__(self, pmodel, mk, llh, log_prior, theta, gamma_t, original_log_w):
        if mk is not None:
            mk_list = pmodel.getModelKeys(theta)
            assert mk in mk_list
            assert len(mk_list) == 1
            # Do we assign mk to self?
        self.theta = theta.copy()  # theta.copy()
        self.llh = llh.copy()  # llh.copy()
        self.log_prior = log_prior.copy()  # log_prior.copy()
        self.gamma_t = gamma_t
        self.pmodel = pmodel
        self.original_log_w = original_log_w.copy()

    def size(self):
        return self.llh.shape[0]

    def dim(self):
        """need to ensure theta dimensions == 2. Could come unstuck in particle impoverishment scenarios."""
        return self.theta.shape[1]

    def log_Zt_Zt1(self, gamma_t_1):
        delta_gamma = self.gamma_t - gamma_t_1
        lZtZt1 = logsumexp(
            delta_gamma * (self.llh + self.log_prior) - delta_gamma * self.pmodel.evalStartingDistribution(self.theta)
        ) - np.log(self.size())
        return lZtZt1

    def log_Zt_Zt1_k(self, k, gamma_t_1):
        """
        TODO deprecate
        """
        log_Zt_Zt1_k = 0
        mkdict, rev = self.pmodel.enumerateModels(self.theta)
        delta_gamma = self.gamma_t - gamma_t_1
        if k in mkdict.keys():
            weights = delta_gamma * (
                self.llh[mkdict[k]] + self.log_prior[mkdict[k]]
            ) - delta_gamma * self.pmodel.evalStartingDistribution(self.theta[mkdict[k]])
            log_Zt_Zt1_k = logsumexp(weights) - np.log(mkdict[k].shape[0])
        return log_Zt_Zt1_k


class SMC1:
    def __init__(
        self,
        parametric_model,
        starting_distribution="prior",
        temperature_sequence=None,
        n_mutations=None,
        store_ar=False,
    ):
        """
        ESS-Adaptive Static Sequential Monte Carlo Sampler

        T                     : number of intermediate distributions
        parametric_model      : a ParametricModelSpace object defining the space of parameters and models
        starting_distribution : string to state whether sampler starts from prior,
                                or a RandomVariableBlock object that matches the parametric_model
        """
        self.pmodel = parametric_model
        self.tempseq = temperature_sequence
        self.store_ar = store_ar
        self.tempseq_done = [
            0.0,
        ]
        self.essseq_done = {0: 0}
        self.mkseq_done = {}
        if self.tempseq is not None:
            assert self.tempseq[0] == 0
            assert self.tempseq[-1] == 1
            assert np.sum(np.array(self.tempseq) > 1) == 0
            assert np.sum(np.array(self.tempseq) < 0) == 0
            assert np.sum((np.array(self.tempseq)[1:] - np.array(self.tempseq)[:1]) < 0) == 0
        self.nmutations = 0
        if starting_distribution == "prior":
            self.starting_dist = self.pmodel
        else:
            assert self.pmodel.islike(starting_distribution)
            self.starting_dist = starting_distribution
        if n_mutations is not None:
            assert isinstance(n_mutations, int)
        self.n_mutations = n_mutations
        assert isinstance(self.pmodel, ParametricModelSpace)
        self.pmodel.setStartingDistribution(self.starting_dist)
        # init future things
        self.previous_model_targets = SingleModelMPD(self.pmodel)
        self.init_more()

    def init_more(self):
        pass

    def computeIncrementalWeights(
        self,
        new_t,
        last_t,
        llh,
        log_prior,
        pmodel,
        theta,
        log_w_norm_last_t,
        store_D=False,
    ):
        log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
            (1 - new_t) - (1 - last_t)
        ) * pmodel.evalStartingDistribution(theta)
        return log_target_ratio

    def preImportanceSampleHook(self, last_t, llh, log_prior, smcq, pmodel, next_t=None):
        pass

    def ESSThreshold(self, new_t, last_t, ess_threshold, llh, log_prior, smcq, pmodel, N):
        # importance sample
        log_target_ratio = self.computeIncrementalWeights(
            new_t,
            last_t,
            llh,
            log_prior,
            pmodel,
            smcq.theta,
            smcq.log_w_norm,
            store_D=False,
        )
        log_w = smcq.log_w_norm + log_target_ratio
        log_w_sum = logsumexp(log_w)
        log_w_norm = log_w - log_w_sum
        log_ESS = -logsumexp(2 * log_w_norm)
        if VERBOSITY_HIGH():
            print("in situ ess", np.exp(log_ESS))
        return np.exp(log_ESS) - ess_threshold(new_t, N)

    def makeSMCQuantities(self, N, llh, log_prior, theta, pmodel, mk):
        return SMCQuantities(np.arange(N), llh, log_prior, theta, pmodel, mk)

    def getEmpiricalModelProbs(self, theta):
        kblocks, rev = self.pmodel.enumerateModels(theta)
        Ninv = 1.0 / float(theta.shape[0])
        mkprobs = {mk: len(idx) * Ninv for mk, idx in kblocks.items()}
        # account for zero prob models
        mklist = self.pmodel.getModelKeys()
        for mk in mklist:
            if mk not in mkprobs.keys():
                mkprobs[mk] = 0
        return mkprobs

    def run(self, N=100, ess_threshold=0.5):
        """
        Sample from prior
        For t=0,...,1
            Importance Sample
            Resample
            Mutate

        Particles (theta) are represented as 2D matrix (N-particles,Theta-dim)
        Uses adaptive resampling at ESS < threshold * N
        """
        if isinstance(ess_threshold, float):
            assert ess_threshold < 1 and ess_threshold > 0

            def ess_th_fn(t, N):
                return ess_threshold * N

        else:
            assert callable(ess_threshold)
            ess_th_fn = ess_threshold

        def dummyzero(a, b):
            return 0

        self.rbar_list = []
        # init memory for particles
        theta = np.zeros((N, self.pmodel.dim()))
        llh = np.zeros(N)  # log likelihood
        log_prior = np.zeros(N)  # log prior
        theta[:] = self.starting_dist.draw(N)
        # compute initial likelihood
        llh[:] = self.pmodel.compute_llh(theta)
        log_prior[:] = self.pmodel.compute_prior(theta)
        theta, llh, log_prior = self.orderByModel(theta, llh, log_prior)
        # init the SMC quantities: weights and Z
        models_grouped_by_indices = self.enumerateModels(theta)
        if VERBOSITY_HIGH():
            print("Model identifiers", models_grouped_by_indices.keys())
        smc_quantities = self.makeSMCQuantities(N, llh, log_prior, theta, self.pmodel, None)
        self.setInitialDensity(smc_quantities, llh, log_prior, theta)
        # iterate over sequence of distributions
        t = 0.0
        self.essseq_done[0] = N
        self.mkseq_done[0] = self.getEmpiricalModelProbs(theta)
        progress(t, 1, status="Initialising SMC sampler")
        while t < 1.0:
            last_t = t
            if self.tempseq is None:
                self.preImportanceSampleHook(last_t, llh, log_prior, smc_quantities, self.pmodel, next_t=None)
                # do a safe threshold check
                if (
                    self.ESSThreshold(
                        t,
                        last_t,
                        ess_th_fn,
                        llh,
                        log_prior,
                        smc_quantities,
                        self.pmodel,
                        N,
                    )
                    > 0
                ):
                    max_t = max(1e-20, min(1.0, 2 ** np.ceil(np.log2(t))))
                    if VERBOSITY_HIGH():
                        print("Starting bisection search with max_t={}".format(max_t))
                    while (
                        max_t < 1
                        and self.ESSThreshold(
                            max_t,
                            last_t,
                            ess_th_fn,
                            llh,
                            log_prior,
                            smc_quantities,
                            self.pmodel,
                            N,
                        )
                        > 0
                    ):
                        max_t = 2 ** np.ceil(np.log2(max_t) + 1)
                        if VERBOSITY_HIGH():
                            print("Increasing to max_t={}".format(max_t))
                    if (
                        self.ESSThreshold(
                            max_t,
                            last_t,
                            ess_th_fn,
                            llh,
                            log_prior,
                            smc_quantities,
                            self.pmodel,
                            N,
                        )
                        < 0
                    ):
                        next_t, rres = scipy.optimize.bisect(
                            self.ESSThreshold,
                            t,
                            max_t,
                            args=(
                                last_t,
                                ess_th_fn,
                                llh,
                                log_prior,
                                smc_quantities,
                                self.pmodel,
                                N,
                            ),
                            full_output=True,
                            rtol=1e-6,
                        )
                        t = next_t
                    else:
                        t = 1.0
            else:
                # use the temperature sequence provided.
                # assumes last element of self.tempseq is 1
                t = self.tempseq[self.tempseq.index(t) + 1]
                self.preImportanceSampleHook(last_t, llh, log_prior, smc_quantities, self.pmodel, next_t=t)
            if True:
                self.tempseq_done.append(t)
                # importance sample
                log_target_ratio = self.computeIncrementalWeights(
                    t,
                    last_t,
                    llh,
                    log_prior,
                    self.pmodel,
                    theta,
                    smc_quantities.log_w_norm,
                    store_D=True,
                )
                smc_quantities.log_w[:] = smc_quantities.log_w_norm + log_target_ratio
                log_w_sum = logsumexp(smc_quantities.log_w)
                smc_quantities.log_Zt += log_w_sum
                smc_quantities.log_w_norm[:] = smc_quantities.log_w - log_w_sum
                smc_quantities.log_ESS = -logsumexp(2 * smc_quantities.log_w_norm)
                if VERBOSITY_HIGH():
                    print("Calibrating at t = {}, Total ESS = {}...".format(t, np.exp(smc_quantities.log_ESS)))
                progress(
                    t,
                    1,
                    status="Calibrating at inverse temperature = {}, ESS = {}".format(
                        t, np.exp(smc_quantities.log_ESS)
                    ),
                )
                self.essseq_done[t] = np.exp(smc_quantities.log_ESS)

                # CALIBRATE HERE
                self.appendToMixtureTargetDensity(smc_quantities, llh, log_prior, theta, float(t))
                self.pmodel.calibrateProposalsMMMPD(self.getMixtureTargetDensity(), N, float(t))

                if VERBOSITY_HIGH():
                    print("Resampling at t = {}, Total ESS = {}...".format(t, np.exp(smc_quantities.log_ESS)))
                progress(
                    t,
                    1,
                    status="Resampling at inverse temperature = {}, ESS = {}".format(t, np.exp(smc_quantities.log_ESS)),
                )

                # resample
                smc_quantities.resample_indices = self.resample(np.exp(smc_quantities.log_w_norm), N)
                resample_indices_global = smc_quantities.indices[smc_quantities.resample_indices]
                theta[smc_quantities.indices] = theta[resample_indices_global]
                llh[smc_quantities.indices] = llh[resample_indices_global]
                log_prior[smc_quantities.indices] = log_prior[resample_indices_global]

                # set empirical model probs
                self.mkseq_done[t] = self.getEmpiricalModelProbs(theta)

                if VERBOSITY_HIGH():
                    print("Mutating at t = {}, Total ESS = {}...".format(t, np.exp(smc_quantities.log_ESS)))
                progress(
                    t,
                    1,
                    status="MCMC mutation at inverse temperature = {}, ESS = {}".format(
                        t, np.exp(smc_quantities.log_ESS)
                    ),
                )
                # mutate
                theta[:], llh[:], log_prior[:], accepted = self.mutate(theta, llh, log_prior, N, t, smc_quantities)
                # TODO may need to order accepted too
                theta, llh, log_prior = self.orderByModel(theta, llh, log_prior)
                # update the smc quantities
                smc_quantities.updateAfterMutate(np.arange(N), llh, log_prior, theta)
                # Set this posterior to use the final particles after mutation
        progress(t, 1, status="Finished")
        return (
            theta,
            smc_quantities,
            llh,
            log_prior,
            self.getMixtureTargetDensity(),
            self.rbar_list,
        )

    def orderByModel(self, theta, llh, log_prior):
        # order by model
        ordered_theta_list = []
        ordered_llh_list = []
        ordered_log_prior_list = []
        models_grouped_by_indices = self.enumerateModels(theta)
        for k, m_indices in models_grouped_by_indices.items():
            ordered_theta_list.append(theta[m_indices])
            ordered_llh_list.append(llh[m_indices])
            ordered_log_prior_list.append(log_prior[m_indices])
        return (
            np.vstack(ordered_theta_list),
            np.hstack(ordered_llh_list),
            np.hstack(ordered_log_prior_list),
        )

    def computeMixtureTargetDensity(self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True):
        """
        Deprecated
        """
        # init previous_model_targets each time
        previous_model_targets = SingleModelMPD(self.pmodel)
        previous_model_targets.addComponent(self.initialdensity)
        previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))
        return previous_model_targets.getParticleDensityForTemperature(gamma_t, resample)

    def setInitialDensity(self, smc_quantities, llh, log_prior, theta):
        self.initialdensity = smc_quantities.makePPPD(0)
        self.previous_model_targets.addComponent(self.initialdensity)

    def appendToMixtureTargetDensity(self, smc_quantities, llh, log_prior, theta, gamma_t):
        # init previous_model_targets each time
        self.previous_model_targets = SingleModelMPD(self.pmodel)
        self.previous_model_targets.addComponent(self.initialdensity)
        self.previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))

    def getMixtureTargetDensity(self):
        return self.previous_model_targets

    def enumerateModels(self, theta):
        """
        Associate each model with a key. Typically in a single rjmcmc scheme this would be a tuple of the number of layers (n,)
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return model_key_dict

    def reverseEnumerateModels(self, theta):
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return reverse_key_ref

    def getModelKeyArray(self, theta):
        """
        return numpy array of model keys of each row of theta
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        mk_keys = model_key_dict.keys()
        ncols = len(list(mk_keys)[0])
        keyarray = np.zeros((theta.shape[0], ncols))
        for mk, idx in model_key_dict.items():
            keyarray[idx] = np.array(list(mk))
        return keyarray

    def resample(self, weights, n):
        indices = np.zeros(n, dtype=np.int32)
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
                if j >= n:
                    j = 0
            indices[i] = j
        return indices

    def mutate(self, theta, llh, log_prior, N, t, smc_quantities):
        global TESTBIAS
        global PLOTMARGINALS
        # switch to say whether to use total acceptance rate or min acceptance rate
        use_arn_total = False
        """
        Uses Rt method (Drovandi & Pettitt)
        R_t = ceil(log(0.01) / log(1 - ar))
        where 0<=ar<=1 is the acceptance rate
        """
        # pilot run with acceptance rate
        ar = 0.44
        new_theta = np.zeros_like(theta)
        new_llh = np.zeros_like(llh)
        new_log_prior = np.zeros_like(log_prior)
        new_theta[:] = theta
        new_llh[:] = llh
        new_log_prior[:] = log_prior
        ar_total = 1.0
        arn_total = 0
        arn_repeats = 3
        min_ar = 1.0
        if self.n_mutations is not None:
            R_t = self.n_mutations
            total_accepted = np.zeros(N)
        else:
            for i in range(arn_repeats):
                (
                    new_theta[:],
                    new_llh[:],
                    new_log_prior[:],
                    accepted,
                ) = self.single_mutation(new_theta, new_llh, N, t)
            for pid, prop in Proposal.idpropdict.items():
                this_ar = prop.getAvgARN(arn_repeats) / (N * arn_repeats)
                arn_total += this_ar
                min_ar = min(min_ar, prop.getLastAR())
            if use_arn_total:
                ar_total = arn_total
            else:
                ar_total = min_ar
            # TODO compute minimum acceptance ratio given each proposal type
            if VERBOSITY_HIGH():
                print("Acceptance rate = {}".format(ar_total))
            # R_t = int(np.ceil(np.log(0.001) / np.log(1 - ar_total)))
            R_t = np.ceil(np.log(0.001) / np.log(1 - ar_total))
            if np.isfinite(R_t):
                R_t = int(R_t)
            else:
                R_t = 50
            if TESTBIAS:
                R_t = max(100, R_t)
            elif True:
                R_t = R_t  # min(500,R_t)
            total_accepted = np.zeros(N) + accepted
        if VERBOSITY_HIGH():
            print("mutating {} times".format(R_t))
        # include pilot run and do remaining runs
        for r in range(R_t):
            new_theta[:], new_llh[:], new_log_prior[:], accepted = self.single_mutation(new_theta, new_llh, N, t)
            total_accepted += accepted
        return (
            new_theta,
            new_llh,
            new_log_prior,
            total_accepted,
        )  # what are we doing with the accepted totals? Also include proposals?

    def single_mutation(self, theta, llh, N, t, blocksize=1000):
        global PLOTMARGINALS
        # kblocks = np.unique(theta[:,0])
        kblocks, rev = self.pmodel.enumerateModels(theta)
        # print("kblocks",kblocks)
        for mk, mkidx in kblocks.items():
            if VERBOSITY_HIGH():
                print("Single mutation count {} particles for model {}".format(mkidx.shape[0], mk))  # n,k
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros_like(llh)
        prop_llh = np.full(llh.shape, -np.inf)
        cur_prior = np.zeros_like(llh)
        prop_prior = np.zeros_like(llh)
        prop_id = np.zeros_like(llh)
        prop_lpqratio = np.zeros_like(llh)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # propose
        nblocks = int(np.ceil((1.0 * N) / blocksize))
        blocks = [np.arange(i * blocksize, min(N, (i + 1) * blocksize)) for i in range(nblocks)]
        # TODO reuse this computation from constructor for mmmpd
        for bidx in blocks:
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(theta[bidx], bidx.shape[0])
        cur_prior[:] = self.pmodel.compute_prior(theta)
        prop_prior[:] = self.pmodel.compute_prior(prop_theta)
        ninfprioridx = np.where(~np.isfinite(cur_prior))
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(np.isfinite(prop_prior), np.isfinite(prop_lpqratio))
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta,
            prop_theta,
            prop_lpqratio,
            prop_llh,
            llh,
            cur_prior,
            prop_prior,
            float(t),
        )

        # store acceptance ratios
        if self.store_ar:
            self.rbar_list.append(
                rbar(
                    log_acceptance_ratio,
                    float(t),
                    self.getModelKeyArray(theta),
                    self.getModelKeyArray(prop_theta),
                )
            )

        Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, float(t))
        for pid, prop in Proposal.idpropdict.items():
            if VERBOSITY_HIGH():
                print("for pid {}\tprop {}\tar {}".format(pid, prop.printName(), prop.getLastAR()))

        # accept/reject
        log_u = np.log(uniform.rvs(0, 1, size=N))
        reject_indices = log_acceptance_ratio < log_u
        prop_theta[reject_indices] = theta[reject_indices]
        prop_llh[reject_indices] = llh[reject_indices]
        prop_prior[reject_indices] = cur_prior[reject_indices]
        # a boolean array of accepted proposals
        accepted = np.ones(N)
        accepted[reject_indices] = 0
        self.nmutations += 1
        return prop_theta, prop_llh, prop_prior, np.exp(log_acceptance_ratio)

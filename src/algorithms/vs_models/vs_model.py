import numpy as np

from src.variables import ParametricModelSpace


class RobustBlockVSModel(ParametricModelSpace):
    def __init__(self, problem, k=None, rj_only=False, **kwargs):
        self.problem = problem

        self.nblocks = 3
        self.blocksizes = [1, 1, 2]
        if k is None:
            self.minblockcount = [1, 0, 0]
            self.maxblockcount = [1, 1, 1]
        else:
            self.minblockcount = k
            self.maxblockcount = k

        self.blocknames = [f"b{i}" for i in range(self.nblocks)]
        self.betanames = [f"beta{i}{j}" for i in range(self.nblocks) for j in range(self.blocksizes[i])]
        self.gammanames = [f"gamma{i}" for i in range(self.nblocks)]

        random_variables = self._setup_random_variables(k, rj_only, **kwargs)
        proposal = self._setup_proposal(k, rj_only, **kwargs)
        super().__init__(random_variables, proposal)

    def _setup_random_variables(self, k, rj_only, **kwargs):
        raise NotImplementedError

    def _setup_proposal(self, k, rj_only, **kwargs):
        raise NotImplementedError

    def compute_llh(self, theta):
        """
        target <- function(x){
          p <- length(x)
          a <- X%*%x
          mn <- exp(-(y - a)^2/2) + exp(-(y - a)^2/200)/10 # Normal mix part
          phi_0 <- log(mn)   ## Log likelihood
          log_q <- sum(phi_0)  + sum(x^2/200)  ## Add a N(0,10) prior
          return(list(log_q = log_q))
        }
        """
        x_data = self.problem.x_data
        y_data = self.problem.y_data

        cols = self.generateRVIndices()
        betas_stack = []

        for bn in self.betanames:
            # bn = self.betanames[i]
            if len(cols[bn]) > 0:
                betas_stack.append(theta[:, cols[bn]])
            else:
                betas_stack.append(np.zeros((theta.shape[0], 1)))
        betas = np.column_stack(betas_stack)
        gammas = np.column_stack([theta[:, cols[i]] for i in self.gammanames])

        model_enumeration, rev = self.enumerateModels(theta)
        # likelihood is as follows
        # for i in gammas
        # y_i ~ Bern(p_i)
        # p_i = exp(x^T beta)/(1+exp(x^T beta))
        log_like = np.zeros(theta.shape[0])
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            gamma_vec = gammas[mk_row_idx][0]
            x_data_active = x_data[:, gamma_vec.astype(bool).repeat(self.blocksizes)]  # n x p
            betas_active = betas[mk_row_idx][:, gamma_vec.astype(bool).repeat(self.blocksizes)]
            a = np.dot(
                betas_active, x_data_active.T
            )  # nrows x p dot p x ndata = nrows x ndata #n x p dot p x 1 = n x 1
            log_like[mk_row_idx] = np.log(
                np.exp(-((y_data - a) ** 2) / 2) + np.exp(-((y_data - a) ** 2) / 200) / 10
            ).sum(axis=1)
        return log_like

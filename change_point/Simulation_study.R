## Remove all R objects ##
rm(list = setdiff(ls(), lsf.str()))

# function to compute exactly the ESS per iteration of any discrete and finite state space Markov chain (following Theorem 4.8 in Iosifescu (1980))
# it is used to compute the ESS per iteration for the ideal samplers
ESS_discrete <- function(vect_prob, transition, vect_f){
  # vect_prob: invariant distribution
  # transition: transition matrix
  # vect_f: vector containing the different values of the function f with respect to which we want to compute the ESS
  
  K_max <- length(vect_prob)
  
  A <- matrix(rep(vect_prob, each = K_max), nrow = K_max)
  
  I <- diag(1, nrow = K_max)
  
  Z <- solve(I - (transition - A))
  
  # we need to compute the variance under the target
  exp_f <- t(vect_prob) %*% vect_f
  var_f <- t(vect_prob) %*% as.matrix((vect_f - rep(exp_f, K_max))^2)
  
  vect_inner <- ((Z - A) %*% as.matrix(vect_f - rep(exp_f, K_max))) * as.matrix(vect_f - rep(exp_f, K_max))
  inner <- t(vect_prob) %*% vect_inner

  return(var_f / (2 * inner - var_f))
  
}

# function to compute the PMF defined in Section 4.2
PMF_phi <- function(K_max, phi){
  # we assume that K_max is odd
  
  mode <- (K_max + 1) / 2
  
  vect_prob <- matrix(nrow = K_max, ncol = 1)
  vect_prob[mode] <- 1
  
  seq_factors <- 1 / phi^seq(1, mode - 1)
  
  vect_prob[(mode + 1):K_max] <- seq_factors
  vect_prob[(mode - 1):1] <- seq_factors
  
  return(vect_prob / sum(vect_prob))
  
}

### Section 4.2 ###

# the figure in this section is produced at the same time we produce those in Section 5.1

# Computation of the total mass outside of k^*, k^* - 1 and k^* + 1 when \phi := 7

phi <- 7
# when K_max = 11
const <- 1 + 2 * (1 / phi) * ((1 - 1 / phi^5) / (1 - 1 / phi))
2 * (1 / phi^2) * ((1 - 1 / phi^4) / (1 - 1 / phi)) / const
# limiting case
const <- 1 + 2 * (1 / phi) * (1 / (1 - 1 / phi))
2 * (1 / phi^2) * (1 / (1 - 1 / phi)) / const

### Section 5.1 ###

## Figure 3 and that in Section 4.2 ##

# For Algorithm 1 and RJ we can compute exactly the ESS (considering only the evolution of K)
# using the transition matrices

# For computing the probability to go from model k to model k + 1
# we have to compute an integral with respect to q, given k
# we integrate out the proposal u_{k \mapsto k+1}
# we do this using the following function giving the exact expression
int_q <- function(prob_num, prob_denom, sigma){
  
  if(sigma == 1){
    
    return(min(1, prob_num / prob_denom))
    
  }
  if(sigma > 1){
    
    quantity <- (2 / (1 - 1 / sigma^2)) * (log(prob_num) - log(prob_denom) + log(sigma))
    
    if(quantity > 0){
      
      prob1 <- 2 * (1 - pnorm(sqrt(quantity), mean = 0, sd = 1))
      prob2 <- 2 * pnorm(sqrt(quantity), mean = 0, sd = sigma) - 1
      
    }
    else{
      
      prob1 <- 1
      prob2 <- 0
      
    }
    
    return((prob_num / prob_denom) * prob1 + prob2)
    
  }
  if(sigma < 1){
    
    quantity <- (2 / (1 - 1 / sigma^2)) * (log(prob_num) - log(prob_denom) + log(sigma))
    
    if(quantity > 0){
      
      prob1 <- 2 * (pnorm(sqrt(quantity), mean = 0, sd = 1) - 0.5)
      prob2 <- 2 * (1 - pnorm(sqrt(quantity), mean = 0, sd = sigma))
      
    }
    else{
      
      prob1 <- 0
      prob2 <- 1
      
    }
    
    return((prob_num / prob_denom) * prob1 + prob2)
    
  }
  
}

# For computing the probability to go from Model k + 1 to Model k, we proceed similarly, but
# we have to compute an integral with respect to f
# we integrate out the last parameter x_{k+1, k+1}
# we do this using the following function
int_f <- function(prob_num, prob_denom, sigma){
  
  if(sigma == 1){
    
    return(min(1, prob_num / prob_denom))
    
  }
  if(sigma < 1){
    
    quantity <- -(2 / (1 - 1 / sigma^2)) * (log(prob_num) - log(prob_denom) - log(sigma))
    
    if(quantity > 0){
      
      prob1 <- 2 * (1 - pnorm(sqrt(quantity), mean = 0, sd = sigma))
      prob2 <- 2 * pnorm(sqrt(quantity), mean = 0, sd = 1) - 1
      
    }
    else{
      
      prob1 <- 1
      prob2 <- 0
      
    }
    
    return((prob_num / prob_denom) * prob1 + prob2)
    
  }
  if(sigma > 1){
    
    quantity <- -(2 / (1 - 1 / sigma^2)) * (log(prob_num) - log(prob_denom) - log(sigma))
    
    if(quantity > 0){
      
      prob1 <- 2 * (pnorm(sqrt(quantity), mean = 0, sd = sigma) - 0.5)
      prob2 <- 2 * (1 - pnorm(sqrt(quantity), mean = 0, sd = 1))
      
    }
    else{
      
      prob1 <- 0
      prob2 <- 1
      
    }
    
    return((prob_num / prob_denom) * prob1 + prob2)
    
  }
  
}

# we compute the (one step) transition matrix with the noise coming from the parameter proposals for RJ
transition_RJ_noise <- function(K_max, vect_prob, sigma, mat_prop){
  
  transition <- matrix(ncol = K_max, nrow = K_max, 0)
  
  transition[1, 2] <- int_q(vect_prob[2] * mat_prop[2, 1], vect_prob[1] * mat_prop[1, 2], sigma)
  transition[1, 1] <- 1 - transition[1, 2]
  for(i in 2:(K_max - 1)){
    
    transition[i, i + 1] <- mat_prop[i, 2] * int_q(vect_prob[i + 1] * mat_prop[i + 1, 1], vect_prob[i] * mat_prop[i, 2], sigma)
    transition[i, i - 1] <- mat_prop[i, 1] * int_f(vect_prob[i - 1] * mat_prop[i - 1, 2], vect_prob[i] * mat_prop[i, 1], sigma)
    transition[i, i] <- 1 - transition[i, i + 1] - transition[i, i - 1]
    
  }
  transition[K_max, K_max - 1] <- int_f(vect_prob[K_max - 1] * mat_prop[K_max - 1, 2], vect_prob[K_max] * mat_prop[K_max, 1], sigma)
  transition[K_max, K_max] <- 1 - transition[K_max, K_max - 1]
  
  return(transition)
  
}

# we compute the (one step) transition matrix for NRJ
transition_NRJ_noise <- function(K_max, vect_prob, sigma){
  
  # the first K_max positions correspond to (k, +1) (going to the right), the next K_max positions are (k, -1)
  transition <- matrix(ncol = 2 * K_max, nrow = 2 * K_max, 0)
  
  # we first compute the part where we start with +1
  for(i in 1:(K_max - 1)){
    
    transition[i, i + 1] <- int_q(vect_prob[i + 1], vect_prob[i], sigma)
    transition[i, K_max + i] <- 1 - transition[i, i + 1]
    
  }
  transition[K_max, 2 * K_max] <- 1
  
  # we compute the part where we start with -1
  transition[K_max + 1, 1] <- 1
  for(i in (K_max + 2):(2 * K_max)){
    
    transition[i, i - 1] <- int_f(vect_prob[i - K_max - 1], vect_prob[i - K_max], sigma)
    transition[i, i - K_max] <- 1 -  transition[i, i - 1]
    
  }
  
  return(transition)
  
}

# We now introduce Algorithms 2 and 3 (and what is required to run them)

# The PMF to decide if we update the parameters or switch models
g_NRJ <- function(tau){
  
  u <- runif(1)
  
  if(u <= tau){return(1)} # for parameter updates
  else{
    
    return(2) # for model switchings
    
  }
  
}

Algorithm2 <- function(nb_iter, ini_val_K, ini_val_param, ini_val_d, K_max, sigma, vect_prob, tau, T){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # T is the parameter for the method of Karagiannis and Andrieu (2013)
  
  # We record the states of the chain in three matrices
  # the first one for K
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  # the second one for the parameters
  matrix_states_param <- matrix(ncol = K_max, nrow = nb_iter + 1)
  matrix_states_param[1, 1:ini_val_K] <- ini_val_param
  # the last one for direction
  matrix_states_d <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_d[1] <- ini_val_d
  
  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)
  
  # we count the number of attempts at adding and withdrawing one parameter to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 2, 0)
  success <- matrix(ncol = 1, nrow = 2, 0)
  
  # we define a vect of variances for the proposals when switching models
  vect_gamma <- seq(0, T - 1) / T
  vect_var <- 1 / ((1 - vect_gamma) * sigma^(-2) + vect_gamma)
  
  for(i in 2:(nb_iter + 1)){
    
    # What is the current model?
    k <- matrix_states_K[i - 1]
    
    # we note the direction
    direction <- matrix_states_d[i - 1]
    
    # What is the type of move attempted
    type <- g_NRJ(tau)
    
    if(type == 1){ # we update the parameters, the proposals are always accepted because we generate from the conditional given K
      
      vect_move_type[i] <- 1
      matrix_states_K[i] <- k
      matrix_states_d[i] <- direction
      matrix_states_param[i, 1:k] <- rnorm(k)
      
    }
    if(type == 2 && direction == 1){ # it is proposed to add a parameter
      
      current_param <- matrix_states_param[i - 1, 1:k]
      
      counts[1] <- counts[1] + 1
      vect_move_type[i] <- 2
      
      if(k < K_max){
        
        # we generate T proposals
        proposals <- rnorm(T, mean = 0, sd = sqrt(vect_var))
        
        # we compute the (log) acceptance probability
        acc_prob <- log(vect_prob[k + 1]) - log(vect_prob[k]) + log(sigma) - (mean(proposals^2) / 2) * (1 - 1 / sigma^2)
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_K[i] <- k + 1
          matrix_states_param[i, 1:(k + 1)] <- c(current_param, proposals[T])
          matrix_states_d[i] <- direction
          success[1] <- success[1] + 1
          
        }
        else{
          
          matrix_states_K[i] <- k
          matrix_states_d[i] <- -direction
          matrix_states_param[i, 1:k] <- current_param
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_d[i] <- -direction
        matrix_states_param[i, 1:k] <- current_param
        
      }
      
    }
    if(type == 2 && direction == -1){ # it is proposed to withdraw a parameter
      
      current_param <- matrix_states_param[i - 1, 1:k]
      
      counts[2] <- counts[2] + 1
      vect_move_type[i] <- 2
      
      if(k > 1){
        
        # we generate T - 1 proposals (and we include the current value of the last parameter)
        proposals <- c(current_param[k], rnorm(T - 1, mean = 0, sd = sqrt(vect_var[T:2])))
        
        # we compute the (log) acceptance probability
        acc_prob <- log(vect_prob[k - 1]) - log(vect_prob[k]) - log(sigma) + (mean(proposals^2) / 2) * (1 - 1 / sigma^2)
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_K[i] <- k - 1
          matrix_states_param[i, 1:(k - 1)] <- current_param[1:(k - 1)]
          matrix_states_d[i] <- direction
          success[2] <- success[2] + 1
          
        }
        else{
          
          matrix_states_K[i] <- k
          matrix_states_d[i] <- -direction
          matrix_states_param[i, 1:k] <- current_param
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_d[i] <- -direction
        matrix_states_param[i, 1:k] <- current_param
        
      }
      
    }
    
  }
  
  return(list(matrix_states_K = matrix_states_K[2:(nb_iter + 1)], matrix_states_param = matrix_states_param[2:(nb_iter + 1), ], matrix_states_d = matrix_states_d[2:(nb_iter + 1)],  acc_rate = success / counts, vect_move_type = vect_move_type[2:(nb_iter + 1)]))
  
}

Algorithm3 <- function(nb_iter, ini_val_K, ini_val_param, ini_val_d, K_max, sigma, vect_prob, tau, T, N){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # T is the parameter for the method of Karagiannis and Andrieu (2013)
  # N is the parameter for the method of Andrieu et al. (2018)
  
  # We record the states of the chain in three matrices
  # the first one for K
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  # the second one for the parameters
  matrix_states_param <- matrix(ncol = K_max, nrow = nb_iter + 1)
  matrix_states_param[1, 1:ini_val_K] <- ini_val_param
  # the last one for \nu
  matrix_states_d <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_d[1] <- ini_val_d
  
  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)
  
  # we count the number of attempts at adding and withdrawing one parameter to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 2, 0)
  success <- matrix(ncol = 1, nrow = 2, 0)
  
  # we define a vector of variances for the proposals when switching models
  vect_gamma <- seq(0, T - 1) / T
  vect_var <- 1 / ((1 - vect_gamma) * sigma^(-2) + vect_gamma)
  
  for(i in 2:(nb_iter + 1)){
    
    # What is the current model?
    k <- matrix_states_K[i - 1]
    
    # we note the direction
    direction <- matrix_states_d[i - 1]
    
    # What is the type of move attempted
    type <- g_NRJ(tau)
    
    if(type == 1){ # we update the parameters, the proposals are always accepted because we generate from the conditional given K
      
      vect_move_type[i] <- 1
      matrix_states_K[i] <- k
      matrix_states_d[i] <- direction
      matrix_states_param[i, 1:k] <- rnorm(k)
      
    }
    if(type == 2 && direction == 1){ # it is proposed to add a parameter
      
      current_param <- matrix_states_param[i - 1, 1:k]
      
      counts[1] <- counts[1] + 1
      vect_move_type[i] <- 2
      
      if(k < K_max){
        
        u_c <- runif(1)
        
        matrix_proposals <- matrix(nrow = N, ncol = T)
        vect_acc_prob <- NULL
        
        if(u_c <= 0.5){
          
          for(j in 1:N){
            
            # we generate T proposals
            matrix_proposals[j, ] <- rnorm(T, mean = 0, sd = sqrt(vect_var))
            
            # we compute the (log) acceptance probability
            vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k + 1]) - log(vect_prob[k]) + log(sigma) - (mean(matrix_proposals[j, ]^2) / 2) * (1 - 1 / sigma^2))
            
          }
          
          if(runif(1) <= mean(exp(vect_acc_prob))){
            
            matrix_states_K[i] <- k + 1
            j_star <- sample(1:N, size = 1, prob = exp(vect_acc_prob))
            matrix_states_param[i, 1:(k + 1)] <- c(current_param, matrix_proposals[j_star, T])
            matrix_states_d[i] <- direction
            success[1] <- success[1] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_d[i] <- -direction
            matrix_states_param[i, 1:k] <- current_param
            
          }
          
        }
        else{
          
          # We generate the forward proposal
          matrix_proposals[1, ] <- rnorm(T, mean = 0, sd = sqrt(vect_var))
          
          # we compute the (log) acceptance probability, as if we were coming from k + 1
          vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k]) - log(vect_prob[k + 1]) - log(sigma) + (mean(matrix_proposals[1, ]^2) / 2) * (1 - 1 / sigma^2))
          
          # From the endpoint matrix_proposals[1, T], we generate N - 1 reverse paths
          for(j in 2:N){
            
            matrix_proposals[j, ] <- c(matrix_proposals[1, T], rnorm(T - 1, mean = 0, sd = sqrt(vect_var[T:2])))
            vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k]) - log(vect_prob[k + 1]) - log(sigma) + (mean(matrix_proposals[j, ]^2) / 2) * (1 - 1 / sigma^2))
            
          }
          
          if(runif(1) <= 1 / mean(exp(vect_acc_prob))){
            
            matrix_states_K[i] <- k + 1
            matrix_states_param[i, 1:(k + 1)] <- c(current_param, matrix_proposals[1, T])
            matrix_states_d[i] <- direction
            success[1] <- success[1] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_d[i] <- -direction
            matrix_states_param[i, 1:k] <- current_param
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_d[i] <- -direction
        matrix_states_param[i, 1:k] <- current_param
        
      }
      
    }
    if(type == 2 && direction == -1){ # it is proposed to withdraw a parameter
      
      current_param <- matrix_states_param[i - 1, 1:k]
      
      counts[2] <- counts[2] + 1
      vect_move_type[i] <- 2
      
      if(k > 1){
        
        u_c <- runif(1)
        
        matrix_proposals <- matrix(nrow = N, ncol = T)
        vect_acc_prob <- NULL
        
        if(u_c <= 0.5){
          
          for(j in 1:N){
            
            # we generate T - 1 proposals (and we include the current value of the last parameter)
            matrix_proposals[j, ] <- c(current_param[k], rnorm(T - 1, mean = 0, sd = sqrt(vect_var[T:2])))
            
            # we compute the (log) acceptance probability
            vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k - 1]) - log(vect_prob[k]) - log(sigma) + (mean(matrix_proposals[j, ]^2) / 2) * (1 - 1 / sigma^2))
            
          }
          
          if(runif(1) <= mean(exp(vect_acc_prob))){
            
            matrix_states_K[i] <- k - 1
            matrix_states_param[i, 1:(k - 1)] <- current_param[1:(k - 1)]
            matrix_states_d[i] <- direction
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_d[i] <- -direction
            matrix_states_param[i, 1:k] <- current_param
            
          }
          
        }
        else{
          
          # We generate the forward proposal
          matrix_proposals[1, ] <- c(current_param[k], rnorm(T - 1, mean = 0, sd = sqrt(vect_var[T:2])))
          
          # we compute the (log) acceptance probability, as if we were coming from k - 1
          vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k]) - log(vect_prob[k - 1]) + log(sigma) - (mean(matrix_proposals[1, ]^2) / 2) * (1 - 1 / sigma^2))
          
          # From the endpoint matrix_proposals[1, T] (in fact we don't use it), we generate N - 1 reverse paths
          for(j in 2:N){
            
            matrix_proposals[j, ] <- rnorm(T, mean = 0, sd = sqrt(vect_var))
            vect_acc_prob <- c(vect_acc_prob, log(vect_prob[k]) - log(vect_prob[k - 1]) + log(sigma) - (mean(matrix_proposals[j, ]^2) / 2) * (1 - 1 / sigma^2))
            
          }
          
          if(runif(1) <= 1 / mean(exp(vect_acc_prob))){
            
            matrix_states_K[i] <- k - 1
            matrix_states_param[i, 1:(k - 1)] <- current_param[1:(k - 1)]
            matrix_states_d[i] <- direction
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_d[i] <- -direction
            matrix_states_param[i, 1:k] <- current_param
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_d[i] <- -direction
        matrix_states_param[i, 1:k] <- current_param
        
      }
      
    }
    
  }
  
  return(list(matrix_states_K = matrix_states_K[2:(nb_iter + 1)], matrix_states_param = matrix_states_param[2:(nb_iter + 1), ], matrix_states_d = matrix_states_d[2:(nb_iter + 1)],  acc_rate = success / counts, vect_move_type = vect_move_type[2:(nb_iter + 1)]))
  
}

# Figure 3 (a)
# We compute the ESS per iteration as a function of sigma

phi <- 2
K_max <- 11
vect_prob <- PMF_phi(K_max, phi)
seq_sigma <- seq(0.01, 3, by = 0.01)
vect_ESS_RJ_sym <- matrix(ncol = 1, nrow = length(seq_sigma))
vect_ESS_RJ_skewed <- matrix(ncol = 1, nrow = length(seq_sigma))
vect_ESS_Algo1 <- matrix(ncol = 1, nrow = length(seq_sigma))

# RJ
vect_prob_RJ <- PMF_phi(K_max, phi)
vect_f_RJ <- seq(1, K_max)
# symmetric proposal distribution
mat_prop_sym <- matrix(nrow = K_max, ncol = 2, 1 / 2)
mat_prop_sym[1, 1] <- mat_prop_sym[K_max, 2] <- 0
mat_prop_sym[1, 2] <- mat_prop_sym[K_max, 1] <- 1
# skewed proposal distribution (proportional to \sqrt{\pi(k' \mid D) / \pi(k \mid D)})
mat_prop_ske <- matrix(nrow = K_max, ncol = 2)
mat_prop_ske[1, 1] <- mat_prop_ske[K_max, 2] <- 0
mat_prop_ske[1, 2] <- mat_prop_ske[K_max, 1] <- 1
mat_prop_ske[2:(K_max - 1), 2] <- sqrt(vect_prob_RJ[3:K_max] / vect_prob_RJ[2:(K_max - 1)])
mat_prop_ske[2:(K_max - 1), 1] <- sqrt(vect_prob_RJ[1:(K_max - 2)] / vect_prob_RJ[2:(K_max - 1)])
for(j in 2:(K_max - 1)){
  
  mat_prop_ske[j, ] <- mat_prop_ske[j, ] / sum(mat_prop_ske[j, ])
  
}

# NRJ
vect_f_NRJ <- rep(seq(1, K_max), 2)
vect_prob_NRJ <- as.matrix(rep(vect_prob_RJ, 2)) / 2

for(i in 1:length(seq_sigma)){ # we vary sigma
  
  sigma <- seq_sigma[i]
  
  # RJ
  transition <- transition_RJ_noise(K_max, vect_prob_RJ, sigma, mat_prop_sym)
  vect_ESS_RJ_sym[i] <- ESS_discrete(vect_prob_RJ, transition, vect_f_RJ)
  
  transition <- transition_RJ_noise(K_max, vect_prob_RJ, sigma, mat_prop_ske)
  vect_ESS_RJ_skewed[i] <- ESS_discrete(vect_prob_RJ, transition, vect_f_RJ)
  
  # Algo1
  transition <- transition_NRJ_noise(K_max, vect_prob_RJ, sigma)
  vect_ESS_Algo1[i] <- ESS_discrete(vect_prob_NRJ, transition, vect_f_NRJ)
  
}

set.seed(1)

require(coda)
require(LaplacesDemon)

nb_iter <- 100000
tau <- 0.6

vect_ESS_NRJ2 <- matrix(ncol = 1, nrow = length(seq_sigma), 0)
vect_ESS_NRJ3 <- matrix(ncol = 1, nrow = length(seq_sigma), 0)

for(i in 1:1000){ # run this loop in parallel, one iteration of this loop takes about 55 minutes on a regular laptop
  # We ran it in parallel by running 75 times this script with different seeds in batch mode in Linux on 75 CPUs, so it took about 55 * 1000 / 75 \approx 13 hours
  # To run them in batch, one has to use the console and add a line in the loop to save the results in csv files (for instance)
  # and then put the results together
  # for a test, one can run 24 iterations on a laptop thus taking about 1 day and observe the results
  # the results will be rough approximations of the true values

  for(j in 1:length(seq_sigma)){ 

    # we generate an initial value for K
    ini_val_K <- sample(1:K_max, size = 1, prob = vect_prob)
    ini_val_param <- rnorm(ini_val_K)
    ini_val_d <- sample(c(-1, 1), size = 1)
    results <- Algorithm2(nb_iter, ini_val_K, ini_val_param, ini_val_d, K_max, seq_sigma[j], vect_prob, tau, 15)
    vect_ESS_NRJ2[j] <- vect_ESS_NRJ2[j] + ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))
    results <- Algorithm3(nb_iter, ini_val_K, ini_val_param, ini_val_d, K_max, seq_sigma[j], vect_prob, tau, 15, 15)
    vect_ESS_NRJ3[j] <- vect_ESS_NRJ3[j] + ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))

  }

}

results_sigma_NRJ2 <- vect_ESS_NRJ2 / 1000
results_sigma_NRJ3 <- vect_ESS_NRJ3 / 1000

par(mar = c(4, 5, 1, 3))
plot(seq_sigma, vect_ESS_Algo1, type = "l", cex.lab = 1.5, lty = 1, cex.axis = 1.5, cex = 1.5, lwd = 6, xlab = expression(sigma), ylab = "ESS per iteration", col = "darkgreen", xlim = c(0, 3))
lines(seq_sigma, vect_ESS_RJ_sym, lwd = 6, col = "darkred")
lines(seq_sigma, vect_ESS_RJ_skewed, lwd = 6, col = "darkorange")
# we make a little adjustment for numerical errors --- the value at the mode at 1 should fit exactly with that in vect_ESS_Algo1
results_sigma_NRJ2 <- smooth.spline(seq_sigma, results_sigma_NRJ2, spar = 0.6)$y * (vect_ESS_Algo1[which(seq_sigma == 1)] / smooth.spline(seq_sigma, results_sigma_NRJ2, spar = 0.6)$y[which(seq_sigma == 1)])
lines(seq_sigma, results_sigma_NRJ2, lwd = 6, col = "darkgreen", lty = 3)
results_sigma_NRJ3 <- smooth.spline(seq_sigma, results_sigma_NRJ3, spar = 0.6)$y * (vect_ESS_Algo1[which(seq_sigma == 1)] / smooth.spline(seq_sigma, results_sigma_NRJ3, spar = 0.6)$y[which(seq_sigma == 1)])
lines(seq_sigma, results_sigma_NRJ3, lwd = 6, col = "darkgreen", lty = 2)
legend("bottomright", c("Alg. 1", "Alg. 2", "Alg. 3", "RJ (o.)", "RJ (s.)"), text.width = strwidth("RJ (o.)")[1] * 1.95, col = c("darkgreen", "darkgreen", "darkgreen", "darkorange", "darkred"), lty = c(1, 3, 2, 1, 1), cex = 1.5, lwd = c(6, 6, 6, 6, 6))

## Figure 3 (b) ##

phi <- 2
seq_Kmax <- seq(5, 25, by = 2)
vect_ESS_RJ_sym <- matrix(ncol = 1, nrow = length(seq_Kmax))
vect_ESS_RJ_skewed <- matrix(ncol = 1, nrow = length(seq_Kmax))
vect_ESS_NRJ <- matrix(ncol = 1, nrow = length(seq_Kmax))
for(i in 1:length(seq_Kmax)){ # we vary Kmax

  K_max <- seq_Kmax[i]
  vect_prob <- PMF_phi(K_max, phi)
  # RJ
  vect_f <- seq(1, K_max)
  # symmetric proposal distribution
  mat_prop_sym <- matrix(nrow = K_max, ncol = 2, 1 / 2)
  mat_prop_sym[1, 1] <- mat_prop_sym[K_max, 2] <- 0
  mat_prop_sym[1, 2] <- mat_prop_sym[K_max, 1] <- 1
  transition <- transition_RJ_noise(K_max, vect_prob, 1, mat_prop_sym)
  vect_ESS_RJ_sym[i] <- ESS_discrete(vect_prob, transition, vect_f)
  # skewed proposal distribution (proportional to \sqrt{\pi(k' \mid D) / \pi(k \mid D)})
  mat_prop_ske <- matrix(nrow = K_max, ncol = 2)
  mat_prop_ske[1, 1] <- mat_prop_ske[K_max, 2] <- 0
  mat_prop_ske[1, 2] <- mat_prop_ske[K_max, 1] <- 1
  mat_prop_ske[2:(K_max - 1), 2] <- sqrt(vect_prob[3:K_max] / vect_prob[2:(K_max - 1)])
  mat_prop_ske[2:(K_max - 1), 1] <- sqrt(vect_prob[1:(K_max - 2)] / vect_prob[2:(K_max - 1)])
  for(j in 2:(K_max - 1)){

    mat_prop_ske[j, ] <- mat_prop_ske[j, ] / sum(mat_prop_ske[j, ])

  }
  transition <- transition_RJ_noise(K_max, vect_prob, 1, mat_prop_ske)
  vect_ESS_RJ_skewed[i] <- ESS_discrete(vect_prob, transition, vect_f)
  # NRJ
  transition <- transition_NRJ_noise(K_max, vect_prob, 1)
  vect_prob <- as.matrix(rep(vect_prob, 2)) / 2
  vect_f <- rep(seq(1, K_max), 2)
  vect_ESS_NRJ[i] <- ESS_discrete(vect_prob, transition, vect_f)

}

par(mar = c(4, 5, 1, 3))
plot(seq_Kmax, vect_ESS_NRJ, type = "l", cex.lab = 1.5, lty = 1, cex.axis = 1.5, cex = 1.5, lwd = 6, xlab = expression(K[max]), ylab = "ESS per iteration", col = "darkgreen", ylim = c(0, 0.50))
lines(seq_Kmax, vect_ESS_RJ_sym, lwd = 6, col = "darkred")
lines(seq_Kmax, vect_ESS_RJ_skewed, lwd = 6, col = "darkorange")
legend("topright", c("Algorithm 1", "RJ (opt.)", "RJ (symm.)"), text.width = strwidth("RJ (symm.)")[1] * 1.6, col = c("darkgreen", "darkorange", "darkred"), lty = c(1, 1, 1), cex = 1.5, lwd = c(6, 6, 6))

## Figure in Section 4.2 ##

K_max <- 11
seq_phi <- seq(2, 14, by = 0.01)
vect_ESS_RJ_sym <- matrix(ncol = 1, nrow = length(seq_phi))
vect_ESS_RJ_skewed <- matrix(ncol = 1, nrow = length(seq_phi))
vect_ESS_NRJ <- matrix(ncol = 1, nrow = length(seq_phi))
for(i in 1:length(seq_phi)){ # we vary \phi

  phi <- seq_phi[i]
  vect_prob <- PMF_phi(K_max, phi)
  # RJ
  vect_f <- seq(1, K_max)
  # symmetric proposal distribution
  mat_prop_sym <- matrix(nrow = K_max, ncol = 2, 1 / 2)
  mat_prop_sym[1, 1] <- mat_prop_sym[K_max, 2] <- 0
  mat_prop_sym[1, 2] <- mat_prop_sym[K_max, 1] <- 1
  transition <- transition_RJ_noise(K_max, vect_prob, 1, mat_prop_sym) 
  vect_ESS_RJ_sym[i] <- ESS_discrete(vect_prob, transition, vect_f)
  # skewed proposal distribution (proportional to \sqrt{\pi(k' \mid D) / \pi(k \mid D)})
  mat_prop_ske <- matrix(nrow = K_max, ncol = 2)
  mat_prop_ske[1, 1] <- mat_prop_ske[K_max, 2] <- 0
  mat_prop_ske[1, 2] <- mat_prop_ske[K_max, 1] <- 1
  mat_prop_ske[2:(K_max - 1), 2] <- sqrt(vect_prob[3:K_max] / vect_prob[2:(K_max - 1)])
  mat_prop_ske[2:(K_max - 1), 1] <- sqrt(vect_prob[1:(K_max - 2)] / vect_prob[2:(K_max - 1)])
  for(j in 2:(K_max - 1)){

    mat_prop_ske[j, ] <- mat_prop_ske[j, ] / sum(mat_prop_ske[j, ])

  }
  transition <- transition_RJ_noise(K_max, vect_prob, 1, mat_prop_ske)
  vect_ESS_RJ_skewed[i] <- ESS_discrete(vect_prob, transition, vect_f)
  # NRJ
  transition <- transition_NRJ_noise(K_max, vect_prob, 1)
  vect_prob <- as.matrix(rep(vect_prob, 2)) / 2
  vect_f <- rep(seq(1, K_max), 2)
  ESS <- ESS_discrete(vect_prob, transition, vect_f)
  # if(phi > 2.5){vect_ESS_NRJ[i] <- ESS} # When \phi is small, the chain is close to be aperiodic, which gives incorrect results
  vect_ESS_NRJ[i] <- ESS

}

par(mar = c(4, 5, 1, 3))
plot(seq_phi, vect_ESS_NRJ, type = "l", cex.lab = 1.75, lty = 1, cex.axis = 1.75, cex = 1.75, lwd = 6, xlab = expression(phi), ylab = "ESS per iteration", col = "darkgreen", ylim = c(0, 0.60))
lines(seq_phi, vect_ESS_RJ_sym, lwd = 6, col = "darkred")
lines(seq_phi, vect_ESS_RJ_skewed, lwd = 6, col = "darkorange")
legend("topleft", c("NRJ", "RJ (opt.)", "RJ (symm.)"), text.width = strwidth("RJ (symm.)")[1] * 1.70, col = c("darkgreen", "darkorange", "darkred"), lty = c(1, 1, 1), cex = 1.70, lwd = c(6, 6, 6))

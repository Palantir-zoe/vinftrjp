## Remove all R objects ##
rm(list = setdiff(ls(), lsf.str()))

# Set your working directory to where you have the file "Data_change-point.txt"

### Figure 1 ###

# The symmetric PMF to decide which model to try next
g <- function(k){


  u <- runif(1)

  if(u <= 1 / 2){return(k + 1)}
  else{

    return(k - 1)

  }

}

# The ideal RJ for sampling K only
ideal_RJ <- function(ini_value, nb_iter, vect_prob, K_max){

  # We create a vector to store the states of the chain
  vect_states <- matrix(ncol = 1, nrow = nb_iter + 1)
  vect_states[1] <- ini_value

  for(i in 2:(nb_iter + 1)){

    # We propose a value for the next state
    proposal <- g(vect_states[i - 1])

    # We compute the acceptance probability
    if(proposal <= K_max && proposal >= 1){

      acc_prob <- vect_prob[proposal] / vect_prob[vect_states[i - 1]]

    }
    else{acc_prob <- 0}

    if(runif(1) <= acc_prob){ # the proposal is accepted

      vect_states[i] <- proposal

    }
    else{ # it is rejected

      vect_states[i] <- vect_states[i - 1]

    }

  }

  return(vect_states)

}

post_k <- cbind(1:10, c(0.0639, 0.3028, 0.2717, 0.2031, 0.0998, 0.0388, 0.0135, 0.0044, 0.0014, 0.0004))
# estimated for k = 1, ..., 10 using trial runs
# outside of these values, the estimates are essentially 0

set.seed(1)
ini_value <- sample(1:nrow(post_k), size = 1, prob = post_k[, 2])
nb_iter <- 250
results_ideal_RJ <- ideal_RJ(ini_value = ini_value, nb_iter = nb_iter, vect_prob = post_k[, 2], K_max = nrow(post_k))

# We plot the trace
par(mar = c(4, 5, 1, 3))
plot(1:250, results_ideal_RJ[1:250], type = "l", cex.lab = 1.5, lty = 1, cex.axis = 1.5, cex = 1.5, lwd = 3, xlab = "Iteration", ylab = "k", xlim = c(0, 300), ylim = c(0.5, 10))
# We add the PMF
# the mode is
mode_k <- which.max(post_k[, 2])
for(i in 1:(nrow(post_k))){

  # we compute the length of the line
  length_line <- 40 * post_k[i, 2] / post_k[mode_k, 2]
  lines(c(300 - length_line, 300), rep(i, 2), lwd = 6, col = "darkgrey")

}

# The ideal NRJ for sampling K only
ideal_NRJ <- function(ini_values, nb_iter, vect_prob, K_max){

  # We create a vector to store the states of the chain, that has now an extended state space
  vect_states <- matrix(ncol = 2, nrow = nb_iter + 1)
  vect_states[1, ] <- ini_values

  for(i in 2:(nb_iter + 1)){

    # We propose a value for the next state
    proposal <- vect_states[i - 1, 1] + vect_states[i - 1, 2]

    # We compute the acceptance probability
    if(proposal <= K_max && proposal >= 1){

      acc_prob <- vect_prob[proposal] / vect_prob[vect_states[i - 1, 1]]

    }
    else{acc_prob <- 0}

    if(runif(1) <= acc_prob){ # the proposal is accepted

      vect_states[i, 1] <- proposal
      vect_states[i, 2] <- vect_states[i - 1, 2]

    }
    else{ # it is rejected

      vect_states[i, 1] <- vect_states[i - 1, 1]
      vect_states[i, 2] <- -vect_states[i - 1, 2]

    }

  }

  return(vect_states)

}

set.seed(2)
ini_values <- matrix(ncol = 1, nrow = 2)
ini_values[1] <- sample(1:nrow(post_k), size = 1, prob = post_k[, 2])
ini_values[2] <- sample(c(-1, +1), size = 1)
results_NRJ <- ideal_NRJ(ini_values = ini_values, nb_iter = nb_iter, vect_prob = post_k[, 2], K_max = nrow(post_k))

# We plot the trace
par(mar = c(4, 5, 1, 3))
plot(1:250, results_NRJ[1:250, 1], type = "l", cex.lab = 1.5, lty = 1, cex.axis = 1.5, cex = 1.5, lwd = 3, xlab = "Iteration", ylab = "k", xlim = c(0, 300), ylim = c(0.5, 10))
# We add the PMF
# the mode is
mode_k <- which.max(post_k[, 2])
for(i in 1:(nrow(post_k))){

  # we compute the length of the line
  length_line <- 40 * post_k[i, 2] / post_k[mode_k, 2]
  lines(c(300 - length_line, 300), rep(i, 2), lwd = 6, col = "darkgrey")

}

# the ESS are computed at the same time we make the calculations for Section 5.2

# Figure 2 is created once we have the code for the non-ideal NRJ

## Section 5.2 -  performance evaluation in a multiple change-point problem ##

# Data from Jarrett (1979) - A Note on the Intervals Between Coal-Mining Disasters
data_jarrett <- as.matrix(read.delim("Data_change-point.txt", header = FALSE, sep = " "))[, -1]

y <- NULL
for(i in 1:ncol(data_jarrett)){

  y <- c(y, data_jarrett[, i])

}
y <- c(0, cumsum(y))

# We extend them as precised in Raftery (1986) - Bayesian Analysis of a Poisson Process with a Change-Point
# we have to add the number of days between 1 January and 14 March 1851 (inclusively) to y because there is no incident
y <- y + 31 + 28 + 14

# there is also no incident between the last mentioned in Jarrett (1979) on 21 March 1962 and 31 December 1962
L <- y[length(y)] + 10 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31

# prior
log_prior <- function(k, s, h, lambda, K_max, alpha, beta, L){

  log_prior_k <- dpois(k, lambda, log = TRUE) - ppois(K_max, lambda, lower.tail = TRUE, log.p = TRUE)
  log_prior_h_given_k <- sum(dgamma(h, shape = alpha, rate = beta, log = TRUE))

  if(length(which(s[2:length(s)] - s[1:(length(s) - 1)] <= 0)) == 0){

    s_diff <- s[2:(k + 2)] - s[1:(k + 1)]
    log_prior_s_given_k <- lchoose(2 * k + 1, k) + lfactorial(k) + lfactorial(k + 1) - (2 * k + 1) * log(L) + sum(log(s_diff))
    return(log_prior_k + log_prior_s_given_k + log_prior_h_given_k)

  }
  if(exists("s_diff") == FALSE){return(-Inf)}

}

# likelihood
logL <- function(s, h, k, L, y){

  # # We return -Inf if one of the h is too small
  # if(length(which(h < 1e-07)) == 0){

    if(k  == 0){ # there is 0 step

      part1 <- length(y) * log(h)
      part2 <- L * h
      return(part1 - part2)

    }
    if(k > 0 && length(which(s[2:length(s)] - s[1:(length(s) - 1)] <= 0)) == 0){ # there is at least one step

      # for each step, we count the number of observations within s_{j-1, k} and s_{j, k}
      count <- matrix(ncol = 1, nrow = k + 1)
      for(i in 1:(k + 1)){

          count[i] <- length(which(y > s[i] & y <= s[i + 1]))

      }

      part1 <- sum(count * log(h))
      s_diff <- s[2:(k + 2)] - s[1:(k + 1)]
      part2 <- sum(s_diff * h)
      return(part1 - part2)

    }
    if(exists("s_diff") == FALSE){return(-Inf)}

  # }
  # else{return(-Inf)}

}

# what is required for the algorithms

# The prior on K
p_k <- function(k, K_max){

  if(k <= K_max){

    return(dpois(k, lambda, log = FALSE) / ppois(K_max, lambda, lower.tail = TRUE, log.p = FALSE))

  }
  else{return(0)}

}

# The PMF to decide which model is next explored in RJ
g_RJ <- function(k, tau){

  u <- runif(1)

  if(u <= (1 - tau) / 2){return(k + 1)}
  else{

    if(u <= 1 - tau){return(k - 1)}
    else{return(k)}

  }

}

# The PMF to decide which model is next explored in NRJ
g_NRJ <- function(tau){

  u <- runif(1)

  if(u <= tau){return(1)} # for parameter updates
  else{

    return(2) # for model switches

  }

}

# function to update one of the heights
update_height <- function(k, current_h){

  # randomly choose which h_{j, k} to update
  j <- sample(1:(k + 1), size = 1)
  prop_h <- current_h
  prop_h[j] <- exp(log(current_h[j]) + runif(n = 1, min = -0.5, max = 0.5))

  return(list(prop_h = prop_h, j = j))

}

# function to update one of the starting points
update_starting <- function(k, current_s){

  #  randomly choose which s_{j, k} to update
  j <- sample(1:k, size = 1)
  prop_s <- current_s
  prop_s[j + 1] <- runif(n = 1, min = current_s[j], max = current_s[j + 2])

  return(list(prop_s = prop_s, j = j))

}

# Vanilla RJ, implemented as in Green (1995)
RJ <- function(nb_iter, ini_val_K, ini_val_S, ini_val_H, y, lambda, K_max, alpha, beta, L, tau, B){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # B: burn-in

  # We record the states of the chain 
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  
  current_h <- ini_val_H
  current_s <- c(0, ini_val_S, L)
  
  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)

  # we count the number of attempts for each movement type to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 3, 0)
  success <- matrix(ncol = 1, nrow = 3, 0)

  for(i in 2:(nb_iter + 1)){

    # What is the current model 
    k <- matrix_states_K[i - 1]

    # What is the proposal for the next model to explore?
    prop_k <- g_RJ(k, tau)

    if(prop_k == k){ # it is proposed to make an attempt at updating the parameters

      counts[1] <- counts[1] + 1
      vect_move_type[i] <- 1

      # randomly choose between a height or position change
      type <- sample(c("H", "P"), size = 1)
      if(type == "H"){

        # we update one of the heights
        results_h <- update_height(k, current_h)
        prop_h <- results_h$prop_h
        j <- results_h$j

        # we compute the (log) acceptance probability
        acc_prob <- logL(current_s, prop_h, k, L, y) - logL(current_s, current_h, k, L, y) - beta * (prop_h[j] - current_h[j])

        if(log(runif(1)) <= acc_prob){

          current_h <- prop_h
          success[1] <- success[1] + 1

        }

        matrix_states_K[i] <- k

      }
      else{ # Position change

        # we update one of the starting points
        results_s <- update_starting(k, current_s)
        prop_s <- results_s$prop_s
        j <- results_s$j

        # we compute the (log) acceptance probability
        acc_prob <- logL(prop_s, current_h, k, L, y) - logL(current_s, current_h, k, L, y) + log(prop_s[j + 2] - prop_s[j + 1]) + log(prop_s[j + 1] - prop_s[j]) - log(prop_s[j + 2] - current_s[j + 1]) - log(current_s[j + 1] - prop_s[j])

        if(log(runif(1)) <= acc_prob){

          current_s <- prop_s
          success[1] <- success[1] + 1

        }

        matrix_states_K[i] <- k

      }

    }
    if(prop_k == k + 1){ # it is proposed to add a step

      counts[2] <- counts[2] + 1
      vect_move_type[i] <- 2
      
      if(k < K_max){

        # we generate where we want to add the step
        s_star <- runif(n = 1, min = 0, max = L)

        # we identify where it will be added and define the vectors prop_s and prop_h
        j <- max(which(current_s < s_star)) - 1
        prop_s <- c(current_s[1:(j + 1)], s_star, current_s[(j + 2):(k + 2)])

        # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
        s_j <- current_s[j + 1]
        s_j_1 <- current_s[j + 2]

        if(j == 0){ # it will be the first step

          prop_h <- c(NA, NA, current_h[-1])

        }
        else{

          if(j == k){ # it will be the last step

            prop_h <- c(current_h[-(k + 1)], NA, NA)

          }
          else{ # it will be somewhere in the middle

            prop_h <- c(current_h[1:j], NA, NA, current_h[(j + 2):(k + 1)])

          }

        }

        # we compute how to perturb the step height to obtain the height on the left, and that on the right
        u <- runif(1)
        x <- (1 - u) / u
        prop_h[j + 1] <- current_h[j + 1] / x^((s_j_1 - s_star) / (s_j_1 - s_j))
        prop_h[j + 2] <- prop_h[j + 1] * x

        # we compute the (log) acceptance probability
        acc_prob1 <- logL(prop_s, prop_h, k + 1, L, y) - logL(current_s, current_h, k, L, y)
        acc_prob2 <- log(p_k(k + 1, K_max) / p_k(k, K_max)) + log(2 * (k + 1) * (2 * k + 3)) - 2 * log(L) + log(s_star - s_j) + log(s_j_1 - s_star) - log(s_j_1 - s_j) + log(beta) - beta * (prop_h[j + 1] + prop_h[j + 2] - current_h[j + 1])
        acc_prob3 <- log(L) - log(k + 1)
        acc_prob4 <- 2 * log(prop_h[j + 1] + prop_h[j + 2]) - log(current_h[j + 1])

        acc_prob <- acc_prob1 + acc_prob2 + acc_prob3 + acc_prob4

        if(log(runif(1)) <= acc_prob){

          matrix_states_K[i] <- k + 1
          current_s <- prop_s
          current_h <- prop_h
          success[2] <- success[2] + 1

        }
        else{

          matrix_states_K[i] <- k

        }
      
      }
      else{
        
        matrix_states_K[i] <- k
        
      }

    }
    if(prop_k == k - 1){ # it is proposed to delete a step

      counts[3] <- counts[3] + 1
      vect_move_type[i] <- 2
      
      if(k > 0){

        #  randomly choose which s_{j, k} to delete
        j <- sample(1:k, size = 1)

        prop_s <- current_s[-(j + 1)]
        s_j_1 <- current_s[j + 1]
        s_j <- current_s[j]
        s_j_2 <- current_s[j + 2]

        if(j == 1){

          prop_h <- c(NA, current_h[-c(1, 2)])

        }
        else{

          if(j == k){

            prop_h <- c(current_h[-c(k, k + 1)], NA)

          }
          else{

            prop_h <- c(current_h[1:(j - 1)], NA, current_h[(j + 2):(k + 1)])

          }

        }

        prop_h[j] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(current_h[j]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(current_h[j + 1]))

        # we compute the (log) acceptance probability
        acc_prob1 <- logL(prop_s, prop_h, k - 1, L, y) - logL(current_s, current_h, k, L, y)
        acc_prob2 <- log(p_k(k - 1, K_max) / p_k(k, K_max)) - log(2 * k * (2 * k + 1)) + 2 * log(L) - log(s_j_1 - s_j) - log(s_j_2 - s_j_1) + log(s_j_2 - s_j) - log(beta) + beta * (current_h[j] + current_h[j + 1] - prop_h[j])
        acc_prob3 <- -log(L) + log(k)
        acc_prob4 <- -2 * log(current_h[j] + current_h[j + 1]) + log(prop_h[j])

        acc_prob <- acc_prob1 + acc_prob2 + acc_prob3 + acc_prob4

        if(log(runif(1)) <= acc_prob){

          matrix_states_K[i] <- k - 1
          current_s <- prop_s
          current_h <- prop_h
          success[3] <- success[3] + 1

        }
        else{

          matrix_states_K[i] <- k

        }
      
      }
      else{
        
        matrix_states_K[i] <- k
        
      }

    }

  }

  return(list(matrix_states_K = matrix_states_K[(B + 1):(nb_iter + 1)], acc_rate = success / counts, vect_move_type = vect_move_type[(B + 1):(nb_iter + 1)]))

}

# Algorithm 1, with the same proposal mechanism as the RJ function
Algorithm1 <- function(nb_iter, ini_val_K, ini_val_S, ini_val_H, ini_val_v, y, lambda, K_max, alpha, beta, L, tau, B){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # B: burn-in

  # We record the states of the chain
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  
  direction <- ini_val_v
  current_h <- ini_val_H
  current_s <- c(0, ini_val_S, L)

  # we count the number of attempts for each movement type to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 3, 0)
  success <- matrix(ncol = 1, nrow = 3, 0)

  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)

  for(i in 2:(nb_iter + 1)){

    # What is the current model 
    k <- matrix_states_K[i - 1]

    # What is the type of move attempted
    type <- g_NRJ(tau)

    if(type == 1){ # it is proposed to make an attempt at updating the parameters

      counts[1] <- counts[1]  + 1
      vect_move_type[i] <- 1

      # randomly choose between a height or position change
      type <- sample(c("H", "P"), size = 1)
      if(type == "H"){

        # we update one of the heights
        results_h <- update_height(k, current_h)
        prop_h <- results_h$prop_h
        j <- results_h$j

        # we compute the (log) acceptance probability
        acc_prob <- logL(current_s, prop_h, k, L, y) - logL(current_s, current_h, k, L, y) - beta * (prop_h[j] - current_h[j])

        if(log(runif(1)) <= acc_prob){

          current_h <- prop_h
          success[1] <- success[1] + 1

        }

        matrix_states_K[i] <- k

      }
      else{ # Position change

        # we update one of the starting points
        results_s <- update_starting(k, current_s)
        prop_s <- results_s$prop_s
        j <- results_s$j

        # we compute the (log) acceptance probability
        acc_prob <- logL(prop_s, current_h, k, L, y) - logL(current_s, current_h, k, L, y) + log(prop_s[j + 2] - prop_s[j + 1]) + log(prop_s[j + 1] - prop_s[j]) - log(prop_s[j + 2] - current_s[j + 1]) - log(current_s[j + 1] - prop_s[j])

        if(log(runif(1)) <= acc_prob){

          current_s <- prop_s
          success[1] <- success[1] + 1

        }

        matrix_states_K[i] <- k

      }

    }
    if(type == 2 && direction == 1){ # it is proposed to add a step
      
      counts[2] <- counts[2] + 1
      vect_move_type[i] <- 2

      if(k < K_max){

        # we generate where we want to add the step
        s_star <- runif(n = 1, min = 0, max = L)

        # we identify where it will be added and define the vectors prop_s and prop_h
        j <- max(which(current_s < s_star)) - 1
        prop_s <- c(current_s[1:(j + 1)], s_star, current_s[(j + 2):(k + 2)])

        # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
        s_j <- current_s[j + 1]
        s_j_1 <- current_s[j + 2]

        if(j == 0){ # it will be the first step

          prop_h <- c(NA, NA, current_h[-1])

        }
        else{

          if(j == k){ # it will be the last step

            prop_h <- c(current_h[-(k + 1)], NA, NA)

          }
          else{ # it will be somewhere in the middle

            prop_h <- c(current_h[1:j], NA, NA, current_h[(j + 2):(k + 1)])

          }

        }

        # we compute how to perturb the step height to obtain the height on the left, and that on the right
        u <- runif(1)
        x <- (1 - u) / u
        prop_h[j + 1] <- current_h[j + 1] / x^((s_j_1 - s_star) / (s_j_1 - s_j))
        prop_h[j + 2] <- prop_h[j + 1] * x

        # we compute the (log) acceptance probability
        acc_prob1 <- logL(prop_s, prop_h, k + 1, L, y) - logL(current_s, current_h, k, L, y)
        acc_prob2 <- log(p_k(k + 1, K_max) / p_k(k, K_max)) + log(2 * (k + 1) * (2 * k + 3)) - 2 * log(L) + log(s_star - s_j) + log(s_j_1 - s_star) - log(s_j_1 - s_j) + log(beta) - beta * (prop_h[j + 1] + prop_h[j + 2] - current_h[j + 1])
        acc_prob3 <- log(L) - log(k + 1)
        acc_prob4 <- 2 * log(prop_h[j + 1] + prop_h[j + 2]) - log(current_h[j + 1])

        acc_prob <- acc_prob1 + acc_prob2 + acc_prob3 + acc_prob4

        if(log(runif(1)) <= acc_prob){

          matrix_states_K[i] <- k + 1
          current_s <- prop_s
          current_h <- prop_h
          success[2] <- success[2] + 1

        }
        else{

          matrix_states_K[i] <- k
          direction <- -direction

        }

      }
      else{

        matrix_states_K[i] <- k
        direction <- -direction

      }

    }
    if(type == 2 && direction == -1){ # it is proposed to delete a step

      counts[3] <- counts[3] + 1
      vect_move_type[i] <- 2
      
      if(k > 0){

        #  randomly choose which s_{j, k} to delete
        j <- sample(1:k, size = 1)

        prop_s <- current_s[-(j + 1)]
        s_j_1 <- current_s[j + 1]
        s_j <- current_s[j]
        s_j_2 <- current_s[j + 2]

        if(j == 1){

          prop_h <- c(NA, current_h[-c(1, 2)])

        }
        else{

          if(j == k){

            prop_h <- c(current_h[-c(k, k + 1)], NA)

          }
          else{

            prop_h <- c(current_h[1:(j - 1)], NA, current_h[(j + 2):(k + 1)])

          }

        }

        prop_h[j] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(current_h[j]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(current_h[j + 1]))

        # we compute the (log) acceptance probability
        acc_prob1 <- logL(prop_s, prop_h, k - 1, L, y) - logL(current_s, current_h, k, L, y)
        acc_prob2 <- log(p_k(k - 1, K_max) / p_k(k, K_max)) - log(2 * k * (2 * k + 1)) + 2 * log(L) - log(s_j_1 - s_j) - log(s_j_2 - s_j_1) + log(s_j_2 - s_j) - log(beta) + beta * (current_h[j] + current_h[j + 1] - prop_h[j])
        acc_prob3 <- - log(L) + log(k)
        acc_prob4 <- -2 * log(current_h[j] + current_h[j + 1]) + log(prop_h[j])

        acc_prob <- acc_prob1 + acc_prob2 + acc_prob3 + acc_prob4

        if(log(runif(1)) <= acc_prob){

          matrix_states_K[i] <- k - 1
          current_s <- prop_s
          current_h <- prop_h
          success[3] <- success[3] + 1

        }
        else{

          matrix_states_K[i] <- k
          direction <- -direction

        }
      
      }
      else{
        
        matrix_states_K[i] <- k
        direction <- -direction
        
      }

    }

  }

  return(list(matrix_states_K = matrix_states_K[(B + 1):(nb_iter + 1)], acc_rate = success / counts, vect_move_type = vect_move_type[(B + 1):(nb_iter + 1)]))

}

### Figure 2 ###

set.seed(2)
ini_values <- matrix(ncol = 1, nrow = 2)
ini_values[1] <- sample(1:nrow(post_k), size = 1, prob = post_k[, 2])
ini_values[2] <- sample(c(-1, +1), size = 1)
results_NRJ <- ideal_NRJ(ini_values = ini_values, nb_iter = nb_iter, vect_prob = post_k[, 2], K_max = nrow(post_k))

# We plot the trace
par(mar = c(4, 5, 1, 3))
plot(0:10, results_NRJ[75:85, 1], type = "l", cex.lab = 1.5, lty = 1, cex.axis = 1.5, cex = 1.5, lwd = 3, xlab = "Iteration", ylab = "k", xlim = c(0, 12), ylim = c(0.5, 10.5), col = "lightgrey")
# We add the PMF
# the mode is
mode_k <- which.max(post_k[, 2])
for(i in 1:(nrow(post_k))){

  # we compute the length of the other lines
  length_line <- 1.6 * post_k[i, 2] / post_k[mode_k, 2]
  lines(c(12 - length_line, 12), rep(i, 2), lwd = 6, col = "darkgrey")

}
arrows(0, 10, 0, 8, length = 0.05, lwd = 3)
points(0, 10, pch = 19, cex = 1)
text(1.25, 11, "(10, -1)", pos = 1, cex = 0.9)

set.seed(7) 
nb_iter <- 100000
K_max <- 10
tau <- 0.7
B <- 0
lambda <- 3
alpha <- 1
beta <- 200
# the initial values are values in the high density area, found through trial runs
ini_val_K <- 2
ini_val_S <- c(14313, 35314)
ini_val_H <- c(0.008, 0.002, 0.001)
ini_val_v <- 1
results <- Algorithm1(nb_iter, ini_val_K, ini_val_S, ini_val_H, ini_val_v, y, lambda, K_max, alpha, beta, L, tau, B)
results_K <- results$matrix_states_K[which(results$vect_move_type == 2)]

# We plot the trace
lines(0:10, results_K[19139:19149], lwd = 3)
legend("bottomleft", c("Noisy NRJ", "Ideal NRJ"), lty = 1, lwd = 3, col = c("black", "lightgrey"))

# RJ & Algorithm 3, with the same proposal mechanism as Karagiannis and Andrieu (2013), and incorporating
# the method of Andrieu et al. (2018)

# we beforehand introduce a function to evaluate (log) \rho^{(t)}
rho_t <- function(k, j_star, h, s, h_small, s_small, t, T, L, y, lambda, K_max, alpha, beta){

  # we compute the log posterior under the larger model (plus the other term  under the power t/T)
  term1 <- (t / T) * (logL(s, h, k, L, y) + log_prior(k, s, h, lambda, K_max, alpha, beta, L) - log(k))

  # we compute the log posterior under the smaller model
  term2 <- (1 - t / T) * (logL(s_small, h_small, k - 1, L, y) + log_prior(k - 1, s_small, h_small, lambda, K_max, alpha, beta, L))

  # we compute the other term under the power (T - t) / T
  term3 <- (1 - t / T) * (-log(L) - 2 * log(h[j_star + 1] + h[j_star + 2]) + log(h_small[j_star + 1]))

  return(term1 + term2 + term3)

}

# we define functions that will be used to generate paths
forward_path <- function(k, prop_h, prop_s, current_h, current_s, j, T, L, y, lambda, K_max, alpha, beta){

  # we define five matrices that will contain the path
  path_h <- matrix(nrow = T, ncol = k + 2)
  path_s <- matrix(nrow = T, ncol = k + 3)
  path_h_small <- matrix(nrow = T, ncol = k + 1)
  path_s_small <- matrix(nrow = T, ncol = k + 2)
  path_j <- matrix(nrow = T, ncol = 1)

  path_h[1, ] <- prop_h
  path_s[1, ] <- prop_s
  path_h_small[1, ] <- current_h
  path_s_small[1, ] <- current_s
  path_j[1] <- j

  # we define one matrix to store the (log) acceptance probabilities
  mat_acc_prob <- matrix(ncol = 2, nrow = T)
  mat_acc_prob[1, 1] <- rho_t(k + 1, path_j[1], path_h[1, ], path_s[1, ], path_h_small[1, ], path_s_small[1, ], 0, T, L, y, lambda, K_max, alpha, beta)
  mat_acc_prob[1, 2] <- rho_t(k + 1, path_j[1], path_h[1, ], path_s[1, ], path_h_small[1, ], path_s_small[1, ], 1, T, L, y, lambda, K_max, alpha, beta)

  for(t in 2:T){

    # we generate the random order in which we do the updates
    order <- sample(1:3, size = 3, replace = FALSE)

    # to have the updated values
    path_h[t, ] <- path_h[t - 1, ]
    path_s[t, ] <- path_s[t - 1, ]
    path_h_small[t, ] <- path_h_small[t - 1, ]
    path_s_small[t, ] <- path_s_small[t - 1, ]
    path_j[t] <- path_j[t - 1]

    for(ord in order){

      if(ord == 1){ # we update one of the heights

        results_h <- update_height(k + 1, path_h[t, ])
        prop_h <- results_h$prop_h
        prop_h_small <- path_h_small[t, ]
        j <- results_h$j

        # either we try to update one height that, combined with the previous one or the next one, corresponds to one height in the small model
        # or not
        if(j == path_j[t] + 1 || j == path_j[t] + 2){

          s_j <- path_s[t, path_j[t] + 1]
          s_j_1 <- path_s[t, path_j[t] + 2]
          s_j_2 <- path_s[t, path_j[t] + 3]
          prop_h_small[path_j[t] + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(prop_h[path_j[t] + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(prop_h[path_j[t] + 2]))

        }
        else{

          if(j < path_j[t] + 1){prop_h_small[j] <- prop_h[j]}
          else{prop_h_small[j - 1] <- prop_h[j]}

        }

        # we compute the (log) acceptance probability
        acc_prob <- rho_t(k + 1, path_j[t], prop_h, path_s[t, ], prop_h_small, path_s_small[t, ], t - 1, T, L, y, lambda, K_max, alpha, beta) -
          rho_t(k + 1, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t - 1, T, L, y, lambda, K_max, alpha, beta)

        if(log(runif(1)) <= acc_prob){

          path_h[t, ] <- prop_h
          path_h_small[t, ] <- prop_h_small

        }


      }
      if(ord == 2){ # we update one of the starting points

        results_s <- update_starting(k + 1, path_s[t, ])
        prop_s <- results_s$prop_s
        prop_s_small <- path_s_small[t, ]
        j <- results_s$j

        # either we try to update one starting point that is also in the smaller model, or not
        if(j < path_j[t] + 1){

          prop_s_small[j + 1] <- prop_s[j + 1]

        }
        if(j > path_j[t] + 1){

          prop_s_small[j] <- prop_s[j + 1]

        }

        # we compute the (log) acceptance probability
        acc_prob <- rho_t(k + 1, path_j[t], path_h[t, ], prop_s, path_h_small[t, ], prop_s_small, t - 1, T, L, y, lambda, K_max, alpha, beta) -
          rho_t(k + 1, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t - 1, T, L, y, lambda, K_max, alpha, beta)

        if(log(runif(1)) <= acc_prob){

          path_s[t, ] <- prop_s
          path_s_small[t, ] <- prop_s_small

        }

      }
      if(ord == 3){ # we update j^*

        # we have to compute its distribution to sample from the latter
        probs <- NULL
        for(j in 0:k){

          s_small <- path_s[t, -(j + 2)]
          s_j <- path_s[t, (j + 1)]
          s_j_1 <- path_s[t, (j + 2)]
          s_j_2 <- path_s[t, (j + 3)]

          if(j == 0){

            h_small <- c(NA, path_h[t, -c(1, 2)])

          }
          else{

            if(j == k){

              h_small <- c(path_h[t, -c(k + 1, k + 2)], NA)

            }
            else{

              h_small <- c(path_h[t, 1:j], NA, path_h[t, (j + 3):(k + 2)])

            }

          }

          h_small[j + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(path_h[t, j + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(path_h[t, j + 2]))

          probs <- c(probs, (1 - (t - 1) / T) * (logL(s_small, h_small, k, L, y) + log_prior(k, s_small, h_small, lambda, K_max, alpha, beta, L) +
                                                   2 * log(path_h[t, j + 1] + path_h[t, j + 2]) - log(h_small[j + 1])))

        }

        # we multiply by a constant to avoid numerical errors
        cst <- -max(probs)
        probs <- probs + cst
        probs <- exp(probs)

        path_j[t] <- sample(0:k, size = 1, prob = probs / sum(probs))

        if(path_j[t] != path_j[t - 1]){

          path_s_small[t, ] <- path_s[t, -(path_j[t] + 2)]
          s_j <- path_s[t, (path_j[t] + 1)]
          s_j_1 <- path_s[t, (path_j[t] + 2)]
          s_j_2 <- path_s[t, (path_j[t] + 3)]

          if(path_j[t] == 0){

            path_h_small[t, ] <- c(NA, path_h[t, -c(1, 2)])

          }
          else{

            if(path_j[t] == k){

              path_h_small[t, ] <- c(path_h[t, -c(k + 1, k + 2)], NA)

            }
            else{

              path_h_small[t, ] <- c(path_h[t, 1:path_j[t]], NA, path_h[t, (path_j[t] + 3):(k + 2)])

            }

          }

          path_h_small[t, path_j[t] + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(path_h[t, path_j[t] + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(path_h[t, path_j[t] + 2]))

        }

      }

    }

    mat_acc_prob[t, 1] <- rho_t(k + 1, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t - 1, T, L, y, lambda, K_max, alpha, beta)
    mat_acc_prob[t, 2] <- rho_t(k + 1, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t, T, L, y, lambda, K_max, alpha, beta)

  }

  return(list(path_s = path_s, path_h = path_h, path_s_small = path_s_small, path_h_small = path_h_small, path_j = path_j, mat_acc_prob = mat_acc_prob))

}

backward_path <- function(k, prop_h, prop_s, current_h, current_s, j, T, L, y, lambda, K_max, alpha, beta){

  # we define five matrices that will contain the path
  path_h <- matrix(nrow = T, ncol = k + 1)
  path_s <- matrix(nrow = T, ncol = k + 2)
  path_h_small <- matrix(nrow = T, ncol = k)
  path_s_small <- matrix(nrow = T, ncol = k + 1)
  path_j <- matrix(nrow = T, ncol = 1)

  path_h[T, ] <- current_h
  path_s[T, ] <- current_s
  path_h_small[T, ] <- prop_h
  path_s_small[T, ] <- prop_s
  path_j[T] <- j

  # we define one matrix to store the (log) acceptance probabilities
  mat_acc_prob <- matrix(ncol = 2, nrow = T)
  mat_acc_prob[T, 1] <- rho_t(k, path_j[T], path_h[T, ], path_s[T, ], path_h_small[T, ], path_s_small[T, ], T, T, L, y, lambda, K_max, alpha, beta)
  mat_acc_prob[T, 2] <- rho_t(k, path_j[T], path_h[T, ], path_s[T, ], path_h_small[T, ], path_s_small[T, ], T - 1, T, L, y, lambda, K_max, alpha, beta)

  for(t in (T - 1):1){

    # we generate the random order in which we do the updates
    order <- sample(1:3, size = 3, replace = FALSE)

    # to have the updated values
    path_h[t, ] <- path_h[t + 1, ]
    path_s[t, ] <- path_s[t + 1, ]
    path_h_small[t, ] <- path_h_small[t + 1, ]
    path_s_small[t, ] <- path_s_small[t + 1, ]
    path_j[t] <- path_j[t + 1]

    for(ord in order){

      if(ord == 1){ # we update one of the heights

        results_h <- update_height(k, path_h[t, ])
        prop_h <- results_h$prop_h
        prop_h_small <- path_h_small[t, ]
        j <- results_h$j

        # either we try to update one height that, combined with the previous one or the next one, corresponds to one height in the small model
        # or not
        if(j == path_j[t] + 1 || j == path_j[t] + 2){

          s_j <- path_s[t, path_j[t] + 1]
          s_j_1 <- path_s[t, path_j[t] + 2]
          s_j_2 <- path_s[t, path_j[t] + 3]
          prop_h_small[path_j[t] + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(prop_h[path_j[t] + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(prop_h[path_j[t] + 2]))

        }
        else{

          if(j < path_j[t] + 1){prop_h_small[j] <- prop_h[j]}
          else{prop_h_small[j - 1] <- prop_h[j]}

        }

        # we compute the (log) acceptance probability
        acc_prob <- rho_t(k, path_j[t], prop_h, path_s[t, ], prop_h_small, path_s_small[t, ], t, T, L, y, lambda, K_max, alpha, beta) -
          rho_t(k, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t, T, L, y, lambda, K_max, alpha, beta)

        if(log(runif(1)) <= acc_prob){

          path_h[t, ] <- prop_h
          path_h_small[t, ] <- prop_h_small

        }


      }
      if(ord == 2){ # we update one of the starting points

        results_s <- update_starting(k, path_s[t, ])
        prop_s <- results_s$prop_s
        prop_s_small <- path_s_small[t, ]
        j <- results_s$j

        # either we try to update one starting point that is also in the smaller model, or not
        if(j < path_j[t] + 1){

          prop_s_small[j + 1] <- prop_s[j + 1]

        }
        if(j > path_j[t] + 1){

          prop_s_small[j] <- prop_s[j + 1]

        }

        # we compute the (log) acceptance probability
        acc_prob <- rho_t(k, path_j[t], path_h[t, ], prop_s, path_h_small[t, ], prop_s_small, t, T, L, y, lambda, K_max, alpha, beta) -
          rho_t(k, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t, T, L, y, lambda, K_max, alpha, beta)

        if(log(runif(1)) <= acc_prob){

          path_s[t, ] <- prop_s
          path_s_small[t, ] <- prop_s_small

        }

      }
      if(ord == 3){ # we update j^*

        # we have to compute its distribution to sample from the latter
        probs <- NULL

        for(j in 0:(k - 1)){

          s_small <- path_s[t, -(j + 2)]
          s_j <- path_s[t, (j + 1)]
          s_j_1 <- path_s[t, (j + 2)]
          s_j_2 <- path_s[t, (j + 3)]

          if(j == 0){

            h_small <- c(NA, path_h[t, -c(1, 2)])

          }
          else{

            if(j == k - 1){

              h_small <- c(path_h[t, -c(k, k + 1)], NA)

            }
            else{

              h_small <- c(path_h[t, 1:j], NA, path_h[t, (j + 3):(k + 1)])

            }

          }

          h_small[j + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(path_h[t, j + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(path_h[t, j + 2]))

          probs <- c(probs, (1 - t / T) * (logL(s_small, h_small, k - 1, L, y) + log_prior(k - 1, s_small, h_small, lambda, K_max, alpha, beta, L) +
                                             2 * log(path_h[t, j + 1] + path_h[t, j + 2]) - log(h_small[j + 1])))

        }

        # we multiply by a constant to avoid numerical errors
        cst <- -max(probs)
        probs <- probs + cst
        probs <- exp(probs)

        path_j[t] <- sample(0:(k - 1), size = 1, prob = probs / sum(probs))

        if(path_j[t] != path_j[t + 1]){

          path_s_small[t, ] <- path_s[t, -(path_j[t] + 2)]
          s_j <- path_s[t, (path_j[t] + 1)]
          s_j_1 <- path_s[t, (path_j[t] + 2)]
          s_j_2 <- path_s[t, (path_j[t] + 3)]

          if(path_j[t] == 0){

            path_h_small[t, ] <- c(NA, path_h[t, -c(1, 2)])

          }
          else{

            if(path_j[t] == k - 1){

              path_h_small[t, ] <- c(path_h[t, -c(k, k + 1)], NA)

            }
            else{

              path_h_small[t, ] <- c(path_h[t, 1:path_j[t]], NA, path_h[t, (path_j[t] + 3):(k + 1)])

            }

          }

          path_h_small[t, path_j[t] + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(path_h[t, path_j[t] + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(path_h[t, path_j[t] + 2]))

        }

      }

    }

    mat_acc_prob[t, 1] <- rho_t(k, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t, T, L, y, lambda, K_max, alpha, beta)
    mat_acc_prob[t, 2] <- rho_t(k, path_j[t], path_h[t, ], path_s[t, ], path_h_small[t, ], path_s_small[t, ], t - 1, T, L, y, lambda, K_max, alpha, beta)

  }

  return(list(path_s = path_s, path_h = path_h, path_s_small = path_s_small, path_h_small = path_h_small, path_j = path_j, mat_acc_prob = mat_acc_prob))

}

gen_prop_birth <- function(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max){
  
  # we generate where we want to add the step
  s_star <- runif(n = 1, min = 0, max = L)
  
  # we identify where it will be added and define the vectors prop_s and prop_h
  j <- max(which(current_s < s_star)) - 1
  prop_s <- c(current_s[1:(j + 1)], s_star, current_s[(j + 2):(k + 2)])
  
  # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
  s_j <- current_s[j + 1]
  s_j_1 <- current_s[j + 2]
  
  if(j == 0){ # it will be the first step
    
    prop_h <- c(NA, NA, current_h[-1])
    
  }
  else{
    
    if(j == k){ # it will be the last step
      
      prop_h <- c(current_h[-(k + 1)], NA, NA)
      
    }
    else{ # it will be somewhere in the middle
      
      prop_h <- c(current_h[1:j], NA, NA, current_h[(j + 2):(k + 1)])
      
    }
    
  }
  
  # we compute how to perturb the step height to obtain the height on the left, and that on the right
  u <- runif(1)
  x <- (1 - u) / u
  prop_h[j + 1] <- current_h[j + 1] / x^((s_j_1 - s_star) / (s_j_1 - s_j))
  prop_h[j + 2] <- prop_h[j + 1] * x
  
  # we generate the path
  path <- forward_path(k, prop_h, prop_s, current_h, current_s, j, T, L, y, lambda, K_max, alpha, beta)
  
  # we compute the (log) acceptance probability
  log_acc_prob <- sum(path$mat_acc_prob[, 2] - path$mat_acc_prob[, 1])
  
  return(list(log_acc_prob = log_acc_prob, prop_s = path$path_s[T, ], prop_h = path$path_h[T, ]))
  
}

gen_prop_death <- function(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max){
  
  #  randomly choose which s_{j + 1, k} to delete
  j <- sample(0:(k - 1), size = 1)
  
  prop_s <- current_s[-(j + 2)]
  s_j <- current_s[(j + 1)]
  s_j_1 <- current_s[(j + 2)]
  s_j_2 <- current_s[(j + 3)]
  
  if(j == 0){
    
    prop_h <- c(NA, current_h[-c(1, 2)])
    
  }
  else{
    
    if(j == k - 1){
      
      prop_h <- c(current_h[-c(k, k + 1)], NA)
      
    }
    else{
      
      prop_h <- c(current_h[1:j], NA, current_h[(j + 3):(k + 1)])
      
    }
    
  }
  
  prop_h[j + 1] <- exp(((s_j_1 - s_j) / (s_j_2 - s_j)) * log(current_h[j + 1]) + ((s_j_2 - s_j_1) / (s_j_2 - s_j)) * log(current_h[j + 2]))
  
  # we generate the path
  path <- backward_path(k, prop_h, prop_s, current_h, current_s, j, T, L, y, lambda, K_max, alpha, beta)
  
  # we compute the (log) acceptance probability
  log_acc_prob <- sum(path$mat_acc_prob[, 2] - path$mat_acc_prob[, 1])
  
  return(list(log_acc_prob = log_acc_prob, prop_s = path$path_s_small[1, ], prop_h = path$path_h_small[1, ]))
  
}


# RJ
RJ_v2 <- function(nb_iter, ini_val_K, ini_val_S, ini_val_H, y, lambda, K_max, alpha, beta, L, tau, B, T, N){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # B: burn-in
  
  # Uncomment to run some parts in parallel
  # cl <- makeCluster(N) # the code will be ran on N CPUs
  # # we export required functions to these CPUs
  # clusterExport(cl, c("backward_path", "forward_path", "rho_t", "logL", "log_prior", "update_height", "update_starting"))
  
  # We record the states of the chain in three matrices
  # the first one for K
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  # the second one for S
  matrix_states_S <- matrix(ncol = K_max + 2, nrow = nb_iter + 1)
  matrix_states_S[1, 1:(ini_val_K + 2)] <- c(0, ini_val_S, L)
  # the third one for H
  matrix_states_H <- matrix(ncol = K_max + 1, nrow = nb_iter + 1)
  matrix_states_H[1, 1:(ini_val_K + 1)] <- ini_val_H
  
  # we count the number of attempts for each movement type to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 3, 0)
  success <- matrix(ncol = 1, nrow = 3, 0)
  
  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)
  
  for(i in 2:(nb_iter + 1)){
    
    # What is the current model and parameters, and direction
    k <- matrix_states_K[i - 1]
    current_h <- matrix_states_H[i - 1, 1:(k + 1)]
    current_s <- matrix_states_S[i - 1, 1:(k + 2)]
    
    # What is the proposal for the next model to explore?
    prop_k <- g_RJ(k, tau)
    
    if(prop_k == k){ # it is proposed to make an attempt at updating the parameters
      
      counts[1] <- counts[1]  + 1
      vect_move_type[i] <- 1
      
      # randomly choose between a height or position change
      if(k > 0){type <- sample(c("H", "P"), size = 1)}
      else{type <- "H"}
      if(type == "H"){
        
        # we update one of the heights
        results_h <- update_height(k, current_h)
        prop_h <- results_h$prop_h
        j <- results_h$j
        
        # we compute the (log) acceptance probability
        acc_prob <- logL(current_s, prop_h, k, L, y) - logL(current_s, current_h, k, L, y) - beta * (prop_h[j] - current_h[j])
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_H[i, 1:(k + 1)] <- prop_h
          success[1] <- success[1] + 1
          
        }
        else{
          
          matrix_states_H[i, 1:(k + 1)] <- current_h
          
        }
        
        matrix_states_K[i] <- k
        matrix_states_S[i, 1:(k + 2)] <- current_s
        
      }
      else{ # Position change
        
        # we update one of the starting points
        results_s <- update_starting(k, current_s)
        prop_s <- results_s$prop_s
        j <- results_s$j
        
        # we compute the (log) acceptance probability
        acc_prob <- logL(prop_s, current_h, k, L, y) - logL(current_s, current_h, k, L, y) + log(prop_s[j + 2] - prop_s[j + 1]) + log(prop_s[j + 1] - prop_s[j]) - log(prop_s[j + 2] - current_s[j + 1]) - log(current_s[j + 1] - prop_s[j])
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_S[i, 1:(k + 2)] <- prop_s
          success[1] <- success[1] + 1
          
        }
        else{
          
          matrix_states_S[i, 1:(k + 2)] <- current_s
          
        }
        
        matrix_states_K[i] <- k
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    if(prop_k == k + 1){ # it is proposed to add a step
      
      counts[2] <- counts[2]  + 1
      vect_move_type[i] <- 2
      
      if(k < K_max){
        
        # we decide if we use the equivalent of Step 2.(b-i) or 2.(b-ii) in Algorithm 3
        
        if(runif(1) <= 0.5){ # 2.(b-i)
          
          prop_info <- apply(matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_birth, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_birth, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(unlist(lapply(prop_info, `[[`, 1))) # the lapply selects all the first elements in the list, which are the log-probabilities
          
          if(runif(1) <= mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k + 1
            
            # we generate J^*
            jstar <- sample(1:N, size = 1, prob = vect_acc_prob)
            matrix_states_S[i, 1:(k + 3)] <- prop_info[[jstar]]$prop_s
            matrix_states_H[i, 1:(k + 2)] <- prop_info[[jstar]]$prop_h
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        else{ # 2.(b-ii)
          
          # we generate the forward path
          
          prop_info <- gen_prop_birth(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max)
          
          vect_acc_prob <- -prop_info$log_acc_prob
          
          # from the endpoint, we generate N - 1 reverse paths
          k_end <- k + 1
          s_end <- prop_info$prop_s
          h_end <- prop_info$prop_h
          
          prop_info <- apply(matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_death, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_death, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(c(vect_acc_prob, unlist(lapply(prop_info, `[[`, 1))))
          
          if(runif(1) <= 1 / mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k + 1
            matrix_states_S[i, 1:(k + 3)] <- s_end
            matrix_states_H[i, 1:(k + 2)] <- h_end
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_S[i, 1:(k + 2)] <- current_s
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    if(prop_k == k - 1){ # it is proposed to delete a step
      
      counts[3] <- counts[3]  + 1
      vect_move_type[i] <- 2
      
      if(k > 0){
        
        # we decide if we use the equivalent of Step 2.(b-i) or 2.(b-ii) in Algorithm 3
        
        if(runif(1) <= 0.5){ # 2.(b-i)
          
          prop_info <- apply(matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_death, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_death, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(unlist(lapply(prop_info, `[[`, 1))) # the lapply selects all the first elements in the list, which are the log-probabilities  
          
          if(runif(1) <= mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k - 1
            
            # we generate J^*
            jstar <- sample(1:N, size = 1, prob = vect_acc_prob)
            matrix_states_S[i, 1:(k + 1)] <- prop_info[[jstar]]$prop_s
            matrix_states_H[i, 1:k] <- prop_info[[jstar]]$prop_h
            success[3] <- success[3] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        else{ # 2.(b-ii)
          
          # we generate the forward path
          
          prop_info <- gen_prop_death(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max)
          
          vect_acc_prob <- -prop_info$log_acc_prob
          
          # from the endpoint, we generate N - 1 reverse paths
          k_end <- k - 1
          s_end <- prop_info$prop_s
          h_end <- prop_info$prop_h
          
          prop_info <- apply(matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_birth, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_birth, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(c(vect_acc_prob, unlist(lapply(prop_info, `[[`, 1))))
          
          if(runif(1) <= 1 / mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k - 1
            matrix_states_S[i, 1:(k + 1)] <- s_end
            matrix_states_H[i, 1:k] <- h_end
            success[3] <- success[3] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_S[i, 1:(k + 2)] <- current_s
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    
  }
  
  # Uncomment when using parallel computing
  # stopCluster(cl)
  
  return(list(matrix_states_K = matrix_states_K[(B + 1):(nb_iter + 1)], matrix_states_S = matrix_states_S[(B + 1):(nb_iter + 1), ], matrix_states_H = matrix_states_H[(B + 1):(nb_iter + 1), ], acc_rate = success / counts, vect_move_type = vect_move_type[(B + 1):(nb_iter + 1)]))
  
}

Algorithm3 <- function(nb_iter, ini_val_K, ini_val_S, ini_val_H, ini_val_v, y, lambda, K_max, alpha, beta, L, tau, B, T, N){
  # nb_iter: number of iterations
  # ini_val: initial values for the chain
  # B: burn-in
  
  # Uncomment to run some parts in parallel
  # cl <- makeCluster(N) # the code will be ran on N CPUs
  # # we export required functions to these CPUs
  # clusterExport(cl, c("backward_path", "forward_path", "rho_t", "logL", "log_prior", "update_height", "update_starting"))
  
  # We record the states of the chain in four matrices
  # the first one for K
  matrix_states_K <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_K[1] <- ini_val_K
  # the second one for S
  matrix_states_S <- matrix(ncol = K_max + 2, nrow = nb_iter + 1)
  matrix_states_S[1, 1:(ini_val_K + 2)] <- c(0, ini_val_S, L)
  # the third one for H
  matrix_states_H <- matrix(ncol = K_max + 1, nrow = nb_iter + 1)
  matrix_states_H[1, 1:(ini_val_K + 1)] <- ini_val_H
  # the last one for \nu
  matrix_states_v <- matrix(ncol = 1, nrow = nb_iter + 1)
  matrix_states_v[1] <- ini_val_v
  
  # we count the number of attempts for each movement type to compute the acceptance rates
  counts <- matrix(ncol = 1, nrow = 3, 0)
  success <- matrix(ncol = 1, nrow = 3, 0)
  
  # we store the movement type attempted at each iteration
  vect_move_type <- matrix(ncol = 1, nrow = nb_iter + 1)
  
  for(i in 2:(nb_iter + 1)){
    
    # What is the current model and parameters, and direction
    k <- matrix_states_K[i - 1]
    current_h <- matrix_states_H[i - 1, 1:(k + 1)]
    current_s <- matrix_states_S[i - 1, 1:(k + 2)]
    direction <- matrix_states_v[i - 1]
    
    # What is the type of move attempted
    type <- g_NRJ(tau)
    
    if(type == 1){ # it is proposed to make an attempt at updating the parameters
      
      counts[1] <- counts[1]  + 1
      vect_move_type[i] <- 1
      
      # randomly choose between a height or position change
      if(k > 0){type <- sample(c("H", "P"), size = 1)}
      else{type <- "H"}
      if(type == "H"){
        
        # we update one of the heights
        results_h <- update_height(k, current_h)
        prop_h <- results_h$prop_h
        j <- results_h$j
        
        # we compute the (log) acceptance probability
        acc_prob <- logL(current_s, prop_h, k, L, y) - logL(current_s, current_h, k, L, y) - beta * (prop_h[j] - current_h[j])
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_H[i, 1:(k + 1)] <- prop_h
          success[1] <- success[1] + 1
          
        }
        else{
          
          matrix_states_H[i, 1:(k + 1)] <- current_h
          
        }
        
        matrix_states_K[i] <- k
        matrix_states_v[i] <- direction
        matrix_states_S[i, 1:(k + 2)] <- current_s
        
      }
      else{ # Position change
        
        # we update one of the starting points
        results_s <- update_starting(k, current_s)
        prop_s <- results_s$prop_s
        j <- results_s$j
        
        # we compute the (log) acceptance probability
        acc_prob <- logL(prop_s, current_h, k, L, y) - logL(current_s, current_h, k, L, y) + log(prop_s[j + 2] - prop_s[j + 1]) + log(prop_s[j + 1] - prop_s[j]) - log(prop_s[j + 2] - current_s[j + 1]) - log(current_s[j + 1] - prop_s[j])
        
        if(log(runif(1)) <= acc_prob){
          
          matrix_states_S[i, 1:(k + 2)] <- prop_s
          success[1] <- success[1] + 1
          
        }
        else{
          
          matrix_states_S[i, 1:(k + 2)] <- current_s
          
        }
        
        matrix_states_K[i] <- k
        matrix_states_v[i] <- direction
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    if(type == 2 && direction == 1){ # it is proposed to add a step
      
      counts[2] <- counts[2]  + 1
      vect_move_type[i] <- 2
      
      if(k < K_max){
        
        # we decide if we use the equivalent of Step 2.(b-i) or 2.(b-ii) in Algorithm 3
        
        if(runif(1) <= 0.5){ # 2.(b-i)
          
          prop_info <- apply(matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_birth, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_birth, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(unlist(lapply(prop_info, `[[`, 1))) # the lapply selects all the first elements in the list, which are the log-probabilities
          
          if(runif(1) <= mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k + 1
            matrix_states_v[i] <- direction
            
            # we generate J^*
            jstar <- sample(1:N, size = 1, prob = vect_acc_prob)
            matrix_states_S[i, 1:(k + 3)] <- prop_info[[jstar]]$prop_s
            matrix_states_H[i, 1:(k + 2)] <- prop_info[[jstar]]$prop_h
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_v[i] <- -direction
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        else{ # 2.(b-ii)
          
          # we generate the forward path
          
          prop_info <- gen_prop_birth(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max)
          
          vect_acc_prob <- -prop_info$log_acc_prob
          
          # from the endpoint, we generate N - 1 reverse paths
          k_end <- k + 1
          s_end <- prop_info$prop_s
          h_end <- prop_info$prop_h
          
          prop_info <- apply(matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_death, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_death, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(c(vect_acc_prob, unlist(lapply(prop_info, `[[`, 1))))
          
          if(runif(1) <= 1 / mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k + 1
            matrix_states_v[i] <- direction
            matrix_states_S[i, 1:(k + 3)] <- s_end
            matrix_states_H[i, 1:(k + 2)] <- h_end
            success[2] <- success[2] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_v[i] <- -direction
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_v[i] <- -direction
        matrix_states_S[i, 1:(k + 2)] <- current_s
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    if(type == 2 && direction == -1){ # it is proposed to delete a step
      
      counts[3] <- counts[3]  + 1
      vect_move_type[i] <- 2
      
      if(k > 0){
        
        # we decide if we use the equivalent of Step 2.(b-i) or 2.(b-ii) in Algorithm 3
        
        if(runif(1) <= 0.5){ # 2.(b-i)
          
          prop_info <- apply(matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_death, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N, rep(k, N)), 1, gen_prop_death, current_s = current_s, current_h = current_h, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(unlist(lapply(prop_info, `[[`, 1))) # the lapply selects all the first elements in the list, which are the log-probabilities  
          
          if(runif(1) <= mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k - 1
            matrix_states_v[i] <- direction
            
            # we generate J^*
            jstar <- sample(1:N, size = 1, prob = vect_acc_prob)
            matrix_states_S[i, 1:(k + 1)] <- prop_info[[jstar]]$prop_s
            matrix_states_H[i, 1:k] <- prop_info[[jstar]]$prop_h
            success[3] <- success[3] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_v[i] <- -direction
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        else{ # 2.(b-ii)
          
          # we generate the forward path
          
          prop_info <- gen_prop_death(k, current_s, current_h, y, lambda, alpha, beta, L, T, K_max)
          
          vect_acc_prob <- -prop_info$log_acc_prob
          
          # from the endpoint, we generate N - 1 reverse paths
          k_end <- k - 1
          s_end <- prop_info$prop_s
          h_end <- prop_info$prop_h
          
          prop_info <- apply(matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_birth, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          # use the following line instead of that above to run in parallel
          # prop_info <- parApply(cl, matrix(ncol = 1, nrow = N - 1, rep(k_end, N - 1)), 1, gen_prop_birth, current_s = s_end, current_h = h_end, y = y, lambda = lambda, alpha = alpha, beta = beta, L = L, T = T, K_max = K_max)
          
          vect_acc_prob <- exp(c(vect_acc_prob, unlist(lapply(prop_info, `[[`, 1))))
          
          if(runif(1) <= 1 / mean(vect_acc_prob)){
            
            matrix_states_K[i] <- k - 1
            matrix_states_v[i] <- direction
            matrix_states_S[i, 1:(k + 1)] <- s_end
            matrix_states_H[i, 1:k] <- h_end
            success[3] <- success[3] + 1
            
          }
          else{
            
            matrix_states_K[i] <- k
            matrix_states_v[i] <- -direction
            matrix_states_S[i, 1:(k + 2)] <- current_s
            matrix_states_H[i, 1:(k + 1)] <- current_h
            
          }
          
        }
        
      }
      else{
        
        matrix_states_K[i] <- k
        matrix_states_v[i] <- -direction
        matrix_states_S[i, 1:(k + 2)] <- current_s
        matrix_states_H[i, 1:(k + 1)] <- current_h
        
      }
      
    }
    
  }
  
  # Uncomment when using parallel computing
  # stopCluster(cl)
  
  return(list(matrix_states_K = matrix_states_K[(B + 1):(nb_iter + 1)], matrix_states_S = matrix_states_S[(B + 1):(nb_iter + 1), ], matrix_states_H = matrix_states_H[(B + 1):(nb_iter + 1), ], matrix_states_v = matrix_states_v[(B + 1):(nb_iter + 1)], acc_rate = success / counts, vect_move_type = vect_move_type[(B + 1):(nb_iter + 1)]))
  
}

# We define some functions that are used to compute MAP, to help identify where is the mass, for each model
Neglogpost <- function(param, k, L, y, lambda, K_max, alpha, beta){

  s <- c(0, param[1:k], L)
  h <- param[-(1:k)]
  if(length(which(h <= 0)) == 0){

    return(-logL(s, h, k, L, y) - log_prior(k, s, h, lambda, K_max, alpha, beta, L))
    
  }
  else{Inf}

}

MAP <- function(initial_s, initial_h, k, L, y, lambda, K_max, alpha, beta){

  param <- c(initial_s, initial_h)
  MAP <- optim(param, Neglogpost, gr = NULL, k = k, L = L, y = y, lambda = lambda, K_max = K_max, alpha = alpha, beta = beta, method = "Nelder-Mead", control = list(maxit = 40000, reltol=10^(-12) ))

  return(list(s = MAP$par[1:k], h = MAP$par[-(1:k)], logpost = -MAP$value))

}

# Function to find the MAP for k = 2, ..., 10
find_MAP <- function(ini_s, ini_h, L, y, lambda, K_max, alpha, beta){

  # We create matrices to record the values
  mat_s <- matrix(ncol = K_max, nrow = K_max)
  mat_s[1, 1] <- ini_s
  mat_h <- matrix(ncol = K_max, nrow = K_max + 1) 
  mat_h[1:2, 1] <- ini_h

  for(k in 2:K_max){

    print(k)
    
    # we try 1000 initial values, using those of the previous model
    current_h <- mat_h[1:k, k - 1]
    current_s <- mat_s[1:(k - 1), k - 1]

    maxlogpost <- -Inf

    for(i in 1:1000){

      # we generate where we want to add the step
      s_star <- runif(n = 1, min = 0, max = L)

      # we identify where it will be added and define the vectors prop_s and prop_h
      if(length(which(current_s < s_star)) == 0){ # it will be the first step

        j <- 0
        prop_s <- c(s_star, current_s)
        prop_h <- c(NA, NA, current_h[-1])

        # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
        s_j <- 0
        s_j_1 <- current_s[1]

      }
      else{

        if(length(which(current_s > s_star)) == 0){ # it will be the last step

          j <- k - 1
          prop_s <- c(current_s, s_star)
          prop_h <- c(current_h[-k], NA, NA)

          # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
          s_j <- current_s[k - 1]
          s_j_1 <- L

        }
        else{ # it will be somewhere in the middle

          j <- max(which(current_s < s_star))
          prop_s <- c(current_s[1:j], s_star, current_s[(j + 1):(k - 1)])
          prop_h <- c(current_h[1:j], NA, NA, current_h[(j + 2):k])

          # we define what is s_{j, k} and s_{j + 1, k} because we'll need them later
          s_j <- current_s[j]
          s_j_1 <- current_s[j + 1]

        }

      }

      # we compute how we perturb the step height to obtain the height on the left, and that on the right
      u <- runif(1)
      x <- (1 - u) / u
      prop_h[j + 1] <- current_h[j + 1] / x^((s_j_1 - s_star) / (s_j_1 - s_j))
      prop_h[j + 2] <- prop_h[j + 1] * x

      # we run the minimisation algorithm with these starting values
      results <- MAP(prop_s, prop_h, k, L, y, lambda, K_max, alpha, beta)
      if(results$logpost > maxlogpost){ # we note the values if they beat the best one

        maxlogpost <- results$logpost
        best_s <- results$s
        best_h <- results$h

      }

    }

    mat_s[1:k, k] <- best_s
    mat_h[1:(k + 1), k] <- best_h

  }

  return(list(mat_s = mat_s, mat_h = mat_h))

}

K_max <- 30
lambda <- 3
alpha <- 1
beta <- 200
ini_s <- 14313 # mode for the model with k = 1
ini_h <- c(0.0090, 0.0023)
set.seed(1)
system.time(results <- find_MAP(ini_s, ini_h, L, y, lambda, 10, alpha, beta)) # take about 45 minutes on a regular laptop

# Uncomment if you want to use parallel computing
# require(parallel) # see the code for RJ_v2 and Algorithm3 above for the lines that need to be uncommented 
require(coda)
require(LaplacesDemon)

# details for Algorithm 3 and its reversible counterpart
nb_iter <- 100000
B <- 0.1 * nb_iter
tau <- 0.7
T <- 100
N <- 10
# details for vanilla samplers (see the supplementary material for how we determined the number of iterations and so on)
nb_iter_vanilla <- nb_iter * tau + (1 - tau) * nb_iter * 1.5 * 3 * T
B_vanilla <- 0.1 * nb_iter_vanilla
tau_vanilla <- tau * nb_iter / nb_iter_vanilla

MAP_s <- results$mat_s
MAP_h <- results$mat_h
prob_k <- cbind(1:10, c(0.0639, 0.3028, 0.2717, 0.2031, 0.0998, 0.0388, 0.0135, 0.0044, 0.0014, 0.0004))
# estimated for k = 1, ..., 10 using trial runs, outside the estimates are essentially 0

# little function to compute the posterior distribution of K
compute_post_k <- function(samples){

  denom <- length(samples)

  prob_k <- matrix(nrow = 30, ncol = 1)

  for(i in 1:30){

    prob_k[i] <- length(which(samples == i))

  }

  return(prob_k / denom)

}

# a function to generate truncated normals, we use this to generate starting points for the MCMC
rnorm_trunc <- function(n, mu = 0, sigma = 1, t1, t2){

  b1 <- pnorm((t1 - mu) / sigma)
  b2 <- pnorm((t2 - mu) / sigma)
  x <- mu + sigma * qnorm(runif(n, b1, b2))

  return(x)

}

# We compute the ESS per iteration of 1000 runs
# we create vectors to store the ESS
vect_ESS_RJ <- matrix(ncol = 1, nrow = 1000)
vect_ESS_RJ_v2 <- matrix(ncol = 1, nrow = 1000)
vect_ESS_NRJ <- matrix(ncol = 1, nrow = 1000)
vect_ESS_NRJ3 <- matrix(ncol = 1, nrow = 1000)
# and matrices to store the empirical distributions
mat_prob_RJ <- matrix(ncol = 30, nrow = 1000)
mat_prob_RJ_v2 <- matrix(ncol = 30, nrow = 1000)
mat_prob_NRJ <- matrix(ncol = 30, nrow = 1000)
mat_prob_NRJ3 <- matrix(ncol = 30, nrow = 1000)

set.seed(1)
for(i in 1:1000){ # run this loop in parallel, one iteration of this loop takes about 15 hours on a regular laptop
  # We see here the slowness of R when everything is computed sequentially
  # We coded it this way to make it easy to understand and also because
  # we ran the code in parallel by running 75 times this script with different seeds in batch mode in Linux on 75 CPUs, so it took about 0.5 * 1000 / 75 \approx 7 days
  # To run them in batch, one has to use the console and add lines in the loop to save the results in csv files (for instance)
  # and then put the results together
  # for a test, one can run 14 iterations on a laptop thus taking about 1 week and observe the results
  # We also tried it on a recent computer with 16 cores using parallel computing 
  # and one iteration takes about 3 hours.
  
  # we generate an initial value for K
  ini_val_K <- sample(prob_k[, 1], size = 1, prob = prob_k[, 2])
  ini_val_S <- matrix(ncol = 1, nrow = ini_val_K)
  ini_val_H <- matrix(ncol = 1, nrow = ini_val_K + 1)
  for(j in 1:ini_val_K){

    if(j == 1){

       low_bound <- 0

    }
    else{low_bound <- ini_val_S[j - 1]}
    if(j < ini_val_K){

      up_bound <- MAP_s[j + 1, ini_val_K]

    }
    else{up_bound <- L}

    ini_val_S[j] <- rnorm_trunc(n = 1, mu = MAP_s[j, ini_val_K], sigma = 500, low_bound, up_bound) # we add small noise around the MAP

  }
  for(j in 1:(ini_val_K + 1)){

    ini_val_H[j] <- rnorm_trunc(n = 1, mu = MAP_h[j, ini_val_K], sigma = 0.0005, 0, 1)

  }
  
  results <- RJ(nb_iter_vanilla, ini_val_K, ini_val_S, ini_val_H, y, lambda, K_max, alpha, beta, L, tau_vanilla, B_vanilla)
  vect_ESS_RJ[i] <- ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))
  mat_prob_RJ[i, ] <- compute_post_k(results$matrix_states_K)
  results <- RJ_v2(nb_iter, ini_val_K, ini_val_S, ini_val_H, y, lambda, K_max, alpha, beta, L, tau, B, T, N)
  vect_ESS_RJ_v2[i] <- ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))
  mat_prob_RJ_v2[i, ] <- compute_post_k(results$matrix_states_K)
  ini_val_d <- sample(c(-1, 1), size = 1)
  results <- Algorithm1(nb_iter_vanilla, ini_val_K, ini_val_S, ini_val_H, ini_val_d, y, lambda, K_max, alpha, beta, L, tau_vanilla, B_vanilla)
  vect_ESS_NRJ[i] <- ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))
  mat_prob_NRJ[i, ] <- compute_post_k(results$matrix_states_K)
  results <- Algorithm3(nb_iter, ini_val_K, ini_val_S, ini_val_H, ini_val_d, y, lambda, K_max, alpha, beta, L, tau, B, T, N)
  vect_ESS_NRJ3[i] <- ESS(as.mcmc(results$matrix_states_K[which(results$vect_move_type == 2)])) / length(which(results$vect_move_type == 2))
  mat_prob_NRJ3[i, ] <- compute_post_k(results$matrix_states_K)
  
}

# We now sample the model indicator K using ideal samplers
# for this we use samplers where we just generate K
# we thus need the posterior distribution on K
# we approximate it using what we observed with the chains generated from Algorithm 3 and the corresponding RJ

post_k <- cbind(seq(1, 30), apply(rbind(mat_prob_RJ_v2, mat_prob_NRJ3), 2, mean))

# we create a vector to store the ESS
vect_ESS_ideal_RJ <- matrix(ncol = 1, nrow = 1000)
matrix_prob_ideal_RJ <- matrix(ncol = 30, nrow = 1000)
vect_ESS_ideal_NRJ <- matrix(ncol = 1, nrow = 1000)
matrix_prob_ideal_NRJ <- matrix(ncol = 30, nrow = 1000)
nb_iter <- 27000 # corresponding to the (average) number of iterations in which model switches are proposed in the nonideal samplers
# (100000 - 10000) * 0.3

set.seed(1)
for(i in 1:1000){ # this takes about 4 minutes on a regular laptop

  # we generate an initial value for K
  ini_val_K <- sample(post_k[, 1], size = 1, prob = post_k[, 2])
  results_RJ <- ideal_RJ(ini_val_K, nb_iter, post_k[, 2], K_max)
  matrix_prob_ideal_RJ[i, ] <- compute_post_k(results_RJ)
  vect_ESS_ideal_RJ[i] <- ESS(as.mcmc(results_RJ)) / nb_iter
  ini_val_v <- sample(c(-1, 1), size = 1)
  results_NRJ <- ideal_NRJ(c(ini_val_K, ini_val_v), nb_iter, post_k[, 2], K_max)
  matrix_prob_ideal_NRJ[i, ] <- compute_post_k(results_NRJ[, 1])
  vect_ESS_ideal_NRJ[i] <- ESS(as.mcmc(results_NRJ[, 1])) / nb_iter

}

# we compute the total variation between the empirical distribution and the marginal posterior probabilities 
TV_NRJ3 <- mean(apply(abs(mat_prob_NRJ3 - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))
TV_RJ_v2 <- mean(apply(abs(mat_prob_RJ_v2 - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))
TV_NRJ <- mean(apply(abs(mat_prob_NRJ - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))
TV_RJ <- mean(apply(abs(mat_prob_RJ - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))
TV_NRJ_ideal <- mean(apply(abs(matrix_prob_ideal_NRJ - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))
TV_RJ_ideal <- mean(apply(abs(matrix_prob_ideal_RJ - matrix(ncol = 30, nrow = 1000, rep(post_k[, 2], 1000), byrow = TRUE)), 1, sum))

# we compute the relative differences
(TV_RJ_ideal - TV_NRJ_ideal) / TV_NRJ_ideal
# 0.94
(TV_NRJ3 - TV_NRJ_ideal) / TV_NRJ_ideal
# 0.94
(TV_RJ_v2 - TV_NRJ_ideal) / TV_NRJ_ideal
# 1.50
(TV_NRJ - TV_NRJ_ideal) / TV_NRJ_ideal
# 15.76
(TV_RJ - TV_NRJ_ideal) / TV_NRJ_ideal
# 16.66

# ESS per it. 
mean(vect_ESS_ideal_NRJ)
# 0.35
mean(vect_ESS_ideal_RJ)
# 0.09
mean(vect_ESS_NRJ3)
# 0.15
mean(vect_ESS_RJ_v2)
# 0.07
mean(vect_ESS_NRJ)
# 0.02
mean(vect_ESS_RJ)
# 0.01
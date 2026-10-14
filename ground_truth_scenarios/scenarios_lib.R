# =============================================================================
# Shared definitions for the ground-truth scenarios: settings, deterministic ODE truth, stochastic simulator (Alg. 8, fitted model), observation
# processes. Sourced by ground_truth_scenarios.R (figure) and fit_abc.R (SMC-ABC fits). No side effects.
# =============================================================================

# ---- Setup -------------------------------------------------------------------
N_POP    <- 1000                 # population size (known)
I0       <- 5                    # initially infectious (E0 = R0 = 0)
T_DAYS   <- 60                   # length of the daily series
MIN_PEAK <- 50                   # (unused by the deterministic ground truth; kept for reference)
DELAY    <- 5                    # reporting delay, in days
NB_SIZE  <- 50                   # negative-binomial size: var = mu + mu^2 / NB_SIZE (Poisson as size -> Inf)
SEED     <- 2026

par_names  <- c("lambda", "kappa", "gamma")
theta_true <- c(lambda = 0.50, kappa = 0.25, gamma = 0.20)   # R0 = lambda / gamma = 2.5

# ---- Simulator: Gillespie SEIR (Alg. 8), state recorded once per day -----------
#   S -> E  at rate lambda/N * S * I        (infection)
#   E -> I  at rate kappa * E               (end of latent period)
#   I -> R  at rate gamma * I               (removal)
simulate_seir_daily <- function(lam, kappa, gamma, N = N_POP, i_init = I0, T_days = T_DAYS) {
  s <- N - i_init; e <- 0; i <- i_init; r <- 0
  t <- 0; day <- 1
  path <- matrix(NA_real_, nrow = T_days, ncol = 4, dimnames = list(NULL, c("S", "E", "I", "R")))
  while (day <= T_days) {
    if (e + i == 0) {                                          # epidemic over: state stays frozen
      path[day:T_days, ] <- rep(c(s, e, i, r), each = T_days - day + 1)
      break
    }
    rate_inf <- lam / N * s * i; rate_onset <- kappa * e; rate_rem <- gamma * i
    total <- rate_inf + rate_onset + rate_rem
    t_new <- t + rexp(1, total)                                # time of the next event
    while (day <= T_days && t_new >= day) {                    # the state is constant until then
      path[day, ] <- c(s, e, i, r); day <- day + 1
    }
    u <- runif(1) * total                                      # which event?
    if (u < rate_inf) { s <- s - 1; e <- e + 1
    } else if (u < rate_inf + rate_onset) { e <- e - 1; i <- i + 1
    } else { i <- i - 1; r <- r + 1 }
    t <- t_new
  }
  data.frame(day = seq_len(T_days), path)
}

# ---- Ground truth: deterministic SEIR ODE (mean-field limit of the Gillespie model above) ----
#   dS = -lambda S I / N ; dE = lambda S I / N - kappa E ; dI = kappa E - gamma I ; dR = gamma I
# Classical RK4 in base R (no deSolve dependency), state recorded at integer days; day 1 = initial
# state (same indexing as simulate_seir_daily). No randomness: identical output on every call.
simulate_seir_ode <- function(lam, kappa, gamma, N = N_POP, i_init = I0, T_days = T_DAYS, h = 0.01) {
  f <- function(y) {
    inf <- lam / N * y[1] * y[3]
    c(-inf, inf - kappa * y[2], kappa * y[2] - gamma * y[3], gamma * y[3])
  }
  y <- c(N - i_init, 0, i_init, 0)
  path <- matrix(NA_real_, nrow = T_days, ncol = 4, dimnames = list(NULL, c("S", "E", "I", "R")))
  path[1, ] <- y
  n_sub <- round(1 / h)
  for (day in seq_len(T_days - 1)) {
    for (j in seq_len(n_sub)) {
      k1 <- f(y); k2 <- f(y + h / 2 * k1); k3 <- f(y + h / 2 * k2); k4 <- f(y + h * k3)
      y <- y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    }
    path[day + 1, ] <- y
  }
  data.frame(day = seq_len(T_days), path)
}

# The FITTED model's simulator: plain SEIR observed with Poisson noise (used inside ABC)
simulate_data <- function(theta) {
  I <- simulate_seir_daily(theta[1], theta[2], theta[3])$I
  rpois(length(I), I)
}

# ---- The true epidemic: deterministic, identical for every k ---------------------
# (k only selects the observation-noise draw in observe_scenarios)
make_epidemic <- function(k = 1) {
  simulate_seir_ode(theta_true[["lambda"]], theta_true[["kappa"]], theta_true[["gamma"]])
}

# ---- Observation processes ------------------------------------------------------
delay_series <- function(x, d) if (d <= 0) x else c(rep(0, d), x[seq_len(length(x) - d)])
add_pois     <- function(x) rpois(length(x), x)
add_nb       <- function(x) rnbinom(length(x), size = NB_SIZE, mu = x)

# Main scenarios (Poisson noise) and the negative-binomial sensitivity variants
SCENARIOS_MAIN <- c("clean", "noise", "delay", "delay + noise")
SCENARIOS_NB   <- c("noise (NB)", "delay + noise (NB)")
SCENARIOS_ALL  <- c(SCENARIOS_MAIN, SCENARIOS_NB)

# Observe epidemic `epi` in every scenario; the noise draws are reproducible given (k)
observe_scenarios <- function(epi, k) {
  I <- epi$I; Id <- delay_series(I, DELAY)
  set.seed(SEED + 100000L + k)
  list("clean"              = I,
       "noise"              = add_pois(I),
       "delay"              = Id,
       "delay + noise"      = add_pois(Id),
       "noise (NB)"         = add_nb(I),
       "delay + noise (NB)" = add_nb(Id))
}

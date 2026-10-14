# Settings + code loading for fit_abc.R (sourced by the main script and by every parallel worker).
# The working directory must be ground_truth_scenarios/ (fit_abc.R and the workers set it).
source("scenarios_lib.R")

# ---- priors: independent uniforms ----
PRIOR_LOWER <- c(lambda = 0.05, kappa = 0.05, gamma = 0.05)
PRIOR_UPPER <- c(lambda = 1.50, kappa = 1.00, gamma = 1.00)

# ---- distances ----
DISTANCES_TO_RUN <- c("ED", "KAD")
KAD_L      <- 3                          # half-window in days
KAD_KERNEL <- "box"                      # "box", "linear" or "gaussian"
KAD_FILE   <- normalizePath(file.path("..", "kad_functions.R"))   # absolute, from this directory

# ---- SMC-ABC (identical for every distance; equal simulation budget) ----
N_PARTICLES <- 500
ALPHA       <- 0.5
MIN_ACC     <- 0.02
MAX_GEN     <- 12
BUDGET_GRID <- c(2500, 5000, 10000)      # accuracy is evaluated at each budget; runs go up to the max
BUDGET_SIMS <- 5000                     # budget used for the main tables / figures (cost ~ proportional to max(BUDGET_GRID))
N_PP        <- 100                       # posterior-predictive curves per run

# ---- experiment ----
SCENARIOS_FIT <- SCENARIOS_ALL           # main (Poisson) + negative-binomial sensitivity scenarios
N_EPI   <- 10                            # independent epidemics, each observed in every scenario
N_CORES <- max(1, parallel::detectCores() - 1)
if (nzchar(Sys.getenv("QUICK"))) {
  N_EPI <- 2; N_PARTICLES <- 100; MAX_GEN <- 6; N_PP <- 20
  BUDGET_GRID <- c(1000, 2000, 3000); BUDGET_SIMS <- 2000
}
OUT_DIR <- if (nzchar(Sys.getenv("QUICK"))) "fit_quick" else "fit_results"

# ---- generic SMC-ABC, distances and metrics are reused from distance_comparison/ ----
for (f in c("smc_abc", "distances", "metrics"))
  source(file.path("..", "distance_comparison", "R", paste0(f, ".R")))

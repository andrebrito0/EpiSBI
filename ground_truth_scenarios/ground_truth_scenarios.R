# =============================================================================
# Ground-truth scenarios from ONE deterministic SEIR (ODE) epidemic  (figure + data files)
#
# A single epidemic is obtained by integrating the deterministic SEIR ODE (no demographic
# randomness) in a population of known size, and then
# OBSERVED in different ways (definitions in scenarios_lib.R):
#   1. clean          obs(t) = I(t)
#   2. noise          obs(t) = Poisson(I(t))
#   3. delay          obs(t) = I(t - DELAY)     (cases are seen DELAY days late; 0 before that)
#   4. delay + noise  obs(t) = Poisson(I(t - DELAY))
#   sensitivity: the same two noisy scenarios with negative-binomial (overdispersed) noise,
#   var = mu + mu^2 / NB_SIZE
# All scenarios share the same underlying epidemic, so differences between them are due to
# the observation process alone.
#
# Run from anywhere:   Rscript ground_truth_scenarios.R
# Outputs (in this directory): ground_truth_scenarios.csv / .rds, ground_truth_scenarios.png
# =============================================================================

suppressPackageStartupMessages(library(ggplot2))

# ---- locate this directory (works with Rscript, source(), Positron/RStudio run, any working dir) ----
# Returns the closest folder (the script's own, then the working directory's, and their parents) that contains `marker`.
locate_dir <- function(marker) {
  cand <- c(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)),        # Rscript
            tryCatch(sys.frame(1)$ofile, error = function(e) NULL),                          # source()
            tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) NULL),   # editor (RStudio / Positron)
            file.path(getwd(), "."))
  for (p in cand[nzchar(cand)]) {
    d <- normalizePath(if (dir.exists(p)) p else dirname(p), mustWork = FALSE)
    for (i in 0:4) {
      if (file.exists(file.path(d, marker))) return(d)                  # the project folder containing `marker`
      d <- dirname(d)
    }
  }
  stop("cannot find '", marker, "': run the script from inside the EpiSBI project")
}
GT_DIR <- file.path(locate_dir("ground_truth_scenarios/scenarios_lib.R"), "ground_truth_scenarios"); setwd(GT_DIR)
out_dir <- GT_DIR
source("scenarios_lib.R")

epi  <- make_epidemic(1)
obs  <- observe_scenarios(epi, 1)
true_I <- epi$I

df <- do.call(rbind, lapply(SCENARIOS_ALL, function(s)
  data.frame(scenario = factor(s, levels = SCENARIOS_ALL), day = epi$day, obs_I = obs[[s]], true_I = true_I)))

write.csv(df, file.path(out_dir, "ground_truth_scenarios.csv"), row.names = FALSE)
saveRDS(list(epidemic = epi, scenarios = obs, theta_true = theta_true,
             N_POP = N_POP, I0 = I0, T_DAYS = T_DAYS, DELAY = DELAY, NB_SIZE = NB_SIZE, SEED = SEED),
        file.path(out_dir, "ground_truth_scenarios.rds"))

p <- ggplot(df, aes(day)) +
  geom_line(aes(y = true_I), colour = "firebrick", linetype = 2) +
  geom_point(aes(y = obs_I), size = 1.1) + geom_line(aes(y = obs_I), colour = "grey40", linewidth = 0.3) +
  facet_wrap(~scenario, ncol = 2) + theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank()) +
  labs(x = "day", y = "infectious individuals I(t)",
       title = sprintf("One deterministic SEIR epidemic (N = %d, R0 = %.1f), observed in %d ways", N_POP,
                       theta_true[["lambda"]] / theta_true[["gamma"]], length(SCENARIOS_ALL)),
       subtitle = sprintf("black = observed series, dashed red = true I(t); delay = %d days; NB size = %g", DELAY, NB_SIZE))
ggsave(file.path(out_dir, "ground_truth_scenarios.png"), p, width = 9, height = 8.5, dpi = 110); 

p

cat(sprintf("peak of the true epidemic: I = %.1f on day %d (N = %d)\n", max(true_I), which.max(true_I), N_POP))
cat(sprintf("peak of the delayed series: day %d\n", which.max(obs[["delay"]])))
cat("wrote ground_truth_scenarios.csv/.rds/.png in", out_dir, "\n")

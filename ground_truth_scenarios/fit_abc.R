# =============================================================================
# Fit the ground-truth scenarios with SMC-ABC, using ED and KAD
#
# For each of N_EPI independent epidemics (true parameters theta_true) and each observation
# scenario (clean, noise, delay, delay + noise, plus the negative-binomial variants), the SAME
# model is fitted: a plain SEIR observed with Poisson noise, with NO delay. So
#   clean             slightly misspecified (the data have less noise than the model assumes)
#   noise             well specified
#   delay             misspecified in timing
#   delay + noise     misspecified in timing, noisy
#   (NB)              overdispersed noise
# ED and KAD are run on the same data, with the same seeds and the same simulation budget.
# Compared: accuracy of (lambda, kappa, gamma, R0), CRPS, coverage, and how well the fitted
# curve recovers the TRUE (undelayed) epidemic; and the paired KAD - ED differences.
#
# Run:   Rscript fit_abc.R          (QUICK=1 for a smoke test; settings in fit_setup.R)
# Outputs: fit_results/*.csv and fit_results/*.png
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
source("fit_setup.R")

# One task = one (epidemic k, scenario): one dataset, one SMC-ABC run per distance (common seeds).
run_task <- function(task) {
  epi  <- make_epidemic(task$k)
  obs  <- observe_scenarios(epi, task$k)[[task$scenario]]
  data <- list(obs_I = obs, mu_obs = epi$I)            # curve error is measured against the TRUE epidemic
  sidx <- match(task$scenario, SCENARIOS_ALL)

  rows <- lapply(DISTANCES_TO_RUN, function(dn) {
    dist_fn <- get_distance(dn)
    set.seed(SEED + 7L * task$k + 100000L * sidx)       # same seed for every distance
    t0   <- Sys.time()
    gens <- smc_abc(obs, dist_fn, N_PARTICLES, ALPHA, eps_target = -Inf, MIN_ACC, MAX_GEN, budget = max(BUDGET_GRID))
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cum  <- cumsum(vapply(gens, function(g) g$n_sims, numeric(1)))
    per_gen <- lapply(seq_along(gens), function(g) {
      info <- data.frame(scenario = task$scenario, rep = task$k, distance = dn, gen = g,
                         cum_sims = cum[g], eps = gens[[g]]$eps, secs = secs)
      list(par = cbind(info, summarise_posterior(gens[[g]], theta_true)),
           run = cbind(info, predictive_metrics(gens[[g]], data)))
    })
    list(par = do.call(rbind, lapply(per_gen, `[[`, "par")), run = do.call(rbind, lapply(per_gen, `[[`, "run")))
  })
  list(par = do.call(rbind, lapply(rows, `[[`, "par")), run = do.call(rbind, lapply(rows, `[[`, "run")))
}

tasks <- do.call(c, lapply(SCENARIOS_FIT, function(s) lapply(seq_len(N_EPI), function(k) list(scenario = s, k = k))))
cat(sprintf("%d tasks (%d scenarios x %d epidemics) x %d distances, %d cores, budget %d\n",
            length(tasks), length(SCENARIOS_FIT), N_EPI, length(DISTANCES_TO_RUN), N_CORES, max(BUDGET_GRID)))

t0 <- Sys.time()
cl <- parallel::makeCluster(N_CORES)                    # PSOCK: fork-based mclapply is not allowed in Positron
invisible(parallel::clusterCall(cl, function(wd) { setwd(wd); source("fit_setup.R"); invisible(NULL) }, getwd()))
parallel::clusterExport(cl, "run_task")
out <- parallel::parLapplyLB(cl, tasks, function(tk) tryCatch(run_task(tk), error = function(e) e))
parallel::stopCluster(cl)
failed <- vapply(out, inherits, logical(1), "error")
if (any(failed)) stop(sum(failed), " of ", length(tasks), " task(s) failed; first error (", tasks[[which(failed)[1]]]$scenario, ", epidemic ",
                      tasks[[which(failed)[1]]]$k, "): ", conditionMessage(out[[which(failed)[1]]]), call. = FALSE)
res_all  <- do.call(rbind, lapply(out, `[[`, "par")); runs_all <- do.call(rbind, lapply(out, `[[`, "run"))
cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

res <- at_budget(res_all, BUDGET_SIMS); runs <- at_budget(runs_all, BUDGET_SIMS)    # main analysis
curves  <- budget_curves(res_all, runs_all, BUDGET_GRID)
metrics <- aggregate_metrics(res)
paired  <- paired_differences(res, runs, ref = DISTANCES_TO_RUN[1])
dir.create(OUT_DIR, showWarnings = FALSE)
write.csv(res_all, file.path(OUT_DIR, "runs_par_all_generations.csv"), row.names = FALSE)
write.csv(metrics, file.path(OUT_DIR, "metrics.csv"), row.names = FALSE)
write.csv(paired,  file.path(OUT_DIR, "paired_differences.csv"), row.names = FALSE)
write.csv(curves$par,  file.path(OUT_DIR, "metrics_by_budget.csv"), row.names = FALSE)
write.csv(curves$pred, file.path(OUT_DIR, "predictive_by_budget.csv"), row.names = FALSE)

# ---- console summary ----
options(width = 150)
cat(sprintf("\nAccuracy per distance at %d simulations\n", BUDGET_SIMS))
print(format(metrics[order(metrics$scenario, metrics$par, metrics$distance),
                     c("scenario", "par", "distance", "bias", "rmse", "rel_rmse", "crps", "coverage", "ci_width", "secs")],
             digits = 3), row.names = FALSE)
paired$sig <- ifelse(paired$lo > 0, "ED better", ifelse(paired$hi < 0, "KAD better", ""))
cat(sprintf("\nPaired differences (%s - %s); negative = %s better\n", DISTANCES_TO_RUN[2], DISTANCES_TO_RUN[1], DISTANCES_TO_RUN[2]))
print(format(paired[order(paired$scenario, paired$metric, paired$par), c("scenario", "metric", "par", "n", "mean_diff", "lo", "hi", "sig")],
             digits = 3), row.names = FALSE)

# ---- figures ----
lvl <- c(par_names, "R0")
res$par <- factor(res$par, lvl); curves$par$par <- factor(curves$par$par, lvl)
for (nm in c("res", "runs")) { d <- get(nm); d$scenario <- factor(d$scenario, SCENARIOS_FIT); assign(nm, d) }
curves$par$scenario <- factor(curves$par$scenario, SCENARIOS_FIT); curves$pred$scenario <- factor(curves$pred$scenario, SCENARIOS_FIT)
paired$scenario <- factor(paired$scenario, SCENARIOS_FIT)
res$err_rel <- (res$mean - res$truth) / res$truth
theme_set(theme_bw(base_size = 11) + theme(panel.grid.minor = element_blank()))
cols <- c(ED = "steelblue", KAD = "darkorange")
save_fig <- function(p, name, w = 10, h = 5) ggsave(file.path(OUT_DIR, name), p, width = w, height = h, dpi = 110)

save_fig(ggplot(res, aes(distance, err_rel, fill = distance)) + geom_hline(yintercept = 0, linetype = 2) +
  geom_boxplot(alpha = 0.6, outlier.size = 0.8) + facet_grid(scenario ~ par) +
  scale_fill_manual(values = cols, guide = "none") +
  labs(x = NULL, y = "relative error of posterior mean", title = "Parameter accuracy (0 = true value)"),
  "01_relative_error.png", 10, 1.8 * length(SCENARIOS_FIT) + 1)

pred_long <- rbind(
  data.frame(runs[, c("scenario", "distance")], metric = "|peak day error| (days)", value = abs(runs$peak_day_err)),
  data.frame(runs[, c("scenario", "distance")], metric = "curve RMSE (sqrt scale)", value = runs$curve_rmse))
save_fig(ggplot(pred_long, aes(distance, value, fill = distance)) + geom_boxplot(alpha = 0.6, outlier.size = 0.8) +
  facet_grid(metric ~ scenario, scales = "free_y") + scale_fill_manual(values = cols, guide = "none") +
  labs(x = NULL, y = NULL, title = "Does the fitted curve recover the TRUE epidemic? (lower is better)"),
  "02_predictive.png", 13, 5)

save_fig(ggplot(paired, aes(mean_diff, interaction(par, metric, sep = " | "), colour = scenario)) +
  geom_vline(xintercept = 0, linetype = 2) + geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(0.7), size = 0.25) +
  facet_wrap(~metric, scales = "free", ncol = 3) +
  labs(x = sprintf("paired difference %s - %s (95%% bootstrap CI); < 0: %s better", DISTANCES_TO_RUN[2], DISTANCES_TO_RUN[1], DISTANCES_TO_RUN[2]),
       y = NULL, title = "Where does KAD beat (or lose to) ED?"), "03_paired_differences.png", 14, 8)

save_fig(ggplot(curves$par, aes(budget, rel_rmse, colour = distance)) + geom_line() + geom_point() +
  facet_grid(scenario ~ par, scales = "free_y") + scale_colour_manual(values = cols) + scale_x_log10() +
  labs(x = "simulation budget (log10)", y = "relative RMSE", title = "Parameter accuracy at equal simulation budget"),
  "04_accuracy_vs_budget.png", 10, 1.8 * length(SCENARIOS_FIT) + 1)

cat(sprintf("\nwrote %s/*.csv and *.png\n", OUT_DIR))

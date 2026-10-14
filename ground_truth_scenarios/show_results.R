# =============================================================================
# Show the results of fit_abc.R: console tables + one summary figure.
#
# Reads fit_results/ (fit_quick/ with QUICK=1) and evaluates every run at ONE simulation budget
# (default: the largest budget that was run; see below), the same budget for ED and KAD.
#
# Usage:   Rscript show_results.R             # largest budget
#          Rscript show_results.R 5000        # a given budget (must be <= the largest one that was run)
#          RESULTS_BUDGET=5000 Rscript ...    # same, via environment
# From an R session: source("show_results.R")
# Outputs: fit_results/summary_at_budget.csv, fit_results/results_summary.png
# =============================================================================

suppressPackageStartupMessages({ library(ggplot2); library(cowplot) })

# ---- locate this directory (works with Rscript, source(), Positron/RStudio run, any working dir) ----
locate_dir <- function(marker) {
  cand <- c(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)),        # Rscript
            tryCatch(sys.frame(1)$ofile, error = function(e) NULL),                          # source()
            tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) NULL),   # editor (RStudio / Positron)
            file.path(getwd(), "."))
  for (p in cand[nzchar(cand)]) {
    d <- normalizePath(if (dir.exists(p)) p else dirname(p), mustWork = FALSE)
    for (i in 0:4) {
      if (file.exists(file.path(d, marker))) return(d)
      d <- dirname(d)
    }
  }
  stop("cannot find '", marker, "': run the script from inside the EpiSBI project")
}
GT_DIR <- file.path(locate_dir("ground_truth_scenarios/scenarios_lib.R"), "ground_truth_scenarios"); setwd(GT_DIR)
RES_DIR <- if (nzchar(Sys.getenv("QUICK"))) "fit_quick" else "fit_results"
need <- file.path(RES_DIR, c("runs_par_all_generations.csv", "metrics_by_budget.csv", "predictive_by_budget.csv"))
if (!all(file.exists(need))) stop("missing in ", RES_DIR, ": ", paste(basename(need[!file.exists(need)]), collapse = ", "), "  (run fit_abc.R first)")

runs  <- read.csv(file.path(RES_DIR, "runs_par_all_generations.csv"))   # one row per run x generation x parameter
byB   <- read.csv(file.path(RES_DIR, "metrics_by_budget.csv"))
predB <- read.csv(file.path(RES_DIR, "predictive_by_budget.csv"))

# ---- the budget to show ----
a <- c(commandArgs(trailingOnly = TRUE), Sys.getenv("RESULTS_BUDGET"))
a <- suppressWarnings(as.numeric(a[nzchar(a)]))
max_run <- max(runs$cum_sims)
B <- if (length(a) && !is.na(a[1])) a[1] else max(byB$budget)
if (B > max(byB$budget)) stop(sprintf("budget %g is larger than the largest budget that was run (%g)", B, max(byB$budget)))

# posterior "at budget B": the last generation completed within B simulations (generation 1 if none)
at_B <- function(df, B) {
  key <- paste(df$scenario, df$rep, df$distance)
  ok  <- df$cum_sims <= B | df$gen == 1
  best <- tapply(ifelse(ok, df$gen, NA_real_), key, max, na.rm = TRUE)
  df[df$gen == best[key], ]
}
d <- at_B(runs, B)
d$err <- d$mean - d$truth; d$abs_err <- abs(d$err); d$cover <- d$truth >= d$lo & d$truth <= d$hi

SCEN <- c("clean", "noise", "delay", "delay + noise", "noise (NB)", "delay + noise (NB)")
SCEN <- SCEN[SCEN %in% d$scenario]; PARS <- c("lambda", "kappa", "gamma", "R0")
d$scenario <- factor(d$scenario, SCEN); d$par <- factor(d$par, PARS)
cols <- c(ED = "steelblue", KAD = "darkorange")

# ---- 0. what was run ----
n_epi <- length(unique(d$rep)); ng <- mean(tapply(d$gen, paste(d$scenario, d$rep, d$distance), max))
cat(sprintf("results in %s | %d epidemics x %d scenarios x %s | budget shown: %g simulations (largest run: %g)\n",
            RES_DIR, n_epi, length(SCEN), paste(unique(d$distance), collapse = " & "), B, max_run))
cat(sprintf("SMC-ABC generations completed at this budget: %.1f on average (generation 1 = prior sample)\n", ng))
if (ng < 4) cat("  !! few generations: the posteriors are still close to the prior, so ED vs KAD cannot be told apart at this budget.\n")

# ---- 1. accuracy table (R0 and the others), ED vs KAD side by side ----
summ <- do.call(rbind, lapply(split(d, interaction(d$scenario, d$distance, d$par, drop = TRUE)), function(x) data.frame(
  scenario = x$scenario[1], distance = x$distance[1], par = x$par[1], n = nrow(x),
  bias = mean(x$err), rmse = sqrt(mean(x$err^2)), rel_rmse = sqrt(mean((x$err / x$truth)^2)),
  crps = mean(x$crps), coverage = mean(x$cover), ci_width = mean(x$hi - x$lo))))
summ <- summ[order(summ$scenario, summ$par, summ$distance), ]; rownames(summ) <- NULL
write.csv(summ, file.path(RES_DIR, "summary_at_budget.csv"), row.names = FALSE)
options(width = 160)
wide <- function(par, col) { s <- summ[summ$par == par, ]; w <- reshape(s[, c("scenario", "distance", col)], idvar = "scenario", timevar = "distance", direction = "wide")
  names(w) <- sub(paste0(col, "\\."), "", names(w)); w }
cat("\nR0: relative RMSE (lower is better), CRPS (lower is better), coverage of the 95% interval\n")
tab <- Reduce(function(a, b) merge(a, b, by = "scenario"), list(
  setNames(wide("R0", "rel_rmse"), c("scenario", "relRMSE.ED", "relRMSE.KAD")),
  setNames(wide("R0", "crps"),     c("scenario", "CRPS.ED", "CRPS.KAD")),
  setNames(wide("R0", "coverage"), c("scenario", "cover.ED", "cover.KAD"))))
tab <- tab[match(SCEN, tab$scenario), ]; print(format(tab, digits = 3), row.names = FALSE)
cat("\nAll parameters, ED -> KAD relative RMSE:\n")
rr <- reshape(summ[, c("scenario", "par", "distance", "rel_rmse")], idvar = c("scenario", "par"), timevar = "distance", direction = "wide")
rr$KAD_minus_ED <- rr$rel_rmse.KAD - rr$rel_rmse.ED
print(format(reshape(rr[, c("scenario", "par", "KAD_minus_ED")], idvar = "scenario", timevar = "par", direction = "wide")[match(SCEN, unique(rr$scenario)), ], digits = 2), row.names = FALSE)
cat("(negative = KAD more accurate)\n")

# ---- 2. paired differences per epidemic (same data for both distances) ----
set.seed(1)
pair <- do.call(rbind, lapply(c("abs_err", "crps"), function(mt) do.call(rbind, lapply(split(d, interaction(d$scenario, d$par, drop = TRUE)), function(x) {
  w <- merge(x[x$distance == "ED", c("rep", mt)], x[x$distance == "KAD", c("rep", mt)], by = "rep", suffixes = c("_ED", "_KAD"))
  df <- w[[paste0(mt, "_KAD")]] - w[[paste0(mt, "_ED")]]; bs <- replicate(2000, mean(sample(df, replace = TRUE)))
  data.frame(scenario = x$scenario[1], par = x$par[1], metric = mt, n = length(df), mean_diff = mean(df), lo = quantile(bs, .025), hi = quantile(bs, .975))
}))))
pair$verdict <- ifelse(pair$hi < 0, "KAD better", ifelse(pair$lo > 0, "ED better", "no difference")); rownames(pair) <- NULL
cat(sprintf("\nPaired KAD - ED differences (%d scenarios x 4 parameters x 2 metrics = %d comparisons): ", length(SCEN), nrow(pair)))
print(table(pair$verdict))
cat(sprintf("(by chance alone about %.0f of them would look 'significant' at the 5%% level)\n", 0.05 * nrow(pair)))
if (any(pair$verdict != "no difference")) { cat("those that do:\n"); print(format(pair[pair$verdict != "no difference", c("scenario", "par", "metric", "mean_diff", "lo", "hi", "verdict")], digits = 3), row.names = FALSE) }

# ---- 3. curve accuracy (against the TRUE epidemic) at this budget ----
pb <- predB[predB$budget == B, c("scenario", "distance", "curve_rmse", "abs_peak_day")]
if (nrow(pb)) { cat("\nFitted curve vs the true epidemic: curve RMSE (sqrt scale) and |peak-day error| (days), lower is better\n")
  pw <- reshape(pb, idvar = "scenario", timevar = "distance", direction = "wide"); print(format(pw[match(SCEN, pw$scenario), ], digits = 3), row.names = FALSE) }

# ---- 4. figure ----
theme_set(theme_bw(base_size = 10) + theme(panel.grid.minor = element_blank(), legend.position = "bottom"))
truth_R0 <- unique(d$truth[d$par == "R0"])[1]
r0 <- d[d$par == "R0", ]
p1 <- ggplot(r0, aes(scenario, mean, colour = distance, group = interaction(rep, distance))) +
  geom_hline(yintercept = truth_R0, linetype = 2) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(0.8), size = 0.15, linewidth = 0.3, alpha = 0.7) +
  scale_colour_manual(values = cols) + coord_cartesian(ylim = c(0, min(max(r0$hi), 3 * truth_R0))) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL, y = "R0 (posterior mean and 95% interval)", title = sprintf("R0 estimate per epidemic (dashed = true %.1f)", truth_R0), colour = NULL)

p1

p2 <- ggplot(summ, aes(scenario, rel_rmse, fill = distance)) + geom_col(position = "dodge", alpha = 0.85) +
  facet_wrap(~par, scales = "free_y") + scale_fill_manual(values = cols) + theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(x = NULL, y = "relative RMSE", title = "Accuracy by parameter (lower is better)", fill = NULL)
bb <- byB[byB$par == "R0", ]; bb$scenario <- factor(bb$scenario, SCEN)

p2

p3 <- ggplot(bb, aes(budget, rel_rmse, colour = distance)) + geom_line() + geom_point(size = 1) + facet_wrap(~scenario, nrow = 2) +
  scale_colour_manual(values = cols) + scale_x_log10() + geom_vline(xintercept = B, linetype = 3) +
  labs(x = "simulation budget (log10)", y = "R0 relative RMSE", title = "R0 accuracy vs budget (dotted = budget shown)", colour = NULL)
pair$scenario <- factor(pair$scenario, SCEN)

p3

p4 <- ggplot(pair[pair$par %in% c("R0", "kappa"), ], aes(mean_diff, scenario, colour = verdict)) + geom_vline(xintercept = 0, linetype = 2) +
  geom_pointrange(aes(xmin = lo, xmax = hi), size = 0.25) + facet_grid(par ~ metric, scales = "free_x") +
  scale_colour_manual(values = c("KAD better" = "darkorange", "ED better" = "steelblue", "no difference" = "grey40")) + scale_y_discrete(limits = rev) +
  labs(x = "paired difference KAD - ED (95% bootstrap CI); < 0: KAD better", y = NULL, title = "Paired differences", colour = NULL)

p4

fig <- plot_grid(p1, p2, p3, p4, ncol = 2, labels = c("A", "B", "C", "D"), label_size = 11)
title <- ggdraw() + draw_label(sprintf("ED vs KAD, SMC-ABC at %g simulations (%d epidemics per scenario)", B, n_epi), fontface = "bold", size = 12)
out_png <- file.path(RES_DIR, "results_summary.png")
ggsave(out_png, plot_grid(title, fig, ncol = 1, rel_heights = c(0.04, 1)), width = 15, height = 10.5, dpi = 110, bg = "white")
cat(sprintf("\nwrote %s and %s\n", file.path(RES_DIR, "summary_at_budget.csv"), out_png))

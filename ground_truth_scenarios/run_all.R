# =============================================================================
# Umbrella script: runs the ED vs KAD pipeline end to end, one step after the other, with checks.
#
#   truth        ground_truth_scenarios/ground_truth_scenarios.R   one epidemic, 6 observation scenarios (figure + data)
#   fit          ground_truth_scenarios/fit_abc.R                  SMC-ABC fits of those scenarios with ED and KAD
#   check        distance_comparison/check_implementation.R        sanity checks (vectorised KAD, generators)
#   compare      distance_comparison/run_comparison.R              ED vs KAD under model misspecification (long)
#   sensitivity  distance_comparison/sensitivity_kad.R             KAD window / kernel sensitivity (long)
#
# Usage (any working directory inside the project):
#   Rscript run_all.R                       # default steps: truth fit
#   Rscript run_all.R truth fit check       # chosen steps, in the order given
#   Rscript run_all.R all                   # every step
#   Rscript run_all.R preflight             # only the preflight checks, run nothing
#   QUICK=1 Rscript run_all.R all           # smoke test of everything (small settings, ~2 min)
# From an R session: STEPS <- c("truth", "fit"); source("run_all.R")
#
# Checks along the way:
#   1. PREFLIGHT   R packages, every script/helper file, Rscript itself, and that the steps asked for exist.
#   2. PER STEP    the step's console output is also saved to pipeline_logs/<step>.log, and on failure the
#                  end of that log is shown.
#   3. POST-CHECK  after each step, its outputs are verified: files exist and are fresh (written by this run),
#                  tables have the expected scenarios/distances/parameters, the same number of replicates in
#                  every cell, only finite values, probabilities in [0, 1]; the sanity-check step must report
#                  differences below tolerance. A failed check stops the pipeline.
# Every step runs in its OWN R process: the two studies define the same names with different values
# (e.g. N_POP), so they must not share a session.
# =============================================================================

# ---- locate the project (works with Rscript, source(), Positron/RStudio run, any working dir) ----
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
ROOT  <- locate_dir("kad_functions.R")
QUICK <- nzchar(Sys.getenv("QUICK"))
GT    <- file.path(ROOT, "ground_truth_scenarios")
DC    <- file.path(ROOT, "distance_comparison")
LOGS  <- file.path(ROOT, "pipeline_logs")

# ---- the steps ----
PIPELINE <- c(
  truth       = "ground_truth_scenarios/ground_truth_scenarios.R",
  fit         = "ground_truth_scenarios/fit_abc.R",
  check       = "distance_comparison/check_implementation.R",
  compare     = "distance_comparison/run_comparison.R",
  sensitivity = "distance_comparison/sensitivity_kad.R")
DEFAULT_STEPS <- c("truth", "fit")

# ============================ 1. PREFLIGHT ============================
preflight <- function(steps) {
  problems <- character(0)
  need <- c("ggplot2", "dtw", "gridExtra", "dplyr", "cowplot", "tidyr", "scales", "parallel")
  miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) problems <- c(problems, sprintf("R packages not installed: %s  ->  install.packages(c(%s))",
                                                    paste(miss, collapse = ", "), paste0('"', miss, '"', collapse = ", ")))
  files <- c(file.path(ROOT, "kad_functions.R"), file.path(ROOT, PIPELINE[steps]),
             file.path(GT, c("scenarios_lib.R", "fit_setup.R")),
             file.path(DC, "R", paste0(c("config", "simulator", "distances", "smc_abc", "metrics", "runner"), ".R")))
  gone <- files[!file.exists(files)]
  if (length(gone)) problems <- c(problems, paste("missing file:", sub(paste0(ROOT, "/"), "", gone, fixed = TRUE)))
  if (!file.exists(file.path(R.home("bin"), "Rscript"))) problems <- c(problems, "Rscript not found next to R")
  if (!nzchar(Sys.which("bash"))) cat("note: bash not found, logs will not be saved (output still shown)\n")
  nc <- tryCatch(parallel::detectCores(), error = function(e) NA)
  if (is.na(nc) || nc < 1) problems <- c(problems, "cannot detect CPU cores")
  problems
}

# ============================ 3. POST-CHECKS ============================
# Each returns a character vector of problems (empty = OK). `t0` = start time of the step.
fresh <- function(paths, t0, min_bytes = 200) {
  out <- character(0)
  for (p in paths) {
    if (!file.exists(p)) { out <- c(out, paste("not written:", basename(p))); next }
    fi <- file.info(p)
    if (is.na(fi$size) || fi$size < min_bytes) out <- c(out, paste("empty/too small:", basename(p)))
    if (fi$mtime < t0 - 2) out <- c(out, paste("stale (not written by this run):", basename(p)))
  }
  out
}

# Table of metrics: per scenario x distance x parameter, equal replicates, finite, sensible ranges
check_metrics <- function(path, scenarios, distances, pars = c("lambda", "kappa", "gamma", "R0")) {
  m <- read.csv(path); out <- character(0); nm <- basename(path)
  if (!setequal(unique(m$scenario), scenarios))
    out <- c(out, sprintf("%s: scenarios are {%s}, expected {%s}", nm, paste(sort(unique(m$scenario)), collapse = ", "), paste(sort(scenarios), collapse = ", ")))
  if (!setequal(unique(m$distance), distances))
    out <- c(out, sprintf("%s: distances are {%s}, expected {%s}", nm, paste(sort(unique(m$distance)), collapse = ", "), paste(sort(distances), collapse = ", ")))
  if (!setequal(unique(m$par), pars)) out <- c(out, sprintf("%s: parameters are {%s}", nm, paste(sort(unique(m$par)), collapse = ", ")))
  if (length(unique(m$n_rep)) != 1) out <- c(out, sprintf("%s: unequal number of replicates across cells (%s): some tasks lost", nm, paste(sort(unique(m$n_rep)), collapse = ", ")))
  full <- expand.grid(scenario = scenarios, distance = distances, par = pars, stringsAsFactors = FALSE)
  have <- paste(m$scenario, m$distance, m$par)
  lost <- paste(full$scenario, full$distance, full$par)[!(paste(full$scenario, full$distance, full$par) %in% have)]
  if (length(lost)) out <- c(out, sprintf("%s: %d scenario x distance x parameter combination(s) missing, e.g. %s", nm, length(lost), lost[1]))
  if (anyDuplicated(have)) out <- c(out, sprintf("%s: duplicated scenario x distance x parameter rows", nm))
  nums <- as.matrix(m[vapply(m, is.numeric, logical(1))])
  if (any(!is.finite(nums))) out <- c(out, sprintf("%s: %d non-finite value(s) (NA/NaN/Inf)", nm, sum(!is.finite(nums))))
  if (any(m$coverage < 0 | m$coverage > 1, na.rm = TRUE)) out <- c(out, sprintf("%s: coverage outside [0, 1]", nm))
  if (any(m$rmse < 0 | m$crps < 0 | m$sd_post < 0, na.rm = TRUE)) out <- c(out, sprintf("%s: negative rmse/crps/sd", nm))
  if (any(m$n_sims <= 0, na.rm = TRUE)) out <- c(out, sprintf("%s: non-positive number of simulations", nm))
  out
}

post_checks <- list(
  truth = function(t0) {
    f <- file.path(GT, c("ground_truth_scenarios.csv", "ground_truth_scenarios.rds", "ground_truth_scenarios.png"))
    out <- fresh(f, t0, 1000)
    if (file.exists(f[1])) {
      d <- read.csv(f[1])
      if (length(unique(d$scenario)) != 6) out <- c(out, sprintf("expected 6 scenarios, found %d", length(unique(d$scenario))))
      if (anyNA(d$obs_I) || anyNA(d$true_I)) out <- c(out, "NA in the observed or true series")
      if (any(d$obs_I < 0, na.rm = TRUE)) out <- c(out, "negative observed counts")
      if (max(d$true_I, na.rm = TRUE) < 10) out <- c(out, "the epidemic never took off (peak < 10)")
      cl <- d[d$scenario == "clean", ]; if (nrow(cl) && !all(cl$obs_I == cl$true_I)) out <- c(out, "'clean' scenario differs from the true I(t)")
      rows <- table(d$scenario); if (length(unique(rows)) != 1) out <- c(out, "scenarios have different lengths")
    }
    out
  },
  fit = function(t0) {
    od <- file.path(GT, if (QUICK) "fit_quick" else "fit_results")
    f <- file.path(od, c("metrics.csv", "paired_differences.csv", "metrics_by_budget.csv", "predictive_by_budget.csv",
                         "01_relative_error.png", "02_predictive.png", "03_paired_differences.png", "04_accuracy_vs_budget.png"))
    out <- fresh(f, t0)
    if (file.exists(f[1])) out <- c(out, check_metrics(f[1], c("clean", "noise", "delay", "delay + noise", "noise (NB)", "delay + noise (NB)"), c("ED", "KAD")))
    out
  },
  check = function(t0, log) {
    txt <- if (file.exists(log)) readLines(log, warn = FALSE) else character(0)
    out <- character(0)
    d <- as.numeric(sub(".*= *([0-9.eE+-]+) *$", "\\1", txt[grepl("vectorised - library", txt)]))      # trailing number
    if (length(d) < 5) out <- c(out, sprintf("expected 5 KAD-vs-library comparisons in the log, found %d", length(d)))
    if (length(d) && any(!is.finite(d) | d > 1e-8)) out <- c(out, sprintf("vectorised KAD differs from the library (max %.2e > 1e-8)", max(d)))
    z <- as.numeric(sub(".*max \\|z\\| over days = *([0-9.]+).*", "\\1", txt[grepl("max \\|z\\|", txt)]))
    if (!length(z)) out <- c(out, "generator check (max |z|) missing from the log")
    else if (z > 4.5) out <- c(out, sprintf("staged generator (k=1) differs from the plain simulator: max |z| = %.2f > 4.5", z))
    out
  },
  compare = function(t0) {
    rd <- file.path(DC, if (QUICK) "results_quick" else "results_misspec"); fd <- file.path(DC, if (QUICK) "figures_quick" else "figures_misspec")
    only <- Sys.getenv("ONLY")
    scen <- if (nzchar(only)) strsplit(only, ",")[[1]] else c("control", "erlang", "intervention", "underreport")
    f <- c(file.path(rd, c("metrics.csv", "paired_differences.csv", "metrics_by_budget.csv", "predictive_by_budget.csv")),
           file.path(fd, c("01_relative_error.png", "05_paired_differences.png", "07_accuracy_vs_budget.png", "08_predictive_vs_budget.png")))
    out <- fresh(f, t0)
    if (file.exists(f[1])) out <- c(out, check_metrics(f[1], scen, c("ED", "KAD")))
    out
  },
  sensitivity = function(t0) {
    rd <- file.path(DC, if (QUICK) "results_quick" else "results_misspec"); fd <- file.path(DC, if (QUICK) "figures_quick" else "figures_misspec")
    f <- c(file.path(rd, c("sensitivity_metrics.csv", "sensitivity_paired.csv")), file.path(fd, "sensitivity.png"))
    out <- fresh(f, t0)
    if (file.exists(f[1])) out <- c(out, check_metrics(f[1], c("erlang", "intervention"),
                                                       c("ED", "KAD_L1_box", "KAD_L3_box", "KAD_L7_box", "KAD_L3_gaussian", "KAD_L7_gaussian")))
    out
  })

# ============================ run ============================
steps <- if (exists("STEPS", inherits = FALSE)) STEPS else commandArgs(trailingOnly = TRUE)
steps <- steps[nzchar(steps)]
if (length(steps) == 0) steps <- DEFAULT_STEPS
only_preflight <- identical(steps, "preflight")
if (identical(steps, "all")) steps <- names(PIPELINE)
if (only_preflight) steps <- names(PIPELINE)
bad <- setdiff(steps, names(PIPELINE))
if (length(bad)) stop("unknown step(s): ", paste(bad, collapse = ", "), ". Available: ", paste(names(PIPELINE), collapse = ", "), ", all, preflight", call. = FALSE)

cat(sprintf("project: %s\npipeline: %s%s\n\n", ROOT, paste(steps, collapse = " -> "), if (QUICK) "   [QUICK mode]" else ""))
cat("== preflight ==\n")
pf <- preflight(steps)
if (length(pf)) stop("preflight failed:\n  - ", paste(pf, collapse = "\n  - "), call. = FALSE)
cat("ok: packages, files and Rscript found\n\n")
if (only_preflight) { cat("preflight only: nothing run\n"); quit_ok <- TRUE } else {
  dir.create(LOGS, showWarnings = FALSE)
  rscript <- file.path(R.home("bin"), "Rscript")
  summary_tab <- data.frame(step = steps, status = "not run", minutes = NA_real_, stringsAsFactors = FALSE)
  for (i in seq_along(steps)) {
    st <- steps[i]; script <- file.path(ROOT, PIPELINE[[st]]); log <- file.path(LOGS, paste0(st, ".log"))
    cat(sprintf("==== [%s] %s ====\n", st, PIPELINE[[st]]))
    t0 <- Sys.time()
    if (nzchar(Sys.which("bash"))) {                              # live output AND a log file; exit status of Rscript kept
      cmd <- sprintf("set -o pipefail; %s %s 2>&1 | tee %s", shQuote(rscript), shQuote(script), shQuote(log))
      status <- system2("bash", c("-c", shQuote(cmd)))
    } else status <- system2(rscript, shQuote(script))
    summary_tab$minutes[i] <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
    fail <- function(why, detail = NULL) {
      summary_tab$status[i] <<- "FAILED"
      cat(sprintf("\n!!!! [%s] %s\n", st, why))
      if (length(detail)) cat(paste0("  - ", detail, collapse = "\n"), "\n")
      if (file.exists(log)) { tl <- tail(readLines(log, warn = FALSE), 15); cat(sprintf("\nlast lines of %s:\n", log), paste0("  | ", tl, collapse = "\n"), "\n")}
      cat("\n"); print(summary_tab, row.names = FALSE)
      stop(sprintf("pipeline stopped at step '%s'", st), call. = FALSE)
    }
    if (status != 0) fail(sprintf("the script exited with status %d", status))
    cat(sprintf("---- [%s] finished in %.1f min; checking outputs ...\n", st, summary_tab$minutes[i]))
    pc <- post_checks[[st]]
    problems <- tryCatch(if (st == "check") pc(t0, log) else pc(t0), error = function(e) paste("the output check itself failed:", conditionMessage(e)))
    if (length(problems)) fail("output checks failed", problems)
    summary_tab$status[i] <- "ok"
    cat(sprintf("---- [%s] outputs ok\n\n", st))
  }
  cat("pipeline finished:\n"); print(summary_tab, row.names = FALSE)
}

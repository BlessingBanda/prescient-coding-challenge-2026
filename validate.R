# -----------------------------------------------------------------------------
# Prescient Coding Challenge 2026 -- pre-submission validator (R).
#
# RUN THIS BEFORE YOU SUBMIT.   Rscript validate.R
#
# We score your solution on a window you have never seen, PLUS the four
# historical windows below. If your solution breaks a constraint or crashes on
# any of them, it scores nothing. This script runs every window we will run, so
# you can find that out now rather than after the deadline.
#
# A green board here does not mean you will win. It means you will be scored.
# -----------------------------------------------------------------------------

source("harness.R")
source("solution.R")

data <- load_data()
cat(sprintf("---> data available to %s\n", max(data$dates)))
cat("---> the scoring window starts after this date and you do not have it\n\n")

windows <- c(list("practice (2025)" = PRACTICE_WINDOW), ROBUSTNESS_WINDOWS)

cat(sprintf("%16s %10s %9s %7s %8s %9s %7s\n",
            "window", "result", "excess", "IR", "active", "turnover", "cost"))
cat(strrep("-", 73), "\n")

failures <- 0
t0 <- Sys.time()

for (label in names(windows)) {
  w <- windows[[label]]
  bt <- tryCatch(
    run_backtest(generate_weights, PARAMS, w[1], w[2], data = data, verbose = FALSE),
    error = function(e) e
  )
  if (inherits(bt, "error")) {
    failures <- failures + 1
    first <- strsplit(conditionMessage(bt), "\n")[[1]][1]
    cat(sprintf("%16s %10s   %s\n", label, "FAILED", substr(first, 1, 60)))
    next
  }
  m <- metrics(bt)
  cat(sprintf("%16s %10s %8.2f%% %7.2f %7.2f%% %8.2f%% %6.2f%%\n",
              label, "ok", 100 * m$excess_return, m$information_ratio,
              100 * m$mean_active_weight, 100 * m$avg_turnover, 100 * m$cost_drag))
}

elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(strrep("-", 73), "\n")
cat(sprintf("---> declared parameters : %d\n", length(PARAMS)))
cat(sprintf("---> total run time      : %.1fs (%s the 10 minute limit)\n",
            elapsed, if (elapsed < 600) "within" else "OVER"))

# determinism: the same inputs must give the same answer twice
a <- run_backtest(generate_weights, PARAMS, PRACTICE_WINDOW[1], PRACTICE_WINDOW[2],
                  data = data, verbose = FALSE)
b <- run_backtest(generate_weights, PARAMS, PRACTICE_WINDOW[1], PRACTICE_WINDOW[2],
                  data = data, verbose = FALSE)
stable <- max(abs(a$port_return - b$port_return)) < 1e-12
cat(sprintf("---> deterministic       : %s\n",
            if (stable) "yes" else "NO -- set.seed() your randomness"))

if (failures > 0 || !stable) {
  cat("\n*** NOT READY TO SUBMIT ***\n")
  cat("Fix the failures above. A submission that fails any window scores nothing.\n")
} else {
  cat("\nAll windows passed. You are ready to submit.\n")
}

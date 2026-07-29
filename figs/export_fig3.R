# Figure 3 panel a: per (exp, n_points) response-time scaling (full-view experiments).
suppressWarnings(suppressMessages(library(data.table)))
d  <- readRDS("dataana/outputs/data_cache.rds")
tr <- as.data.table(d$trials)
tr <- tr[is.finite(duration_ms) & duration_ms > 0]
# geometric mean RT per set size (for the log-log scaling curve)
rt <- tr[, .(gm_ms = exp(mean(log(duration_ms))),
             med_ms = median(duration_ms),
             lo = exp(mean(log(duration_ms)) - sd(log(duration_ms))/sqrt(.N)),
             hi = exp(mean(log(duration_ms)) + sd(log(duration_ms))/sqrt(.N)),
             n = .N),
         by = .(exp, n_points)][order(exp, n_points)]
fwrite(rt, "figs_nhb/data_for_figs/f3_rt_by_np.csv")
cat("RT scaling exported\n"); print(rt[exp %in% c("exp1","exp3")])

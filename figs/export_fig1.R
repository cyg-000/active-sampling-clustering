# Figure 1 panel d: per-participant reliability, so the two bars can be compared with a
# genuine PAIRED test instead of a Welch approximation off the summary means.
#
#   internal  = test-retest FM (rq1$fm), pair-level -> averaged within participant
#   inter     = inter-participant FM (rq2$agg), already one row per (exp, sid, base_uuid)
#
# Both live in results.rds keyed the same way (exp, sid, base_uuid), so every participant
# contributes exactly one internal and one inter value and the test is paired within sid.
suppressWarnings(suppressMessages(library(data.table)))
r <- readRDS("dataana/outputs/results.rds")

int <- as.data.table(r$rq1$fm)[!is.na(fm), .(fm_internal = mean(fm)), by = .(exp, sid)]
ipt <- as.data.table(r$rq2$agg)[!is.na(fm), .(fm_inter = mean(fm)), by = .(exp, sid)]

d <- merge(int, ipt, by = c("exp", "sid"))[order(exp, sid)]
fwrite(d, "figs_nhb/data_for_figs/f1_reliability_bysid.csv")

cat("per-participant reliability exported\n")
print(d[, .(n_sid = .N,
            internal = mean(fm_internal), inter = mean(fm_inter),
            delta = mean(fm_internal - fm_inter)), by = exp][order(exp)])

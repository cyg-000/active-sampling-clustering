# Export data for Figure 2 (invariant slopes + format reallocation + trait stability).
suppressWarnings(suppressMessages({
  library(data.table); library(lme4); library(lmerTest)
}))
dir.create("figs_nhb/data_for_figs", showWarnings = FALSE, recursive = TRUE)

d   <- readRDS("dataana/outputs/data_cache.rds")
tr  <- as.data.table(d$trials)
cl  <- as.data.table(d$clusters)

vis <- c(exp1="full", exp2="funnel", exp3="full", exp4="funnel")
fmt <- c(exp1="lasso", exp2="lasso", exp3="anchor", exp4="anchor")
tr[, visibility := vis[exp]][, format := fmt[exp]]
cl[, visibility := vis[exp]][, format := fmt[exp]]

se <- function(x) sd(x) / sqrt(length(x))

# ---- panels a/b: per (exp, n_points) mean +/- SE ----
kc <- tr[, .(mean = mean(n_clusters), se = se(n_clusters), n = .N),
         by = .(exp, n_points)][order(exp, n_points)]
fwrite(kc, "figs_nhb/data_for_figs/f2_nclusters_by_np.csv")

nz <- cl[, .(mean = mean(numerosity), se = se(numerosity), n = .N),
         by = .(exp, n_points)][order(exp, n_points)]
fwrite(nz, "figs_nhb/data_for_figs/f2_numerosity_by_np.csv")

# ---- panel c: pooled 2x2 interaction model (visibility x format x n_points) ----
tr[, format := relevel(factor(format), "lasso")]
tr[, visibility := relevel(factor(visibility), "full")]
cl[, format := relevel(factor(format), "lasso")]
cl[, visibility := relevel(factor(visibility), "full")]

grab <- function(m, lbl) {
  s <- summary(m)$coefficients
  keep <- grep("^n_points", rownames(s), value = TRUE)
  data.table(model = lbl, term = keep,
             beta = s[keep, "Estimate"], se = s[keep, "Std. Error"],
             p = s[keep, "Pr(>|t|)"])
}
# additive moderators (n_points:visibility + n_points:format, no three-way) to
# match the manuscript's reported reallocation coefficients.
mk <- lmer(n_clusters ~ n_points * (visibility + format) + (1|sid) + (1|base_uuid),
           data = tr, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa"))
mn <- lmer(numerosity  ~ n_points * (visibility + format) + (1|sid) + (1|base_uuid),
           data = cl, REML = FALSE,
           control = lmerControl(optimizer = "bobyqa"))
coefs <- rbind(grab(mk, "n_clusters"), grab(mn, "numerosity"))
fwrite(coefs, "figs_nhb/data_for_figs/f2_pooled_coefs.csv")
cat("--- pooled n_points interaction terms ---\n"); print(coefs)

# ---- panel d: test-retest of per-participant style (exp1, exp3 have 2 presentations) ----
kt <- tr[exp %in% c("exp1", "exp3"),
         .(mk = median(n_clusters)), by = .(exp, sid, presentation)]
ktw <- dcast(kt, exp + sid ~ presentation, value.var = "mk")
setnames(ktw, c("1", "2"), c("p1", "p2"))
ktw <- ktw[!is.na(p1) & !is.na(p2)]
fwrite(ktw, "figs_nhb/data_for_figs/f2_retest_nclusters.csv")
for (e in c("exp1", "exp3"))
  cat(e, "test-retest r(median k) =",
      round(cor(ktw[exp == e]$p1, ktw[exp == e]$p2), 3),
      "| n =", nrow(ktw[exp == e]), "\n")

cat("DONE fig2 export\n")

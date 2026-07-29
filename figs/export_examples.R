# Export real stimulus coords + example partitions for the NHB figures.
suppressWarnings(suppressMessages(library(data.table)))
dir.create("figs_nhb/data_for_figs", showWarnings = FALSE, recursive = TRUE)

d   <- readRDS("dataana/outputs/data_cache.rds")
pts <- as.data.table(d$points)

# ---- choose a representative exp1 (full-view, lasso), non-flipped stimulus ----
# want a mid/large array so imposed structure is visible; clustered group.
cand <- unique(pts[exp == "exp1" & flipped == FALSE,
                   .(base_uuid, n_points, group)])
cand <- cand[n_points == 30 & group == "clustered"]
target <- cand$base_uuid[1]
cat("target stimulus:", target, "\n")

sub <- pts[base_uuid == target & exp == "exp1" & flipped == FALSE]

# ---- panel b: the stimulus coordinates (fixed geometry across participants) ----
stim <- unique(sub[, .(point_idx, x, y)], by = "point_idx")[order(point_idx)]
fwrite(stim, "figs_nhb/data_for_figs/stim_coords.csv")
cat("stim points:", nrow(stim), "\n")

# ---- panel c: several participants' partitions (presentation 1) ----
sids <- sub[presentation == 1, .N, by = sid][order(-N)]$sid
sids <- head(sids, 6)
part <- sub[presentation == 1 & sid %in% sids,
            .(sid, point_idx, x, y, cluster_id, n_clusters)]
fwrite(part, "figs_nhb/data_for_figs/example_partitions.csv")
cat("example participants:", length(sids), "\n")

# ---- internal-reliability illustration: one participant, two presentations ----
# pick a sid that has both presentations of this stimulus
both <- sub[, .(np = uniqueN(presentation)), by = sid][np == 2]$sid
sid_rt <- both[1]
retest <- sub[sid == sid_rt, .(sid, presentation, point_idx, x, y, cluster_id)]
fwrite(retest, "figs_nhb/data_for_figs/retest_partition.csv")
cat("retest sid:", sid_rt, "| rows:", nrow(retest), "\n")

cat("DONE export\n")

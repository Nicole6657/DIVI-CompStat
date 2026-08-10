
suppressPackageStartupMessages(library(sparcl))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8) {
  stop("Expected 8 arguments: x_csv y_csv out_csv K nperms grid_size min_w seed")
}

x_csv    <- args[[1]]
y_csv    <- args[[2]]
out_csv  <- args[[3]]
K        <- as.integer(args[[4]])
nperms   <- as.integer(args[[5]])
grid_n   <- as.integer(args[[6]])
min_w    <- as.numeric(args[[7]])
seed     <- as.integer(args[[8]])

set.seed(seed)
x <- as.matrix(read.csv(x_csv, header=FALSE, check.names=FALSE))
y <- scan(y_csv, quiet=TRUE)
p <- ncol(x)

# In sparse k-means, ||w||_2 <= 1 implies 1 <= ||w||_1 <= sqrt(p).
# Avoid the exact lower boundary because an excessively sparse one-feature solution
# can be numerically unstable in permutation tuning.
max_w <- sqrt(p)
if (min_w >= max_w) min_w <- max(1.01, 0.5 * max_w)
wbounds <- seq(min_w, max_w, length.out=grid_n)

ptm <- proc.time()[[3]]
perm_fit <- KMeansSparseCluster.permute(
  x=x,
  K=K,
  wbounds=wbounds,
  nperms=nperms
)
bestw <- perm_fit$bestw
fit <- KMeansSparseCluster(x=x, K=K, wbounds=bestw)
runtime <- proc.time()[[3]] - ptm

# KMeansSparseCluster() can return either a direct fitted object or an
# unnamed one-element list, depending on how wbounds is represented.
# Recursively locate the first object containing a p-vector named `ws`.
find_spkm_fit <- function(obj, p, depth=0L) {
  if (depth > 5L) return(NULL)
  if (is.list(obj) && !is.null(obj$ws) && length(obj$ws) == p) return(obj)
  if (is.list(obj)) {
    for (ii in seq_along(obj)) {
      ans <- find_spkm_fit(obj[[ii]], p, depth + 1L)
      if (!is.null(ans)) return(ans)
    }
  }
  return(NULL)
}
fit_core <- find_spkm_fit(fit, p)
if (is.null(fit_core)) {
  stop(paste0(
    "Could not locate SPKM fit with a length-p `ws` vector. ",
    "top-level class=", paste(class(fit), collapse="/"),
    "; length=", length(fit),
    "; names=", paste(names(fit), collapse=","),
    "; structure=", paste(capture.output(str(fit, max.level=2)), collapse=" | ")
  ))
}
weights <- as.numeric(fit_core$ws)
weights[!is.finite(weights)] <- 0
weights <- pmax(weights, 0)
if (sum(weights) <= 0) stop("All SPKM feature weights are zero")

# Prefer the clustering returned by sparcl when it can be converted safely.
clusters <- NULL
Cs <- fit_core$Cs
if (!is.null(Cs)) {
  if (is.atomic(Cs) && length(Cs) == nrow(x)) {
    clusters <- as.integer(Cs)
  } else if (is.list(Cs) && length(Cs) == K) {
    tmp <- integer(nrow(x))
    for (kk in seq_along(Cs)) {
      idx <- as.integer(Cs[[kk]])
      idx <- idx[is.finite(idx) & idx >= 1L & idx <= nrow(x)]
      if (length(idx)) tmp[idx] <- kk
    }
    if (all(tmp > 0L)) clusters <- tmp
  }
}

# Fallback: optimize the conditional assignment for the learned SPKM weights
# by k-means after x_ij -> sqrt(w_j) x_ij. This has the same weighted
# within-cluster sum-of-squares objective for fixed w.
if (is.null(clusters)) {
  x_weighted <- sweep(x, 2L, sqrt(weights), FUN="*")
  set.seed(seed + 104729L)
  km <- stats::kmeans(x_weighted, centers=K, nstart=50L, iter.max=100L)
  clusters <- as.integer(km$cluster)
}
if (length(clusters) != nrow(x) || anyNA(clusters)) {
  stop(sprintf("Weighted k-means did not return valid labels: length=%d", length(clusters)))
}

# Return one row per feature; cluster labels are written as a semicolon-separated field.
out <- data.frame(
  feature=seq_len(p)-1L,
  weight=weights,
  bestw=rep(bestw, p),
  runtime_sec=rep(runtime, p),
  clusters=rep(paste(clusters, collapse=";"), p),
  stringsAsFactors=FALSE
)
write.csv(out, out_csv, row.names=FALSE)

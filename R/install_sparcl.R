#!/usr/bin/env Rscript
#
# Install the R dependency required by experiments/table07_spkm.py.
#
# Sparse K-means (SPKM; Witten & Tibshirani, 2010) is provided by the `sparcl`
# package. The Python driver shells out to R/_run_sparcl_spkm.R, which calls
# sparcl::KMeansSparseCluster and sparcl::KMeansSparseCluster.permute.
#
# Usage:
#     Rscript R/install_sparcl.R

repo <- "https://cloud.r-project.org"

if (!requireNamespace("sparcl", quietly = TRUE)) {
  message("Installing sparcl from CRAN ...")
  install.packages("sparcl", repos = repo)
} else {
  message("sparcl is already installed.")
}

if (!requireNamespace("sparcl", quietly = TRUE)) {
  stop("Installation failed: sparcl is still not available.")
}

cat("\n--- environment ---\n")
cat(R.version.string, "\n")
cat("sparcl", as.character(packageVersion("sparcl")), "\n")
cat("\nRecord these versions in README.md, section 'Environment'.\n")

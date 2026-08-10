suppressPackageStartupMessages({
  req_pkgs <- c(
    "optparse", "Seurat", "mclust", "aricode",
    "SingleCellExperiment", "scater", "SC3"
  )
  missing_pkgs <- req_pkgs[!vapply(req_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(
      "Missing required R packages: ",
      paste(missing_pkgs, collapse = ", "),
      "\nInstall them first."
    )
  }

  library(optparse)
  library(Seurat)
  library(mclust)
  library(aricode)
  library(SingleCellExperiment)
  library(scater)
  library(SC3)
})

flush_msg <- function(...) {
  cat(..., "\n")
  flush.console()
}

atomic_write_csv <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp")
  utils::write.csv(df, tmp, row.names = FALSE)
  ok <- file.rename(tmp, path)
  if (!ok) {
    file.copy(tmp, path, overwrite = TRUE)
    unlink(tmp)
  }
}

calc_metrics <- function(y_true, y_pred) {
  y_true <- as.character(y_true)
  y_pred <- as.character(y_pred)
  keep <- !(is.na(y_true) | is.na(y_pred))
  y_true <- y_true[keep]
  y_pred <- y_pred[keep]
  if (length(y_true) == 0L) {
    return(list(ARI = NA_real_, NMI = NA_real_, AMI = NA_real_))
  }
  list(
    ARI = as.numeric(mclust::adjustedRandIndex(y_true, y_pred)),
    NMI = as.numeric(aricode::NMI(y_true, y_pred)),
    AMI = as.numeric(aricode::AMI(y_true, y_pred))
  )
}

normalize_string <- function(x) {
  x <- as.character(x)
  trimws(x)
}

pick_label_col <- function(df) {
  cand <- c("label", "celltype", "cell_type", "predicted.id", "predicted_celltype")
  hit <- cand[cand %in% colnames(df)]
  if (length(hit) > 0L) return(hit[1])
  non_barcode <- setdiff(colnames(df), c("barcode", "cell_id", "sample_id", "X"))
  if (length(non_barcode) == 1L) return(non_barcode[1])
  stop("Could not identify label column in y file.")
}

prepare_inputs <- function(X_path, y_path, cell_ids_path = NULL, feature_names_path = NULL) {
  flush_msg("[prepare] Reading X matrix from:", X_path)
  X_df <- utils::read.csv(X_path, check.names = FALSE)

  # Drop accidental index column if present
  bad_first_names <- c("", "X", "Unnamed: 0", "...1")
  if (ncol(X_df) > 1L && colnames(X_df)[1] %in% bad_first_names) {
    suppressWarnings({
      idx_vals <- as.character(X_df[[1]])
      if (all(grepl("^[0-9]+$", idx_vals))) {
        X_df <- X_df[, -1, drop = FALSE]
      }
    })
  }

  X <- as.matrix(X_df)
  storage.mode(X) <- "double"
  rm(X_df)
  gc()

  n <- nrow(X)
  d <- ncol(X)
  flush_msg("[prepare] X shape =", paste0(n, " x ", d))

  feature_names <- colnames(X)
  if (!is.null(feature_names_path)) {
    if (file.exists(feature_names_path)) {
      feature_names <- readLines(feature_names_path, warn = FALSE)
      if (length(feature_names) == d) {
        colnames(X) <- feature_names
      } else {
        flush_msg("[prepare] feature_names_path length does not match ncol(X); using X column names.")
      }
    }
  }
  if (is.null(colnames(X)) || any(colnames(X) == "")) {
    colnames(X) <- sprintf("gene_%04d", seq_len(d))
  }

  flush_msg("[prepare] Reading labels from:", y_path)
  y_df <- utils::read.csv(y_path, stringsAsFactors = FALSE, check.names = FALSE)
  label_col <- pick_label_col(y_df)

  barcode_col <- c("barcode", "cell_id", "sample_id")
  barcode_hit <- barcode_col[barcode_col %in% colnames(y_df)]
  if (length(barcode_hit) == 0L) {
    stop("y file must contain one of: barcode, cell_id, sample_id")
  }
  barcode_col <- barcode_hit[1]

  y_df[[barcode_col]] <- normalize_string(y_df[[barcode_col]])
  y_df[[label_col]] <- normalize_string(y_df[[label_col]])

  cell_ids <- NULL
  if (!is.null(cell_ids_path)) {
    flush_msg("[prepare] Reading cell IDs from:", cell_ids_path)
    cell_ids <- utils::read.csv(cell_ids_path, stringsAsFactors = FALSE, check.names = FALSE)
    id_hit <- c("barcode", "cell_id", "sample_id")
    id_hit <- id_hit[id_hit %in% colnames(cell_ids)]
    if (length(id_hit) == 0L) {
      stop("cell_ids file must contain one of: barcode, cell_id, sample_id")
    }
    id_col <- id_hit[1]
    cell_ids[[id_col]] <- normalize_string(cell_ids[[id_col]])
    keep_barcodes <- cell_ids[[id_col]]
  } else {
    keep_barcodes <- y_df[[barcode_col]]
  }

  keep_barcodes <- unique(keep_barcodes)
  y_sub <- y_df[y_df[[barcode_col]] %in% keep_barcodes, c(barcode_col, label_col), drop = FALSE]
  y_sub <- y_sub[!duplicated(y_sub[[barcode_col]]), , drop = FALSE]

  # Assume X rows correspond to cell_ids order if provided, otherwise y order.
  if (!is.null(cell_ids)) {
    id_hit <- c("barcode", "cell_id", "sample_id")
    id_hit <- id_hit[id_hit %in% colnames(cell_ids)][1]
    if (length(cell_ids[[id_hit]]) != nrow(X)) {
      stop("nrow(X) does not match number of rows in cell_ids file.")
    }
    barcodes <- normalize_string(cell_ids[[id_hit]])
  } else {
    if (nrow(y_sub) != nrow(X)) {
      stop("Without cell_ids_path, nrow(X) must match nrow(y) after deduplication.")
    }
    barcodes <- normalize_string(y_sub[[barcode_col]])
  }

  label_map <- stats::setNames(y_sub[[label_col]], y_sub[[barcode_col]])
  y_aligned <- unname(label_map[barcodes])
  keep <- !is.na(y_aligned)

  if (!any(keep)) {
    stop("No overlapping rows between X and y (and optional cell_ids).")
  }

  if (!all(keep)) {
    flush_msg("[prepare] Dropping", sum(!keep), "rows without labels.")
    X <- X[keep, , drop = FALSE]
    barcodes <- barcodes[keep]
    y_aligned <- y_aligned[keep]
  }

  rownames(X) <- barcodes

  list(
    X = X,
    y = y_aligned,
    barcodes = barcodes,
    feature_names = colnames(X)
  )
}

run_pca_kmeans <- function(X, y, barcodes, k, pca_dim, seed) {
  set.seed(seed)
  pca_fit <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = min(pca_dim, ncol(X), nrow(X) - 1L))
  emb <- pca_fit$x[, seq_len(min(pca_dim, ncol(pca_fit$x))), drop = FALSE]
  km <- stats::kmeans(emb, centers = k, nstart = 20, iter.max = 100)
  metrics <- calc_metrics(y, km$cluster)
  list(
    pred = as.character(km$cluster),
    metrics = metrics,
    pca_var = sum((pca_fit$sdev[seq_len(ncol(emb))]^2)) / sum(pca_fit$sdev^2),
    pca = pca_fit,
    model = km
  )
}

run_pca_gmm <- function(X, y, barcodes, k, pca_dim, seed) {
  set.seed(seed)
  pca_fit <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = min(pca_dim, ncol(X), nrow(X) - 1L))
  emb <- pca_fit$x[, seq_len(min(pca_dim, ncol(pca_fit$x))), drop = FALSE]
  gmm <- mclust::Mclust(emb, G = k, verbose = FALSE)
  metrics <- calc_metrics(y, gmm$classification)
  list(
    pred = as.character(gmm$classification),
    metrics = metrics,
    pca_var = sum((pca_fit$sdev[seq_len(ncol(emb))]^2)) / sum(pca_fit$sdev^2),
    pca = pca_fit,
    model = gmm
  )
}

run_seurat_graph <- function(X, y, barcodes, k, pca_dim, seed, resolution_grid = seq(0.1, 2.0, by = 0.1)) {
  set.seed(seed)
  pca_fit <- stats::prcomp(X, center = TRUE, scale. = FALSE, rank. = min(pca_dim, ncol(X), nrow(X) - 1L))
  emb <- pca_fit$x[, seq_len(min(pca_dim, ncol(pca_fit$x))), drop = FALSE]

  dummy_counts <- matrix(1, nrow = 1, ncol = nrow(X), dimnames = list("dummy", barcodes))
  obj <- Seurat::CreateSeuratObject(counts = dummy_counts)
  obj[["pca"]] <- Seurat::CreateDimReducObject(
    embeddings = emb,
    key = "PC_",
    assay = Seurat::DefaultAssay(obj)
  )
  obj <- Seurat::FindNeighbors(obj, reduction = "pca", dims = seq_len(ncol(emb)), verbose = FALSE)

  best_pred <- NULL
  best_res <- NA_real_
  best_diff <- Inf
  best_metrics <- list(ARI = NA_real_, NMI = NA_real_, AMI = NA_real_)

  for (res in resolution_grid) {
    obj2 <- suppressMessages(Seurat::FindClusters(obj, resolution = res, random.seed = seed, verbose = FALSE))
    pred <- as.character(obj2$seurat_clusters)
    n_cl <- length(unique(pred))
    diff <- abs(n_cl - k)
    if (diff < best_diff) {
      best_diff <- diff
      best_res <- res
      best_pred <- pred
      best_metrics <- calc_metrics(y, pred)
    }
  }

  list(
    pred = best_pred,
    metrics = best_metrics,
    chosen_resolution = best_res,
    chosen_k = length(unique(best_pred)),
    pca_var = sum((pca_fit$sdev[seq_len(ncol(emb))]^2)) / sum(pca_fit$sdev^2),
    pca = pca_fit,
    model = NULL
  )
}

run_sc3_from_matrix <- function(X, y, barcodes, k, seed) {
  set.seed(seed)
  mat <- t(X)
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(logcounts = mat)
  )
  SingleCellExperiment::colData(sce)$barcode <- barcodes
  rownames(sce) <- colnames(X)
  SummarizedExperiment::rowData(sce)$feature_symbol <- rownames(sce)

  sce <- SC3::sc3(sce, ks = k, biology = FALSE, gene_filter = FALSE, svm_num_cells = NULL, n_cores = 1)
  pred_col <- paste0("sc3_", k, "_clusters")
  pred <- as.character(SummarizedExperiment::colData(sce)[[pred_col]])
  metrics <- calc_metrics(y, pred)

  list(
    pred = pred,
    metrics = metrics,
    pca_var = NA_real_,
    model = NULL
  )
}

make_summary <- function(df) {
  if (nrow(df) == 0L) return(data.frame())
  stats::aggregate(
    cbind(ARI, NMI, AMI, runtime_sec, pca_explained_variance) ~ method + k + pca_dim,
    data = df,
    FUN = function(x) c(mean = mean(x, na.rm = TRUE), sd = stats::sd(x, na.rm = TRUE))
  ) -> agg

  # flatten aggregate columns
  out <- data.frame(method = agg$method, k = agg$k, pca_dim = agg$pca_dim)
  for (nm in c("ARI", "NMI", "AMI", "runtime_sec", "pca_explained_variance")) {
    out[[paste0("mean_", nm)]] <- agg[[nm]][, "mean"]
    out[[paste0("sd_", nm)]] <- agg[[nm]][, "sd"]
  }
  out[order(-out$mean_ARI, -out$mean_NMI), ]
}

option_list <- list(
  make_option("--X_path", type = "character", default = NULL,
              help = "Path to X_pbmc_gene.csv (cells x genes)."),
  make_option("--y_path", type = "character", default = NULL,
              help = "Path to y_pbmc.csv with barcode and label columns."),
  make_option("--cell_ids_path", type = "character", default = NULL,
              help = "Optional path to cell_ids_pbmc.csv for row alignment."),
  make_option("--feature_names_path", type = "character", default = NULL,
              help = "Optional path to gene_names_pbmc.txt."),
  make_option("--k_values", type = "character", default = "6,8",
              help = "Comma-separated k values, e.g. 6,8,10"),
  make_option("--seeds", type = "character", default = "1,2,3,4,5",
              help = "Comma-separated seeds."),
  make_option("--pca_dims", type = "character", default = "10,20,30,50",
              help = "Comma-separated PCA dims for PCA/Seurat baselines."),
  make_option("--methods", type = "character", default = "seurat,pca_kmeans,pca_gmm,sc3",
              help = "Comma-separated methods to run."),
  make_option("--output_dir", type = "character", default = "pbmc_classic_baselines_csv"),
  make_option("--save_assignments", action = "store_true", default = FALSE)
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$X_path) || is.null(opt$y_path)) {
  stop("Both --X_path and --y_path are required.")
}

k_values <- as.integer(strsplit(opt$k_values, ",")[[1]])
seeds <- as.integer(strsplit(opt$seeds, ",")[[1]])
pca_dims <- as.integer(strsplit(opt$pca_dims, ",")[[1]])
methods <- trimws(strsplit(opt$methods, ",")[[1]])

outdir <- opt$output_dir
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
per_run_path <- file.path(outdir, "per_run.csv")
summary_path <- file.path(outdir, "summary.csv")
assign_path <- file.path(outdir, "cluster_assignments.csv")
config_path <- file.path(outdir, "config.json")
heartbeat_path <- file.path(outdir, "heartbeat.txt")

config <- list(
  X_path = opt$X_path,
  y_path = opt$y_path,
  cell_ids_path = opt$cell_ids_path,
  feature_names_path = opt$feature_names_path,
  k_values = k_values,
  seeds = seeds,
  pca_dims = pca_dims,
  methods = methods,
  output_dir = outdir
)
write(jsonlite::toJSON(config, auto_unbox = TRUE, pretty = TRUE), config_path)

inp <- prepare_inputs(
  X_path = opt$X_path,
  y_path = opt$y_path,
  cell_ids_path = opt$cell_ids_path,
  feature_names_path = opt$feature_names_path
)
X <- inp$X
y <- inp$y
barcodes <- inp$barcodes

flush_msg("[main] Prepared matrix:", paste(dim(X), collapse = " x "))
flush_msg("[main] Unique labels:", paste(sort(unique(y)), collapse = ", "))

rows <- list()
assign_rows <- list()
job_i <- 0L

total_jobs <- 0L
for (m in methods) {
  if (m %in% c("pca_kmeans", "pca_gmm", "seurat")) {
    total_jobs <- total_jobs + length(k_values) * length(pca_dims) * length(seeds)
  } else if (m == "sc3") {
    total_jobs <- total_jobs + length(k_values) * length(seeds)
  }
}

flush_msg("[main] Total jobs =", total_jobs)

for (method in methods) {
  if (method %in% c("pca_kmeans", "pca_gmm", "seurat")) {
    for (k in k_values) {
      for (pca_dim in pca_dims) {
        for (seed in seeds) {
          job_i <- job_i + 1L
          flush_msg(sprintf("[%d/%d] method=%s k=%d pca_dim=%d seed=%d", job_i, total_jobs, method, k, pca_dim, seed))
          t0 <- proc.time()[3]

          out <- tryCatch({
            if (method == "pca_kmeans") {
              run_pca_kmeans(X, y, barcodes, k, pca_dim, seed)
            } else if (method == "pca_gmm") {
              run_pca_gmm(X, y, barcodes, k, pca_dim, seed)
            } else {
              run_seurat_graph(X, y, barcodes, k, pca_dim, seed)
            }
          }, error = function(e) {
            list(error = conditionMessage(e))
          })

          runtime_sec <- proc.time()[3] - t0

          if (!is.null(out$error)) {
            row <- data.frame(
              method = method, k = k, pca_dim = pca_dim, seed = seed,
              ARI = NA_real_, NMI = NA_real_, AMI = NA_real_,
              runtime_sec = runtime_sec,
              pca_explained_variance = NA_real_,
              status = "failed",
              error_message = out$error,
              stringsAsFactors = FALSE
            )
          } else {
            row <- data.frame(
              method = method, k = k, pca_dim = pca_dim, seed = seed,
              ARI = out$metrics$ARI,
              NMI = out$metrics$NMI,
              AMI = out$metrics$AMI,
              runtime_sec = runtime_sec,
              pca_explained_variance = out$pca_var,
              status = "ok",
              error_message = "",
              stringsAsFactors = FALSE
            )
          }

          rows[[length(rows) + 1L]] <- row
          per_df <- do.call(rbind, rows)
          atomic_write_csv(per_df, per_run_path)
          atomic_write_csv(make_summary(per_df), summary_path)
          writeLines(sprintf("last_saved_rows=%d", nrow(per_df)), heartbeat_path)

          if (isTRUE(opt$save_assignments) && is.null(out$error)) {
            assign_rows[[length(assign_rows) + 1L]] <- data.frame(
              method = method,
              k = k,
              pca_dim = pca_dim,
              seed = seed,
              barcode = barcodes,
              pred = out$pred,
              label = y,
              stringsAsFactors = FALSE
            )
            atomic_write_csv(do.call(rbind, assign_rows), assign_path)
          }
        }
      }
    }
  } else if (method == "sc3") {
    for (k in k_values) {
      for (seed in seeds) {
        job_i <- job_i + 1L
        flush_msg(sprintf("[%d/%d] method=%s k=%d seed=%d", job_i, total_jobs, method, k, seed))
        t0 <- proc.time()[3]

        out <- tryCatch({
          run_sc3_from_matrix(X, y, barcodes, k, seed)
        }, error = function(e) {
          list(error = conditionMessage(e))
        })

        runtime_sec <- proc.time()[3] - t0

        if (!is.null(out$error)) {
          row <- data.frame(
            method = method, k = k, pca_dim = NA_integer_, seed = seed,
            ARI = NA_real_, NMI = NA_real_, AMI = NA_real_,
            runtime_sec = runtime_sec,
            pca_explained_variance = NA_real_,
            status = "failed",
            error_message = out$error,
            stringsAsFactors = FALSE
          )
        } else {
          row <- data.frame(
            method = method, k = k, pca_dim = NA_integer_, seed = seed,
            ARI = out$metrics$ARI,
            NMI = out$metrics$NMI,
            AMI = out$metrics$AMI,
            runtime_sec = runtime_sec,
            pca_explained_variance = NA_real_,
            status = "ok",
            error_message = "",
            stringsAsFactors = FALSE
          )
        }

        rows[[length(rows) + 1L]] <- row
        per_df <- do.call(rbind, rows)
        atomic_write_csv(per_df, per_run_path)
        atomic_write_csv(make_summary(per_df), summary_path)
        writeLines(sprintf("last_saved_rows=%d", nrow(per_df)), heartbeat_path)

        if (isTRUE(opt$save_assignments) && is.null(out$error)) {
          assign_rows[[length(assign_rows) + 1L]] <- data.frame(
            method = method,
            k = k,
            pca_dim = NA_integer_,
            seed = seed,
            barcode = barcodes,
            pred = out$pred,
            label = y,
            stringsAsFactors = FALSE
          )
          atomic_write_csv(do.call(rbind, assign_rows), assign_path)
        }
      }
    }
  } else {
    flush_msg("[warn] Unknown method skipped:", method)
  }
}

flush_msg("[done] Results saved to:", outdir)
flush_msg(" -", per_run_path)
flush_msg(" -", summary_path)
if (isTRUE(opt$save_assignments)) flush_msg(" -", assign_path)

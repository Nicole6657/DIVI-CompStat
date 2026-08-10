#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  req_pkgs <- c("optparse", "Seurat", "Matrix", "jsonlite")
  missing_pkgs <- req_pkgs[!vapply(req_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(
      "Missing required R packages: ", paste(missing_pkgs, collapse = ", "),
      "\nInstall them first. Required: optparse, Seurat, Matrix, jsonlite."
    )
  }
  library(optparse)
  library(Seurat)
  library(Matrix)
  library(jsonlite)
})

flush_msg <- function(...) {
  cat(..., "\n")
  flush.console()
}

safe_read_csv <- function(path) {
  if (is.null(path) || path == "") return(NULL)
  if (!file.exists(path)) stop("File not found: ", path)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

find_barcode_col <- function(df) {
  candidates <- c("barcode", "cell_id", "cell", "sample_id", "Cell", "Barcode")
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) == 0) {
    if (ncol(df) == 1) return(colnames(df)[1])
    stop("Could not find barcode column. Columns: ", paste(colnames(df), collapse = ", "))
  }
  hit[1]
}

normalize_barcode_for_match <- function(x) {
  x <- as.character(x)
  # Keep full barcode when possible; also prepare a suffix-free key.
  sub("-1$", "", x)
}

load_pbmc_object <- function(data_dir, use_seuratdata) {
  if (isTRUE(use_seuratdata)) {
    if (!requireNamespace("SeuratData", quietly = TRUE)) {
      stop("use_seuratdata=TRUE requires SeuratData. Install with remotes::install_github('satijalab/seurat-data', ref='seurat5').")
    }
    flush_msg("[load] Loading PBMC3k from SeuratData")
    suppressPackageStartupMessages(library(SeuratData))
    # InstallData errors if already installed in some versions; catch safely.
    tryCatch({
      SeuratData::InstallData("pbmc3k")
    }, error = function(e) {
      flush_msg("[load] InstallData('pbmc3k') skipped or failed; attempting LoadData('pbmc3k') directly.")
      flush_msg("       message: ", conditionMessage(e))
    })
    obj <- SeuratData::LoadData("pbmc3k")
    obj <- UpdateSeuratObject(obj)
    return(obj)
  }

  if (is.null(data_dir) || data_dir == "") {
    stop("Either --data_dir must be provided or --use_seuratdata must be set.")
  }
  if (!dir.exists(data_dir)) stop("data_dir does not exist: ", data_dir)
  flush_msg("[load] Reading 10x data from: ", data_dir)
  counts <- Read10X(data.dir = data_dir)
  obj <- CreateSeuratObject(counts = counts, project = "pbmc_sct", min.cells = 0, min.features = 0)
  return(obj)
}

option_list <- list(
  make_option("--data_dir", type = "character", default = "", help = "10x matrix directory. Not required if --use_seuratdata is set."),
  make_option("--use_seuratdata", action = "store_true", default = FALSE, help = "Load PBMC3k using SeuratData::LoadData('pbmc3k')."),
  make_option("--output_dir", type = "character", default = "/content/outputs_pbmc_sct"),
  make_option("--cell_ids_path", type = "character", default = "", help = "Optional existing cell_ids csv to enforce exact cell order/subset."),
  make_option("--y_path", type = "character", default = "", help = "Optional y csv; used only to align/subset cells if cell_ids_path not provided."),
  make_option("--n_hvg", type = "integer", default = 2000),
  make_option("--min_features", type = "integer", default = 200),
  make_option("--max_features", type = "integer", default = 2500),
  make_option("--max_percent_mt", type = "double", default = 5.0),
  make_option("--vars_to_regress", type = "character", default = "percent.mt", help = "Comma-separated variables to regress in SCTransform; use 'none' for no regression."),
  make_option("--seed", type = "integer", default = 1),
  make_option("--clip_value", type = "double", default = 10.0, help = "Clip output values to [-clip_value, clip_value]; set <=0 to disable."),
  make_option("--save_rds", action = "store_true", default = FALSE),
  make_option("--verbose", action = "store_true", default = FALSE)
)
opt <- parse_args(OptionParser(option_list = option_list))

set.seed(opt$seed)
dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

flush_msg("============================================================")
flush_msg("[start] PBMC SCTransform preprocessing for DIVI / HR-DIVI")
flush_msg("[config] output_dir       = ", opt$output_dir)
flush_msg("[config] use_seuratdata   = ", opt$use_seuratdata)
flush_msg("[config] data_dir         = ", opt$data_dir)
flush_msg("[config] n_hvg            = ", opt$n_hvg)
flush_msg("[config] QC min_features  = ", opt$min_features)
flush_msg("[config] QC max_features  = ", opt$max_features)
flush_msg("[config] QC max_percent_mt= ", opt$max_percent_mt)
flush_msg("============================================================")

# 1. Load data
obj <- load_pbmc_object(opt$data_dir, opt$use_seuratdata)
flush_msg("[1/8] raw object: cells=", ncol(obj), ", genes=", nrow(obj))

# 2. Basic QC
flush_msg("[2/8] computing percent.mt and applying QC")
obj[["percent.mt"]] <- PercentageFeatureSet(obj, pattern = "^MT-")

qc_before <- ncol(obj)
obj <- subset(
  obj,
  subset = nFeature_RNA >= opt$min_features &
    nFeature_RNA <= opt$max_features &
    percent.mt <= opt$max_percent_mt
)
flush_msg("[2/8] cells after QC: ", ncol(obj), " / ", qc_before)

# 3. Optional align/subset to existing cell ids / labels
cell_ids_df <- safe_read_csv(opt$cell_ids_path)
y_df <- safe_read_csv(opt$y_path)

if (!is.null(cell_ids_df)) {
  col <- find_barcode_col(cell_ids_df)
  requested <- as.character(cell_ids_df[[col]])
  flush_msg("[3/8] aligning to cell_ids_path with ", length(requested), " requested cells")
} else if (!is.null(y_df)) {
  col <- find_barcode_col(y_df)
  requested <- as.character(y_df[[col]])
  flush_msg("[3/8] aligning to y_path with ", length(requested), " requested cells")
} else {
  requested <- colnames(obj)
  flush_msg("[3/8] no external cell order provided; using object order")
}

obj_barcodes <- colnames(obj)
# Try exact match first.
exact_keep <- requested[requested %in% obj_barcodes]
if (length(exact_keep) == length(requested)) {
  ordered_cells <- requested
} else {
  # Try matching after stripping -1 suffix.
  obj_key <- normalize_barcode_for_match(obj_barcodes)
  req_key <- normalize_barcode_for_match(requested)
  key_to_obj <- setNames(obj_barcodes, obj_key)
  mapped <- unname(key_to_obj[req_key])
  if (any(is.na(mapped))) {
    missing_n <- sum(is.na(mapped))
    flush_msg("[3/8] WARNING: ", missing_n, " requested cells not found after exact/suffix-free matching")
    missing_examples <- requested[is.na(mapped)][seq_len(min(10, missing_n))]
    flush_msg("[3/8] missing examples: ", paste(missing_examples, collapse = ", "))
    mapped <- mapped[!is.na(mapped)]
  }
  ordered_cells <- mapped
}

if (length(ordered_cells) == 0) stop("No cells remain after cell alignment.")
obj <- subset(obj, cells = ordered_cells)
# Reorder explicitly
obj <- obj[, ordered_cells]
flush_msg("[3/8] cells after alignment: ", ncol(obj))

# 4. SCTransform
flush_msg("[4/8] running SCTransform")
vars <- trimws(unlist(strsplit(opt$vars_to_regress, ",")))
vars <- vars[vars != "" & tolower(vars) != "none"]
if (length(vars) == 0) vars <- NULL
flush_msg("[4/8] vars.to.regress = ", ifelse(is.null(vars), "none", paste(vars, collapse = ",")))

obj <- SCTransform(
  obj,
  vars.to.regress = vars,
  variable.features.n = opt$n_hvg,
  verbose = opt$verbose
)

# 5. Extract HVGs and SCT scale.data
flush_msg("[5/8] extracting SCT HVG scale.data")
hvg <- VariableFeatures(obj)
hvg <- hvg[seq_len(min(opt$n_hvg, length(hvg)))]

scale_data <- GetAssayData(obj, assay = "SCT", layer = "scale.data")
# Some Seurat versions only store variable features in scale.data; ensure HVG subset exists.
hvg <- hvg[hvg %in% rownames(scale_data)]
if (length(hvg) == 0) stop("No HVGs found in SCT scale.data.")

X <- t(as.matrix(scale_data[hvg, , drop = FALSE]))
# rows = cells, cols = genes
flush_msg("[5/8] SCT matrix dim: ", paste(dim(X), collapse = " x "))

# 6. Clean numeric matrix
flush_msg("[6/8] cleaning numeric values")
X[!is.finite(X)] <- 0
if (!is.null(opt$clip_value) && opt$clip_value > 0) {
  X[X > opt$clip_value] <- opt$clip_value
  X[X < -opt$clip_value] <- -opt$clip_value
}

# 7. Save outputs
flush_msg("[7/8] saving CSV outputs")
out_x <- file.path(opt$output_dir, "X_pbmc_sct.csv")
out_genes <- file.path(opt$output_dir, "gene_names_pbmc_sct.txt")
out_cells <- file.path(opt$output_dir, "cell_ids_pbmc_sct.csv")
out_qc <- file.path(opt$output_dir, "pbmc_sct_qc_metrics.csv")
out_meta <- file.path(opt$output_dir, "meta_pbmc_sct.json")

write.csv(X, out_x, row.names = FALSE)
write.table(colnames(X), out_genes, quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(data.frame(barcode = rownames(X), stringsAsFactors = FALSE), out_cells, row.names = FALSE)

qc <- obj@meta.data[, intersect(c("nCount_RNA", "nFeature_RNA", "percent.mt", "nCount_SCT", "nFeature_SCT"), colnames(obj@meta.data)), drop = FALSE]
qc$barcode <- rownames(qc)
write.csv(qc, out_qc, row.names = FALSE)

meta <- list(
  preprocessing = "QC + SCTransform + SCT scale.data HVGs",
  source = ifelse(opt$use_seuratdata, "SeuratData::pbmc3k", opt$data_dir),
  n_cells = nrow(X),
  n_features = ncol(X),
  n_hvg_requested = opt$n_hvg,
  n_hvg_used = length(hvg),
  min_features = opt$min_features,
  max_features = opt$max_features,
  max_percent_mt = opt$max_percent_mt,
  vars_to_regress = ifelse(is.null(vars), "none", paste(vars, collapse = ",")),
  clip_value = opt$clip_value,
  seed = opt$seed,
  output_files = list(
    X = out_x,
    gene_names = out_genes,
    cell_ids = out_cells,
    qc_metrics = out_qc
  )
)
write(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE), out_meta)

if (isTRUE(opt$save_rds)) {
  flush_msg("[7/8] saving Seurat RDS")
  saveRDS(obj, file.path(opt$output_dir, "pbmc_sct_seurat_object.rds"))
}

# 8. Done
flush_msg("[8/8] done")
flush_msg("Saved:")
flush_msg("  ", out_x)
flush_msg("  ", out_genes)
flush_msg("  ", out_cells)
flush_msg("  ", out_qc)
flush_msg("  ", out_meta)
flush_msg("============================================================")

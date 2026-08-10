suppressPackageStartupMessages({
  req_pkgs <- c("optparse", "Seurat", "SeuratData")
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
  library(SeuratData)
})

flush_msg <- function(...) {
  cat(..., "\n")
  flush.console()
}

normalize_barcode <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  sub("-1$", "", x)
}

option_list <- list(
  make_option("--data_dir", type = "character", help = "10x filtered_feature_bc_matrix directory"),
  make_option("--cell_ids_path", type = "character", help = "CSV exported by preprocess_pbmc3k_for_divi.py"),
  make_option("--output_y_path", type = "character", default = "/content/outputs_pbmc/y_pbmc.csv"),
  make_option("--output_meta_path", type = "character", default = "/content/outputs_pbmc/pbmc_mapping_metadata.csv"),
  make_option("--reference_dataset", type = "character", default = "pbmcsca", help = "SeuratData reference dataset name"),
  make_option("--reference_label_col", type = "character", default = "auto", help = "Reference metadata label column; use auto to choose"),
  make_option("--dims", type = "integer", default = 30),
  make_option("--min_features", type = "integer", default = 200),
  make_option("--min_cells", type = "integer", default = 3),
  make_option("--install_reference", action = "store_true", default = TRUE),
  make_option("--no_install_reference", dest = "install_reference", action = "store_false"),
  make_option("--allow_unmatched", action = "store_true", default = FALSE)
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$data_dir) || is.null(opt$cell_ids_path)) {
  stop("Both --data_dir and --cell_ids_path are required.")
}

dir.create(dirname(opt$output_y_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(opt$output_meta_path), recursive = TRUE, showWarnings = FALSE)

flush_msg("============================================================")
flush_msg("[start] PBMC label transfer via Seurat reference mapping")
flush_msg("[config] data_dir           =", opt$data_dir)
flush_msg("[config] cell_ids_path      =", opt$cell_ids_path)
flush_msg("[config] output_y_path      =", opt$output_y_path)
flush_msg("[config] output_meta_path   =", opt$output_meta_path)
flush_msg("[config] reference_dataset  =", opt$reference_dataset)
flush_msg("[config] reference_label_col=", opt$reference_label_col)
flush_msg("[config] dims               =", opt$dims)
flush_msg("============================================================")

flush_msg("[1/8] reading 10x data")
counts <- Read10X(data.dir = opt$data_dir)
query <- CreateSeuratObject(
  counts = counts,
  project = "pbmc_query",
  min.cells = opt$min_cells,
  min.features = opt$min_features
)
flush_msg("[1/8] query cells x genes =", ncol(query), "x", nrow(query))

flush_msg("[2/8] reading processed cell ids")
cell_ids <- read.csv(opt$cell_ids_path, stringsAsFactors = FALSE)
cn <- colnames(cell_ids)
if ("barcode" %in% cn) {
  barcode_col <- "barcode"
} else if ("cell_id" %in% cn) {
  barcode_col <- "cell_id"
} else if ("sample_id" %in% cn) {
  barcode_col <- "sample_id"
} else {
  stop("cell_ids_path must contain one of: barcode, cell_id, sample_id")
}
cell_ids$barcode <- normalize_barcode(cell_ids[[barcode_col]])

query_barcodes_norm <- normalize_barcode(colnames(query))
keep_norm <- intersect(query_barcodes_norm, cell_ids$barcode)
if (length(keep_norm) == 0) {
  stop("No overlap found between query barcodes and cell_ids_path.")
}

keep_query_idx <- which(query_barcodes_norm %in% keep_norm)
query <- subset(query, cells = colnames(query)[keep_query_idx])
query_norm_after <- normalize_barcode(colnames(query))
flush_msg("[2/8] kept processed cells =", ncol(query))

flush_msg("[3/8] normalizing query")
query <- NormalizeData(query, verbose = FALSE)
query <- FindVariableFeatures(query, verbose = FALSE)
query <- ScaleData(query, verbose = FALSE)
query <- RunPCA(query, verbose = FALSE)

flush_msg("[4/8] loading reference dataset")
if (isTRUE(opt$install_reference)) {
  tryCatch({
    SeuratData::InstallData(opt$reference_dataset)
  }, error = function(e) {
    flush_msg("[4/8] InstallData message:", conditionMessage(e))
  })
}
ref <- SeuratData::LoadData(opt$reference_dataset)
ref <- UpdateSeuratObject(ref)
flush_msg("[4/8] reference cells x genes =", ncol(ref), "x", nrow(ref))

flush_msg("[5/8] selecting reference label column")
meta_cols <- colnames(ref@meta.data)
if (opt$reference_label_col == "auto") {
  candidates <- c(
    "seurat_annotations",
    "celltype.l2",
    "celltype.l1",
    "predicted.celltype.l2",
    "predicted.celltype.l1",
    "CellType",
    "celltype"
  )
  label_hits <- candidates[candidates %in% meta_cols]
  if (length(label_hits) == 0) {
    stop("No suitable reference label column found. Available metadata columns: ",
         paste(meta_cols, collapse = ", "))
  }
  label_col <- label_hits[1]
} else {
  label_col <- opt$reference_label_col
  if (!(label_col %in% meta_cols)) {
    stop("Requested reference_label_col not found: ", label_col)
  }
}
flush_msg("[5/8] using reference label column =", label_col)

DefaultAssay(ref) <- DefaultAssay(ref)
ref <- NormalizeData(ref, verbose = FALSE)
ref <- FindVariableFeatures(ref, verbose = FALSE)
ref <- ScaleData(ref, verbose = FALSE)
ref <- RunPCA(ref, verbose = FALSE)

use_dims <- seq_len(opt$dims)
flush_msg("[6/8] finding transfer anchors")
anchors <- FindTransferAnchors(
  reference = ref,
  query = query,
  dims = use_dims,
  normalization.method = "LogNormalize",
  reference.reduction = "pca"
)

flush_msg("[7/8] transferring labels")
pred <- TransferData(
  anchorset = anchors,
  refdata = ref[[label_col, drop = TRUE]],
  dims = use_dims
)
query <- AddMetaData(query, metadata = pred)

if (!("predicted.id" %in% colnames(query@meta.data))) {
  stop("TransferData did not create predicted.id")
}

query_meta <- query@meta.data
query_meta$barcode_raw <- rownames(query_meta)
query_meta$barcode <- normalize_barcode(query_meta$barcode_raw)

flush_msg("[8/8] merging back to processed cell order")
cell_ids_out <- cell_ids
cell_ids_out$.order <- seq_len(nrow(cell_ids_out))
meta2 <- merge(
  cell_ids_out,
  query_meta,
  by = "barcode",
  all.x = TRUE,
  sort = FALSE
)
meta2 <- meta2[order(meta2$.order), , drop = FALSE]

n_unmatched <- sum(is.na(meta2$predicted.id))
flush_msg("[8/8] unmatched cells =", n_unmatched)

if (n_unmatched > 0 && !isTRUE(opt$allow_unmatched)) {
  write.csv(meta2, opt$output_meta_path, row.names = FALSE)
  stop(
    n_unmatched,
    " cells were unmatched during reference mapping. ",
    "Saved partial metadata to ", opt$output_meta_path,
    ". Rerun with --allow_unmatched if this is acceptable."
  )
}

y_out <- data.frame(
  barcode = cell_ids_out$barcode,
  label = meta2$predicted.id,
  stringsAsFactors = FALSE
)

write.csv(y_out, opt$output_y_path, row.names = FALSE)
write.csv(meta2, opt$output_meta_path, row.names = FALSE)

flush_msg("============================================================")
flush_msg("[done] Saved y to:", opt$output_y_path)
flush_msg("[done] Saved metadata to:", opt$output_meta_path)
flush_msg("============================================================")

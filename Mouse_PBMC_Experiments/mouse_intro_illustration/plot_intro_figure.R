# ==============================================================================
# Intro figure: 3-panel illustration of transfer-assisted clustering on the
# mouse PBMC dataset (Han et al. 2018 Mouse Cell Atlas, peripheral blood
# subset, 6 PeripheralBlood_<i> batches), reproducing the style of the
# paper's Figure 1 (Section 1's motivating example).
#
#   Panel 1: UMAP of the full atlas (all cells, all 6 batches, all 9 cell
#            types) -- shared across every (contrast, target, metric) run.
#   Panel 2: single-source downsampling sweep -- Target-only vs. each of the
#            5 other batches used individually as a source (oracle branches
#            only, no adaptive selection).
#   Panel 3: multi-source downsampling sweep -- Target-only vs. all 5 sources
#            pooled (no single-source line here -- that's panel 2's role;
#            picking out just one of the 5 sources to re-show here would be
#            an arbitrary/unusual choice).
#
# One run = one (CONTRAST_DIR, TARGET_BATCH, METRIC) combination -- set the
# EXPERIMENT SWITCH below and re-run, or pass them positionally on the command
# line (see generate_all_figures.sh, which loops all 24). Each run writes
# figures/<CONTRAST_DIR>/<TARGET_BATCH>_<METRIC>.{pdf,png}.
#
# Inputs (produced by the sibling Python scripts -- run those first):
#   panel1_umap.csv                                   (shared, this directory,
#                                                       from 01_umap_overview.py)
#   <CONTRAST_DIR>/results/<TARGET_BATCH>_<METRIC>.csv (from run_sweep.py)
#
# Batches are displayed as "Batch_<i>" (not "PeripheralBlood_<i>") throughout.
# Y-axis limits are NOT a fixed constant -- each figure's two sweep panels
# share a y-range computed from that figure's own data (see below).
# ==============================================================================

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        EXPERIMENT SWITCH                        ║
# ╠══════════════════════════════════════════════════════════════════╣
# Defaults below are used for interactive/manual runs (`Rscript plot_intro_figure.R`
# with no arguments). For batch-generating all 24 combinations, pass them
# positionally instead: `Rscript plot_intro_figure.R <CONTRAST_DIR> <TARGET_BATCH> <METRIC>`
# -- see generate_all_figures.sh.
args <- commandArgs(trailingOnly = TRUE)

CONTRAST_DIR  <- if (length(args) >= 1) args[1] else "B_cell_vs_Macrophage"   # or "T_cell_vs_NK_cell"
TARGET_BATCH  <- if (length(args) >= 2) args[2] else "PeripheralBlood_1"      # or _2 .. _6
METRIC        <- if (length(args) >= 3) args[3] else "ari"                   # or "misclustering"

# Every figure lands in figures/<CONTRAST_DIR>/, one subfolder per contrast.
OUT_DIR <- file.path("figures", CONTRAST_DIR)
# ╚══════════════════════════════════════════════════════════════════╝

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(latex2exp)
})

## ---- Shared base theme ------------------------------------------------------
## IMPORTANT: `pt`/`cm` sizes in a ggplot theme are ABSOLUTE physical units
## on the rendered PDF page, not relative to the ggsave() canvas size --
## these sizes are calibrated for the 30x10in canvas set below.
## aspect.ratio = 1 lives here (not per-panel) so all three panels -- the UMAP
## and both sweep panels -- render as the same square shape instead of the
## sweep panels stretching wide while only the UMAP stayed square.
base_theme <- theme_classic(base_size = 13) +
  theme(
    aspect.ratio     = 1,
    plot.title       = element_text(hjust = 0.5, size = 15, face = "plain"),
    plot.subtitle    = element_text(hjust = 0.5, size = 13, color = "grey30"),
    axis.title       = element_text(size = 18),
    axis.text        = element_text(size = 15),
    axis.ticks       = element_line(linewidth = 0.3),
    axis.line        = element_line(linewidth = 0.4),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin      = margin(3, 3, 3, 3, "pt")
  )

## ---- Batch display-name mapping ---------------------------------------------
## PeripheralBlood_<i> -> Batch_<i>, applied to every axis/legend/title label
## involving a batch identity (never to file paths or CSV lookups).
batch_display <- function(x) gsub("PeripheralBlood_", "Batch_", x, fixed = TRUE)

cat("Contrast:", gsub("_", " ", CONTRAST_DIR), " Target:", batch_display(TARGET_BATCH),
    " Metric:", METRIC, "\n")

METRIC_LABEL <- if (METRIC == "ari") "Adjusted Rand Index" else "Misclustering loss"

## ---- Panel 1: full-atlas UMAP ------------------------------------------------
## Hand-picked 9-color palette, chosen for maximum distinctness between
## the 9 cell types.
CELL_TYPE_COLORS <- c(
  "B cell"         = "#1F77B4",
  "Basophil"       = "#FF7F0E",
  "Dendritic cell" = "#9467BD",
  "Erythroblast"   = "#D62728",
  "Macrophage"     = "#2CA02C",
  "Monocyte"       = "#8C564B",
  "NK cell"        = "#E377C2",
  "Neutrophil"     = "#7F7F7F",
  "T cell"         = "#17BECF"
)

set.seed(42)
df1 <- read.csv("panel1_umap.csv")
df1 <- df1[sample(nrow(df1)), ]   # shuffle draw order so no cell type is systematically on top
df1$cell_type <- factor(df1$cell_type, levels = names(CELL_TYPE_COLORS))

p1 <- ggplot(df1, aes(x = UMAP1, y = UMAP2, color = cell_type)) +
  geom_point(size = 1.1, stroke = 0, alpha = 0.8) +
  scale_color_manual(values = CELL_TYPE_COLORS, name = NULL, drop = TRUE) +
  ## Wrapped onto two lines: at this font size a one-line title risks
  ## running past this square panel's own column (which also has to leave
  ## room for the legend to its right).
  labs(title = "Full atlas\n(6 batches, 9 cell types)", x = "UMAP 1", y = "UMAP 2") +
  ## coord_fixed(), not theme(aspect.ratio=1) (which base_theme sets for the
  ## other two panels), because aspect.ratio + a right-hand legend is a known
  ## ggplot2 gtable quirk: the panel gets sized as if the legend weren't
  ## there, leaving a large stray blank gutter before the axis. coord_fixed()
  ## still yields a square panel here (UMAP1/UMAP2 ranges are nearly equal)
  ## without that quirk.
  coord_fixed(ratio = 1) +
  base_theme +
  theme(
    aspect.ratio    = NULL,
    legend.text     = element_text(size = 30),
    legend.key.size = unit(0.9, "cm"),
    legend.position = "right"
  ) +
  guides(color = guide_legend(override.aes = list(size = 8), ncol = 1))

## ---- Shared color mapping for panels 2 & 3 ---------------------------------
## Same entity -> same color across both panels (batch color follows the
## BATCH identity, not its role as target/source).
BATCH_COLORS <- c(
  "PeripheralBlood_1" = "#ff7f0e",
  "PeripheralBlood_2" = "#2ca02c",
  "PeripheralBlood_3" = "#d62728",
  "PeripheralBlood_4" = "#8c564b",
  "PeripheralBlood_5" = "#e377c2",
  "PeripheralBlood_6" = "#bcbd22"
)
TARGET_ONLY_COLOR  <- "#000000"  # black, drawn on top as the baseline every other method is compared against
MULTI_SOURCE_COLOR <- "#4a3aa7"  # violet

method_color <- function(method_str) {
  if (method_str == "Target-only") return(TARGET_ONLY_COLOR)
  if (grepl("Multi-source", method_str, fixed = TRUE)) return(MULTI_SOURCE_COLOR)
  for (b in names(BATCH_COLORS)) {
    if (grepl(b, method_str, fixed = TRUE)) return(BATCH_COLORS[[b]])
  }
  stop(paste("method_color(): no color rule matches method label:", method_str))
}

## Order: Target-only, then sources in BATCH_COLORS order, then multi-source.
method_rank <- function(method_str) {
  if (method_str == "Target-only") return(0)
  if (grepl("Multi-source", method_str, fixed = TRUE)) return(99)
  b_idx <- which(sapply(names(BATCH_COLORS), function(b) grepl(b, method_str, fixed = TRUE)))
  if (length(b_idx) == 0) stop(paste("method_rank(): no batch matches method label:", method_str))
  b_idx[1]
}

## Relabel "Source: PeripheralBlood_i" / method strings for display, swapping
## in Batch_i -- done last, after color/rank lookups (which key off the raw
## PeripheralBlood_i strings from the CSV).
method_display <- function(method_str) batch_display(method_str)

## ---- Sweep plot helper (panels 2 & 3 share this shape) ---------------------
## x-axis labeled $n_T$ via latex2exp::TeX(). No ribbon/error bars -- mean
## lines only. y-axis is NOT a fixed constant -- each FIGURE (one target x
## contrast x metric) gets its own y-range, computed from that figure's own
## data's min/max (with a little padding) and passed in as `y_limits` so
## panel 2 and panel 3 stay directly comparable to each other; a different
## target/contrast/metric gets a different range, sized to its own data.
## `legend_nrow` is chosen per call site -- panel 2 shows 6 methods
## (target-only + 5 sources) vs. panel 3's 2, and at this legend's font/key
## size a 3-column layout (nrow=2 for 6 items) is wider than panel 2's own
## column and bleeds into panel 3's; more rows/fewer columns keeps each
## panel's legend within its own column width.
plot_metric_sweep <- function(df, title, y_limits, legend_nrow) {
  methods_present <- unique(df$method)
  methods_present <- methods_present[order(sapply(methods_present, method_rank))]
  method_colors   <- setNames(sapply(methods_present, method_color), methods_present)
  method_labels   <- setNames(sapply(methods_present, method_display), methods_present)

  df <- df %>% mutate(method = factor(method, levels = methods_present))
  k_breaks <- sort(unique(df$k))

  ggplot(df, aes(k, mean_value, color = method)) +
    geom_line(linewidth = 1.8) +
    geom_point(size = 4.5) +
    scale_x_log10(breaks = k_breaks) +
    scale_color_manual(values = method_colors, labels = method_labels, drop = TRUE) +
    coord_cartesian(ylim = y_limits) +
    labs(title = title, x = TeX("$n_T$"), y = METRIC_LABEL) +
    base_theme +
    theme(legend.position = "bottom", legend.title = element_blank(),
          legend.justification = "left",
          legend.margin = margin(t = 2, r = 0, b = 0, l = 8),
          legend.text = element_text(size = 33), legend.key.size = unit(1.2, "cm")) +
    guides(color = guide_legend(nrow = legend_nrow))
}

## ---- Panels 2 & 3: single-source and multi-source downsampling -------------
df_raw <- read.csv(file.path(CONTRAST_DIR, "results", paste0(TARGET_BATCH, "_", METRIC, ".csv")))
df2 <- df_raw %>% filter(method == "Target-only" | grepl("^Source: ", method))
df3 <- df_raw %>% filter(method == "Target-only" | method == "Multi-source (pooled)")

## Shared y-range for this one figure's two sweep panels, from df_raw (the
## superset of everything either panel can show) -- 5% padding, floored at 0
## for misclustering (never negative by construction).
data_range <- range(df_raw$mean_value)
pad <- diff(data_range) * 0.05
y_limits <- c(data_range[1] - pad, data_range[2] + pad)
if (METRIC == "misclustering") y_limits[1] <- max(0, y_limits[1])

## Extra right margin on p2 / left margin on p3 opens a small horizontal gap
## between the two sweep panels, which otherwise sit flush against each
## other.
p2 <- plot_metric_sweep(df2, "Single source vs. target-only", y_limits, legend_nrow = 3) +
  theme(plot.margin = margin(3, 20, 3, 3, "pt"))
p3 <- plot_metric_sweep(df3, "Multi-source pooling vs. target-only", y_limits, legend_nrow = 2) +
  theme(plot.margin = margin(3, 3, 3, 20, "pt"))

## ---- Combine and save -------------------------------------------------------
## Panels 2 & 3 each keep their own legend below their own panel (each panel's
## legend.justification = "left", set inside plot_metric_sweep(), anchors it
## to that SAME panel's own left boundary); panel 1 uses an unrelated color
## scale (cell type) and keeps its own (right-side) legend.
## Flattened to three columns (not nested) so each square panel gets its
## own share of the page width -- a nested layout under-allocates panel
## 1's width.
fig <- (p1 | p2 | p3) +
  plot_layout(widths = c(0.75, 1, 1)) +
  plot_annotation(
    theme = theme(plot.background = element_rect(fill = "white", color = NA))
  )

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
# Contrast is already encoded by the subfolder (figures/<CONTRAST_DIR>/); the
# filename itself just needs target + metric to stay unique within it.
base_name <- file.path(OUT_DIR, paste0(TARGET_BATCH, "_", METRIC))

## 30x10in canvas -- the font/legend/point sizes above are calibrated for it.
ggsave(paste0(base_name, ".pdf"), fig, width = 30, height = 10, units = "in",
       bg = "white", device = "pdf")
ggsave(paste0(base_name, ".png"), fig, width = 30, height = 10, units = "in",
       dpi = 300, bg = "white")
cat("Saved:", paste0(base_name, ".{pdf,png}"), "\n")

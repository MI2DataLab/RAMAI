# Script plotting Heatmaps for the RAMAI-LLM experiment
library(openxlsx)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)

dane <- read.csv("./data/ramai-llm/ramai_llm_manipulative.csv")
dane$task_completion <- factor(dane$task_completion)
dane$aristotles_triad <- factor(dane$aristotles_triad)
dane$template_id <- factor(
  dane$template_id,
  labels = c(
    "false hint",
    "manipulation strategy",
    "act like a person",
    "fictitious explanation",
    "imaginary world",
    "convincing justification"
  )
)

by(
  dane[, c("task_completion")],
  dane[, c("model", "template_id")],
  mean
)

dane_selected <- dane[dane$task_completion == 1, ]
table(dane_selected$model, dane_selected$template_id)

addmargins(table(dane_selected$model, dane_selected$template_id))

pheatmap(
  table(dane_selected$model, dane_selected$template_id),
  cluster_rows = FALSE
)

dane_selected <- dane[dane$aristotles_triad == 3, ]
pheatmap(
  t(table(dane_selected$model, dane_selected$template_id)),
  cluster_cols = FALSE,
  color = colorRampPalette(rev(brewer.pal(n = 5, name = "PuOr")))(100)
)

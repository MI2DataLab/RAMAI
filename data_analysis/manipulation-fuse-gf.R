# Script plotting generator-fuse performance
library(ggplot2)
library(tidyr)

for (context_setting in c("low", "high")) {
  df_long <- list()
  for (metric in c("fn", "fp", "tn", "tp")) {
    path <- paste0(
      "./data/manipulation-fuse/confusion/",
      context_setting,
      "-context_",
      metric,
      ".csv"
    )
    df_long[[metric]] <- pivot_longer(
      read.table(path, sep = ",", h = TRUE),
      cols = -X
    )
    colnames(df_long[[metric]]) <- c("Fuse", "Generator", toupper(metric))
  }

  long <- cbind(
    df_long[["fn"]], df_long[["fp"]], df_long[["tn"]], df_long[["tp"]]
  )[, c(1, 2, 3, 6, 9, 12)]

  long$Precision <- long$TP / (long$TP + long$FP)
  long$Recall    <- long$TP / (long$TP + long$FN)
  long$N <- long$TP + long$FN + long$FP + long$TN
  long$PP <- (long$TP + long$FN) / long$N
  long$P <- (long$TP + long$FP) / long$N

  long$Generator <- factor(long$Generator)
  levels(long$Generator) <- c(
    "Dolphin (43%)",
    "Gemini-Pro (53%)",
    "GPT-3.5-turbo (64%)",
    "GPT-4 (75%)",
    "Mixtral-8x7B (18%)"
  )
  long$Fuse <- factor(long$Fuse)
  levels(long$Fuse) <- c(
    "Dolphin",
    "Gemini-Pro",
    "GPT-3.5-turbo",
    "GPT-4",
    "Mixtral-8x7B"
  )

  plt <- ggplot(long, aes(Recall, Precision, color = Fuse, shape = Generator)) +
    geom_point(size = 5) +
    scale_color_manual(values = c(
      "Mixtral-8x7B" = "#CA0020",
      "Dolphin" = "#0571B0",
      "GPT-3.5-turbo" = "#15D148",
      "GPT-4" = "#0B4D1D",
      "Gemini-Pro" = "#D4C60D"
    )) +
    scale_x_continuous(
      "Recall TP/(TP+FN)", labels = scales::percent_format()
    ) +
    scale_y_continuous(
      "Precision TP/(TP+FP)", labels = scales::percent_format()
    ) +
    scale_shape_manual("Generator", values = LETTERS) +
    theme_bw()

  path <- paste0(
    "./plots/gf_",
    context_setting,
    "-context.png"
  )
  plot_path <- paste0(path)
  ggsave(plot_path, plt, device = "png", height = 4, width = 6)
}

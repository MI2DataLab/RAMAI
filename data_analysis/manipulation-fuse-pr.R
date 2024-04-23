# Script plotting Manipulation Fuses PR
library(ggplot2)

data <- read.csv("./data/manipulation-fuse/classifier_pr.csv")
data$Setting <- as.factor(data$Setting)

plt <- ggplot(
  data,
  aes(x = Recall, y = Precision, color = Model, shape = Setting)
) +
  geom_point(size = 7, stroke = 2, alpha = 0.8) +
  scale_color_manual(values = c(
    "Mixtral-8x7B" = "#CA0020",
    "Dolphin" = "#0571B0",
    "GPT-3.5-turbo" = "#15D148",
    "GPT-4" = "#0B4D1D",
    "Gemini-Pro" = "#D4C60D"
  )) +
  scale_shape_manual(values = c(1, 2, 3)) +
  labs(title = "Precision vs. Recall", x = "Recall", y = "Precision") +
  theme_minimal() +
  ggtitle("") +
  scale_shape(solid = TRUE) +
  scale_x_continuous(limits = c(0.25, 1)) +
  scale_y_continuous(limits = c(0.4, 0.7)) +
  theme(text = element_text(size = 14))

plot_path <- paste0("./plots/", "classifier_pr.svg")
ggsave(plot_path, plt, device = "svg", height = 4, width = 6)

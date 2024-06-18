# Stat tests and plots for RAMAI-LLM
library(dplyr)
library(ggradar)
library(fmsb)
library(ggplot2)
library(tidyr)

# Load data
liwc_truthful <- read.csv(
  "./data/ramai-llm/liwc_truthful.csv", sep = ","
)
liwc_manipulative <- read.csv(
  "./data/ramai-llm/liwc_manipulative.csv", sep = ","
)

# Defined categories
categories <- c(
  "Analytical",
  "Emotionality",
  "Abstraction",
  "Word Count",
  "Self-references",
  "Certainty",
  "Hedges",
  "Lexical Diversity",
  "Reading Difficulty"
)
cat_labels <- c()
significant <- c()

# Arrange and group data
liwc_truthful <- liwc_truthful %>%
  arrange(model, question_id)
liwc_manipulative <- aggregate(
    . ~ model + question_id,
    liwc_manipulative,
    mean
  ) %>% arrange(model, question_id)
colnames(liwc_truthful) <- c(
  "chat_no", "model", "template_id", "question_id", categories
)
colnames(liwc_manipulative) <- c(
  "model", "question_id", "chat_no", "template_id", categories
)
liwc_manipulative["group"] <- "Manipulative"
liwc_truthful["group"] <- "Truthful"

# Print p-values
for (category in categories) {
  cat(paste0("Category: ", category, "\n"))
  pval <- t.test(
    liwc_truthful[[category]],
    liwc_manipulative[[category]],
    paired = TRUE
  )$p.value
  pval_str <- formatC(round(pval, 3), 3, format="f")
  cat_labels <- c(cat_labels, paste0(category, " (", pval_str, ")"))
  cat(paste0("p-value: ", pval, "\n"))
  cat("---\n")
  if (pval < 0.05) {
    significant <- c(significant, category)
  }
}

# Save data for plotting boxplots
liwc_all <- rbind(liwc_manipulative, liwc_truthful) %>% 
  select(model, group, all_of(categories)) %>%
  mutate(across(where(is.numeric), ~(. - min(.)) / (max(.) - min(.))))

# Prepare data to plot radar
liwc_df <- rbind(liwc_manipulative, liwc_truthful) %>% 
  select(group, all_of(categories)) %>%
  mutate(across(where(is.numeric), ~(. - min(.)) / (max(.) - min(.))))
liwc_df <- aggregate(. ~ group, liwc_df, mean) %>% 
  select(all_of(categories))
rownames(liwc_df) <- c("Manipulative", "Truthful")

liwc_maxmin <- data.frame(
  Analytical = c(0.75, 0),
  Emotionality = c(0.75, 0),
  Abstraction = c(0.75, 0),
  Word.Count = c(0.75, 0),
  Self.reference = c(0.75, 0),
  Certainty = c(0.75, 0),
  Hedges = c(0.75, 0),
  Lexical.Diversity = c(0.75, 0),
  Reading.Difficulty = c(0.75,0)
)
colnames(liwc_maxmin) <- colnames(liwc_df)
rownames(liwc_maxmin) <- c("Max", "Min")
liwc_radar <- rbind(liwc_maxmin, liwc_df)

# Plot - radar
colors_border <- c(rgb(0.85, 0.37, 0.1, 0.9), rgb(0.35, 0.35, 0.9, 0.9))
colors_in <- c(rgb(1, 0.75, 0.3, 0.4), rgb(0.65, 0.74, 0.86, 0.6))
plot_path <- paste0("./plots/", "radar.svg")
svg(filename = plot_path, width = 5, height = 5)
radarchart(
  liwc_radar,
  axistype = 4,
  seg = 3,
  pcol = colors_border,
  pfcol = colors_in,
  plwd = 2,
  plty = 1,
  cglcol = "grey",
  cglty = 1,
  axislabcol = "black",
  caxislabels = c("0.00", "0.25", "0.50", "0.75"),
  calcex = 0.65,
  cglwd = 1,
  vlcex = 0.8,
  vlabels = cat_labels,
)
legend(
  x = 1, y = 1,
  legend = rownames(liwc_radar[-c(1, 2),]),
  bty = "n", pch = 20, col = colors_in,
  text.col = "black", cex = 0.8, pt.cex = 3
)
dev.off()

# Plot - box plots
df_long <- pivot_longer(
  liwc_all, cols = -c(model, group),
  names_to = "Variable", values_to = "Value"
)
df_long <- df_long %>%
  filter(Variable %in% significant)

plt <- ggplot(df_long, aes(x = Value, y = model, fill = group, col = group)) +
  geom_boxplot(outlier.colour = NULL) +
  facet_wrap(~ Variable, ncol = 2) +
  labs(x = "", y = "", title = "") +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.spacing = unit(1.4, "lines")
  ) +
  scale_fill_manual(values = colors_in) +
  scale_color_manual(values = colors_border) +
  theme(text = element_text(size = 8))

plot_path <- paste0("./plots/", "boxplots.svg")
ggsave(plot_path, plt, device = "svg", height = 5, width = 4)

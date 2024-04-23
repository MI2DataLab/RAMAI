# Script plotting demographics data
library(ggplot2)
library(dplyr)
library(ggpubr)

df <- read.csv("./data/ramai-human/demographics.csv")

df$sex <- factor(df$sex, levels = c("female", "male"))
df$age <- factor(df$age, levels = c("0-18", "19-26", "27-39", "40+"))
df$education <- factor(df$education, levels = c(
  "< h. school",
  "h. school",
  "bachelor",
  "master+"
))

p1 <- ggplot(df, aes(x = sex)) +
  geom_bar(fill=rgb(1, 0.75, 0.3)) +
  labs(x = "Sex", y = "Count") +
  facet_grid(rows = vars(group), scales="free_y") +
  theme_minimal()

p2 <- ggplot(df, aes(x = age)) +
  geom_bar(fill=rgb(1, 0.75, 0.3)) +
  labs(x = "Age", y = "Count") +
  facet_grid(rows = vars(group), scales="free_y") +
  theme_minimal()

p3 <- ggplot(df, aes(x = education)) +
  geom_bar(fill=rgb(1, 0.75, 0.3)) +
  labs(x = "Education", y = "Count") +
  facet_grid(rows = vars(group), scales="free_y") +
  theme_minimal()

plt <- ggarrange(p1, p2, p3, nrow = 1, ncol = 3)

plot_path = paste0("./plots/", "demographics.png")
ggsave(plot_path, plt, device = "png", width = 10)
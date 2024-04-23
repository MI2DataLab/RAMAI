# Stat tests for RAMAI-Human
library(lme4)
library(car)
library(knitr)

# Load data
dfs <- list(
  hint_trusted = read.csv("./data/ramai-human/hint_trusted.csv"),
  manipulation_detected = read.csv("./data/ramai-human/manipulation_detected.csv")
)

# Set features and targets
features <- c(
  "history_hint_correct", "hint_density",
  "last_hint_correct", "group", "sex", "age", "education"
)
targets <- c("hint_trusted", "manipulation_detected")


for (target in targets){
  # Change data types
  dfs[[target]]$game_id <- as.character(dfs[[target]]$game_id)
  dfs[[target]][,target] <- as.logical(dfs[[target]][,target])
  
  # Fit Linear Mixed-Effects Model
  formula_str <- paste(target, "(1 | game_id)", sep = " ~ ")
  formula_str <- paste(c(formula_str, features), collapse = " + ")
  lmm <- lmer(formula_str, REML = TRUE, data = dfs[[target]])
  
  # Calculate F statistics and p-values with K-R approximation
  anova_stats <- Anova(lmm, test.statistic="F")
  p_values <- anova_stats[, "Pr(>F)"]
  
  # Bonferroni correction
  num_tests <- 2*length(p_values)
  bonferroni_correction <- p.adjust(
    p_values,
    method = "fdr",
    n = num_tests
  )
  
  # Create results data frame
  results = data.frame(
    variable = rownames(anova_stats),
    fixef = fixef(lmm)[-1],
    F = anova_stats$F,
    p = bonferroni_correction
  )
  
  # Create LaTeX table
  print(kable(
    results,
    caption = "",
    format = "latex",
    booktabs = TRUE
  ))
}
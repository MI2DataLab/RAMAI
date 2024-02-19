library(sjPlot)
library(sjmisc)
library(effects)
library(ggplot2)
library(knitr)
library(dplyr)

dfs <- list(
  followed = read.csv("./data/lr/followed.csv"),
  lie_detected = read.csv("./data/lr/detection.csv")
)

features <- c(
  "history_hint_correct", "hint_density",
  "last_hint_correct", "group", "sex", "age", "education"
)
features_categorical <- c("age", "education")
targets <- c("followed", "lie_detected")

results <- list(
  followed = list(
    single = data.frame(
      variable = character(0),
      coef = numeric(0),
      p = numeric(0)
    ),
    important = data.frame()
  ),
  lie_detected = list(
    single = data.frame(
      variable = character(0),
      coef = numeric(0),
      p = numeric(0)
    ),
    important = data.frame()
  )
)

for (target in targets){
  
  # Single variable models
  for (feature in features){
    formula_str <- paste(target, feature, sep = " ~ ")
    model <- glm(
      as.formula(formula_str),
      data = dfs[[target]],
      family = binomial()
    )
    if (feature %in% features_categorical){
      formula_str <- paste(target, "1", sep = " ~ ")
      model_baseline <- glm(
        formula_str,
        data = dfs[[target]],
        family = binomial()
      )
      anova_results <- anova(model, model_baseline, test = "Chisq")
      coef <- ""
      p <- anova_results$`Pr(>Chi)`[2]
    } else {
      summary_tab <- summary(model)
      coef <- summary_tab$coefficients[, "Estimate"][2]
      p <- summary_tab$coefficients[, "Pr(>|z|)"][2]
    }
    results[[target]][["single"]] <- rbind(
      results[[target]][["single"]],
      data.frame(
        variable = feature,
        coef = coef,
        p = p
      )
    )
  }
  
  # Important variable models
  important_variables <- results[[target]][["single"]]$variable[results[[target]][["single"]]$p < 0.05]
  df_important <- dfs[[target]][c(target, important_variables)]
  formula_str <- paste(target, ".", sep = " ~ ")
  lr_important <- glm(formula_str, df_important, family=binomial())
  summary_tab <- summary(lr_important)
  results[[target]][["important"]] <- data.frame(
    variable_name = names(coef(lr_important))[-1],
    coef = coef(lr_important)[-1],
    p = summary_tab$coefficients[, "Pr(>|z|)"][-1]
  )
  
  # Forest plots
  plt <-plot_model(
    lr_important,
    transform      = NULL,
    show.values    = TRUE,
    show.intercept = T,
    value.offset   = .4,
    vline.color = "black"
  ) +
    ggtitle("") +
    theme_minimal()
  
  plot_path = paste0("./plots/", target, "_forest.png")
  ggsave(plot_path, plt, device = "png", height=2.7, width=3.5)
  
  # Interaction plots
  plt <- ggplot() +
    aes(x = (dfs[[target]]["history_hint_correct"] > 0.5), color = (dfs[[target]]["hint_density"] > 0.5), group = (dfs[[target]]["hint_density"] > 0.5), y = as.numeric(dfs[[target]][target] == 1)) +
    stat_summary(fun.y = mean, geom = "point") +
    stat_summary(fun.y = mean, geom = "line") +
    coord_cartesian(ylim = c(0.4, 0.75)) +
    labs(x = "history_hint_correct", y = paste("Mean of", target), color = "hint_density") +
    theme(legend.position = c(0.5, 0.2)) +
    theme_minimal()
  
  plot_path = paste0("./plots/", target, "_interaction.png")
  ggsave(plot_path, plt, device = "png", height=2, width=3)
  
  # LaTeX tables
  cat(paste0(
    "\n\n\n========== ", target, " (single) ==========\n"
  ))
  latex <- kable(
    results[[target]][["single"]],
    caption = "",
    format = "latex",
    booktabs = TRUE
  )
  print(latex)
  
  cat(paste0(
    "\n\n\n========== ", target, " (important) ==========\n"
  ))
  latex <- kable(
    results[[target]][["important"]],
    caption = "",
    format = "latex",
    booktabs = TRUE
  )
  print(latex)
}


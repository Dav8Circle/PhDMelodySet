# Load required libraries
library(tidyverse)
library(lme4)
library(ggplot2)

# Load in the original item bank file
item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses and aggregate at melody level
df2 <- read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "guessing", "inattention")), by = "item_id") %>%
  group_by(item_id) %>%
  summarise(
    score_item = mean(score),
    oddity = first(oddity),
    displacement = first(displacement),
    contour_dif = first(contour_dif),
    in_key = first(in_key),
    change_note = first(change_note),
    num_notes = first(num_notes),
    ability_WL = mean(ability_WL),# Take mean of ability_WL for each melody
    difficulty = first(difficulty)
  )

# Load the musical features
features <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv") %>%
  select(-matches("\\{.*\\}")) %>%
  select(where(is.numeric))

# Store melody_ids before removing them for analysis
melody_ids <- features$melody_id
features <- features %>% select(-melody_id)

# Remove zero variance columns
zero_var_cols <- features %>%
  summarise(across(everything(), var)) %>%
  pivot_longer(everything()) %>%
  filter(value == 0) %>%
  pull(name)

# Drop zero variance and tempo features
features <- features %>%
  select(-starts_with("duration_features.tempo")) %>%
  select(-any_of(zero_var_cols))

# Scale features
features_scaled <- scale(features)
features_scaled <- features_scaled[, colSums(is.na(features_scaled)) == 0]

# First, perform PCA on the musical features
pca_result <- prcomp(features_scaled)

# Determine number of components to keep (e.g., explaining 80% variance)
var_explained <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
n_components <- which(var_explained >= 0.80)[1]

# Extract PC scores for selected components
pc_scores <- pca_result$x[, 1:n_components]
colnames(pc_scores) <- paste0("PC", 1:n_components)

# Create combined dataset
combined_data <- data.frame(
  melody_id = melody_ids,
  pc_scores
) %>%
  left_join(df2, by = c("melody_id" = "item_id")) %>%
  na.omit()  # Remove any rows with missing values

# First, recreate reg8b for comparison
reg8b <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
              in_key + change_note + ability_WL + num_notes,
            data = combined_data)

# Fit combined model with only significant predictors
reg_combined <- lm(score_item ~ as.factor(oddity) + displacement + 
                    in_key + ability_WL + num_notes +
                    PC2 + PC3 + PC4 + PC8 + PC9 + PC12 + PC13 + PC14,
                  data = combined_data)

# Print model summaries
cat("\nOriginal Model (reg8b) Summary:\n")
print(summary(reg8b))

cat("\nCombined Model (Significant Predictors Only) Summary:\n")
print(summary(reg_combined))

# Calculate improvement in R-squared
r2_improvement <- summary(reg_combined)$r.squared - summary(reg8b)$r.squared
cat("\nR-squared improvement:", round(r2_improvement, 4))

# ANOVA comparison
cat("\nANOVA Comparison of Models:\n")
print(anova(reg8b, reg_combined))

# Create comparison plots
# 1. Predicted vs Actual
plot_data <- data.frame(
  Actual = combined_data$score_item,
  Reg8b_Pred = predict(reg8b, combined_data),
  Combined_Pred = predict(reg_combined, combined_data)
) %>%
  pivot_longer(cols = c(Reg8b_Pred, Combined_Pred),
               names_to = "Model",
               values_to = "Predicted")

p1 <- ggplot(plot_data, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(
    title = "Model Comparison: Predicted vs Actual Values",
    x = "Actual Score",
    y = "Predicted Score"
  )

# 2. Residual plots
residuals_data <- data.frame(
  Reg8b_Residuals = residuals(reg8b),
  Combined_Residuals = residuals(reg_combined),
  Fitted_Reg8b = fitted(reg8b),
  Fitted_Combined = fitted(reg_combined)
) %>%
  pivot_longer(
    cols = c(Reg8b_Residuals, Combined_Residuals),
    names_to = "Model",
    values_to = "Residuals"
  ) %>%
  mutate(
    Fitted = ifelse(Model == "Reg8b_Residuals", 
                    Fitted_Reg8b, Fitted_Combined)
  )

p2 <- ggplot(residuals_data, aes(x = Fitted, y = Residuals, color = Model)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  theme_minimal() +
  labs(
    title = "Residual Plots",
    x = "Fitted Values",
    y = "Residuals"
  )

# 3. Feature importance plot for combined model
coef_data <- data.frame(
  Feature = names(coef(reg_combined))[-1],  # Remove intercept
  Coefficient = coef(reg_combined)[-1],
  Significance = summary(reg_combined)$coefficients[-1, 4]
) %>%
  mutate(
    Feature = gsub("as.factor\\(oddity\\)", "Oddity ", Feature),
    Significant = Significance < 0.05
  ) %>%
  arrange(desc(abs(Coefficient)))

p3 <- ggplot(coef_data, aes(x = reorder(Feature, abs(Coefficient)), 
                           y = Coefficient,
                           fill = Significant)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Feature Importance in Combined Model (Significant Predictors Only)",
    x = "Feature",
    y = "Coefficient",
    fill = "p < 0.05"
  ) +
  scale_fill_manual(values = c("grey70", "steelblue"))

# 4. Variance explained by significant PCs
significant_pcs <- c(2, 3, 4, 8, 9, 12, 13, 14)
p4 <- data.frame(
  PC = significant_pcs,
  Variance = var_explained[significant_pcs] - c(0, head(var_explained[significant_pcs], -1))
) %>%
  ggplot(aes(x = factor(PC), y = Variance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(
    title = "Individual Variance Explained by Significant PCs",
    x = "Principal Component",
    y = "Proportion of Variance"
  )

# Arrange plots
gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)

# Save the models and results
saveRDS(list(
  reg8b = reg8b,
  reg_combined = reg_combined,
  improvement = r2_improvement,
  feature_importance = coef_data,
  significant_pcs = significant_pcs,
  var_explained = var_explained[significant_pcs]
), "combined_model_results.rds") 

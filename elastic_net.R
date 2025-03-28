# Load required libraries
library(tidyverse)
library(caret)
library(glmnet)
library(ggplot2)
library(gridExtra)

# Load in the original item bank file
item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses
df <- 
  read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")

# Load the features and filter for numeric columns
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
features <- scale(features)
features <- features[, colSums(is.na(features)) == 0]

# Create train/test split indices (80/20)
set.seed(123)
train_idx <- sample(1:nrow(features), 0.8 * nrow(features))
test_idx <- setdiff(1:nrow(features), train_idx)

# Split features and melody_ids
train_features <- features[train_idx,]
test_features <- features[test_idx,]
train_melody_ids <- melody_ids[train_idx]
test_melody_ids <- melody_ids[test_idx]

# Perform PCA on training data only
pca_result <- prcomp(train_features, center = TRUE, scale. = TRUE)

# Use more components (80% variance explained)
n_components <- which(cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2)) >= 0.80)[1]
cat("\nUsing", n_components, "components explaining", round(cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))[n_components] * 100, 2), "% variance")

# Extract PC scores
train_pc_scores <- predict(pca_result, train_features)[, 1:n_components]
test_pc_scores <- predict(pca_result, test_features)[, 1:n_components]

# Create polynomial features (adding squared terms)
train_pc_scores_poly <- cbind(
  train_pc_scores,
  train_pc_scores^2
)
test_pc_scores_poly <- cbind(
  test_pc_scores,
  test_pc_scores^2
)

colnames(train_pc_scores_poly) <- c(
  paste0("PC", 1:n_components),
  paste0("PC", 1:n_components, "_squared")
)
colnames(test_pc_scores_poly) <- colnames(train_pc_scores_poly)

# Prepare training data
train_data <- as.data.frame(train_pc_scores_poly) %>%
  mutate(melody_id = train_melody_ids) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(
      mean_score = mean(score),
      sd_score = sd(score),
      n_ratings = n()
    ),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Prepare test data
test_data <- as.data.frame(test_pc_scores_poly) %>%
  mutate(melody_id = test_melody_ids) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(
      mean_score = mean(score),
      sd_score = sd(score),
      n_ratings = n()
    ),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Split response and predictors
train_y <- train_data$mean_score
train_X <- as.matrix(train_data %>% select(-c(mean_score, melody_id, sd_score, n_ratings)))
test_y <- test_data$mean_score
test_X <- as.matrix(test_data %>% select(-c(mean_score, melody_id, sd_score, n_ratings)))

# Calculate weights to give more importance to extreme values
calculate_weights <- function(y) {
  # Calculate distance from mean
  y_mean <- mean(y)
  distances <- abs(y - y_mean)
  # Create weights based on distances (more weight to points far from mean)
  weights <- scale(distances, center = FALSE, scale = TRUE)
  weights <- weights + 1  # Add 1 to ensure all points have some weight
  return(weights)
}

# Standardize target variable
y_mean <- mean(train_data$mean_score)
y_sd <- sd(train_data$mean_score)
train_y_std <- scale(train_data$mean_score)
test_y_std <- (test_data$mean_score - y_mean) / y_sd

# Calculate sample weights
train_weights <- calculate_weights(train_y)

# Set up cross-validation folds
set.seed(123)
nfolds <- 10
folds <- sample(rep(1:nfolds, length.out = nrow(train_X)))

# Create sequence of lambda values
lambda_seq <- 10^seq(-5, -1, length.out = 20)

# Train models for different alpha values
alpha_seq <- seq(0, 1, by = 0.1)
cv_results <- list()
cv_errors <- matrix(NA, length(lambda_seq), length(alpha_seq))

for(i in seq_along(alpha_seq)) {
  # Fit cross-validated elastic net for each alpha
  cv_fit <- cv.glmnet(
    x = train_X,
    y = train_y_std,
    weights = train_weights,
    alpha = alpha_seq[i],
    lambda = lambda_seq,
    nfolds = nfolds,
    foldid = folds,
    family = "gaussian"
  )
  
  cv_results[[i]] <- cv_fit
  cv_errors[, i] <- cv_fit$cvm
}

# Find best alpha and lambda
min_error <- min(cv_errors)
best_indices <- which(cv_errors == min_error, arr.ind = TRUE)
best_alpha <- alpha_seq[best_indices[2]]
best_lambda <- lambda_seq[best_indices[1]]

# Fit final model with best parameters
final_model <- glmnet(
  x = train_X,
  y = train_y_std,
  weights = train_weights,
  alpha = best_alpha,
  lambda = best_lambda,
  family = "gaussian"
)

# Get predictions and transform back to original scale
train_pred_std <- predict(final_model, train_X, s = best_lambda)
test_pred_std <- predict(final_model, test_X, s = best_lambda)

train_pred <- train_pred_std * y_sd + y_mean
test_pred <- test_pred_std * y_sd + y_mean

# Print best parameters
cat("\nBest model parameters:")
cat("\nAlpha (mixing parameter):", best_alpha)
cat("\nLambda (regularization):", best_lambda)

# Get non-zero coefficients
coef_matrix <- as.matrix(coef(final_model, s = best_lambda))
non_zero_coef <- coef_matrix[coef_matrix != 0, , drop = FALSE]
cat("\n\nNumber of non-zero coefficients:", nrow(non_zero_coef) - 1)  # Subtract 1 for intercept

# Calculate metrics separately for different ranges
calculate_range_metrics <- function(actual, predicted, set_name) {
  # Create ranges
  ranges <- cut(actual, 
               breaks = c(0, 0.3, 0.5, 0.7, 0.85, 1),
               labels = c("Very Low", "Low", "Medium", "High", "Very High"))
  
  # Calculate metrics for each range
  range_levels <- levels(ranges)
  metrics_list <- list()
  
  for(range in range_levels) {
    mask <- ranges == range
    if(sum(mask) > 0) {  # Only calculate if we have data in this range
      actual_range <- actual[mask]
      pred_range <- predicted[mask]
      
      metrics_list[[range]] <- data.frame(
        Set = set_name,
        Range = range,
        N = sum(mask),
        MAE = mean(abs(pred_range - actual_range)),
        RMSE = sqrt(mean((pred_range - actual_range)^2)),
        R2 = cor(pred_range, actual_range)^2,
        Bias = mean(pred_range - actual_range)
      )
    }
  }
  
  # Combine all metrics
  do.call(rbind, metrics_list)
}

train_range_metrics <- calculate_range_metrics(train_y, train_pred, "Training")
test_range_metrics <- calculate_range_metrics(test_y, test_pred, "Test")

print("Metrics by range:")
print(rbind(train_range_metrics, test_range_metrics))

# Create prediction plots with confidence intervals
train_plot_data <- data.frame(
  Set = "Training",
  Actual = as.vector(train_y),
  Predicted = as.vector(train_pred),
  SD = train_data$sd_score,
  N = train_data$n_ratings
)

test_plot_data <- data.frame(
  Set = "Test",
  Actual = as.vector(test_y),
  Predicted = as.vector(test_pred),
  SD = test_data$sd_score,
  N = test_data$n_ratings
)

plot_data <- rbind(train_plot_data, test_plot_data)

# Add standard error of the mean
plot_data$SE <- plot_data$SD / sqrt(plot_data$N)

# Create enhanced prediction plot
p1 <- ggplot(plot_data, aes(x = Actual, y = Predicted, color = Set)) +
  geom_point(alpha = 0.5) +
  geom_errorbar(aes(ymin = Actual - SE, ymax = Actual + SE), alpha = 0.2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Weighted Elastic Net Predictions vs Actual Values",
    subtitle = paste("Using", n_components, "PCA components with emphasis on extreme values"),
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  coord_fixed(ratio = 1) +
  scale_color_brewer(palette = "Set1")

# Create residual plots
residual_data <- data.frame(
  Set = plot_data$Set,
  Predicted = plot_data$Predicted,
  Actual = plot_data$Actual,
  Residuals = plot_data$Predicted - plot_data$Actual,
  SE = plot_data$SE
)

p2 <- ggplot(residual_data, aes(x = Predicted, y = Residuals, color = Set)) +
  geom_point(alpha = 0.5) +
  geom_errorbar(aes(ymin = Residuals - SE, ymax = Residuals + SE), alpha = 0.2) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Residual Plot",
    x = "Predicted Values",
    y = "Residuals"
  ) +
  scale_color_brewer(palette = "Set1")

# Plot coefficients for original and squared terms separately
coef_df <- data.frame(
  Component = rownames(coef_matrix)[-1],
  Coefficient = coef_matrix[-1, 1]
) %>%
  mutate(
    Type = ifelse(grepl("squared", Component), "Squared", "Original"),
    Component = gsub("_squared", "", Component)
  ) %>%
  arrange(desc(abs(Coefficient)))

p3 <- ggplot(coef_df, 
             aes(x = reorder(Component, abs(Coefficient)), 
                 y = Coefficient,
                 fill = Type)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "PC Component Coefficients",
    x = "PCA Component",
    y = "Coefficient Value"
  ) +
  scale_fill_brewer(palette = "Set2")

# Create QQ plots
p4 <- ggplot(residual_data, aes(sample = Residuals, color = Set)) +
  stat_qq() +
  stat_qq_line() +
  theme_minimal() +
  labs(
    title = "Normal Q-Q Plot of Residuals",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  scale_color_brewer(palette = "Set1")

# Add bias analysis plot
bias_data <- data.frame(
  Actual = c(train_y, test_y),
  Bias = c(train_pred - train_y, test_pred - test_y),
  Set = c(rep("Training", length(train_y)), rep("Test", length(test_y)))
)

p5 <- ggplot(bias_data, aes(x = Actual, y = Bias, color = Set)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", se = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Bias Analysis",
    x = "Actual Values",
    y = "Prediction Bias (Predicted - Actual)"
  ) +
  scale_color_brewer(palette = "Set1")

# Arrange and display plots
gridExtra::grid.arrange(p1, p2, p3, p4, p5, ncol = 2)

# Save the model and results
saveRDS(list(
  pca_model = pca_result,
  elastic_net_model = final_model,
  n_components = n_components,
  var_explained = cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2)),
  metrics = metrics,
  range_metrics = list(
    train = train_range_metrics,
    test = test_range_metrics
  ),
  standardization = list(
    y_mean = y_mean,
    y_sd = y_sd
  )
), "melody_prediction_model.rds")

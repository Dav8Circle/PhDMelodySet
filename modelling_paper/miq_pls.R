# Load required libraries
library(tidyverse)
library(ggplot2)
library(xgboost)
library(glmnet)

# Load in the original item bank file
item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses
df <- 
  read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e7) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")

# Load features
features <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv") %>%
  select(-matches("\\{.*\\}")) %>%
  select(where(is.numeric))

# Store melody_ids before removing them
melody_ids <- features$melody_id
features <- features %>% select(-melody_id)

# Find and remove zero variance columns
zero_var_cols <- features %>%
  summarise(across(everything(), var)) %>%
  pivot_longer(everything()) %>%
  filter(value == 0) %>%
  pull(name)

if(length(zero_var_cols) > 0) {
  cat("Dropping zero variance features:", paste(zero_var_cols, collapse=", "), "\n")
}

# Drop zero variance columns and tempo features
features <- features %>%
  select(-starts_with("duration_features.tempo")) %>%
  select(-any_of(zero_var_cols))

# Scale features and remove any NaN values
features <- scale(features)
features <- features[, colSums(is.na(features)) == 0]

# Check for any remaining zero variance features after scaling
zero_var_after_scale <- apply(features, 2, function(x) var(x) == 0)
if(any(zero_var_after_scale)) {
  cat("Features with zero variance after scaling:", 
      paste(names(features)[zero_var_after_scale], collapse=", "), 
      "\n")
  features <- features[, !zero_var_after_scale]
}

# Prepare data for XGBoost
# Calculate mean scores for each melody
melody_scores <- df %>%
  group_by(item_id) %>%
  summarise(mean_score = mean(score))

# Combine features with scores
xgb_data <- as.data.frame(features) %>%
  mutate(melody_id = melody_ids) %>%
  left_join(melody_scores, by = c("melody_id" = "item_id")) %>%
  na.omit()

# Split response and predictors
y <- xgb_data$mean_score
X <- xgb_data %>% select(-c(mean_score, melody_id))

# --- Regularized Regression with glmnet (Elastic Net) ---

# Train/test split
set.seed(123)
n <- nrow(X)
train_idx <- sample(seq_len(n), size = 0.8 * n)
test_idx <- setdiff(seq_len(n), train_idx)

X_train <- as.matrix(X[train_idx, , drop = FALSE])
y_train <- y[train_idx]
X_test  <- as.matrix(X[test_idx, , drop = FALSE])
y_test  <- y[test_idx]

# Fit elastic net with cross-validation (alpha=0.5 for elastic net)
cvfit <- cv.glmnet(X_train, y_train, alpha = 0.5)
best_lambda <- cvfit$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Predict on train and test sets
pred_train <- predict(cvfit, newx = X_train, s = best_lambda)
pred_test  <- predict(cvfit, newx = X_test, s = best_lambda)

# Performance metrics
mse_train <- mean((y_train - pred_train)^2)
rmse_train <- sqrt(mse_train)
r2_train <- 1 - sum((y_train - pred_train)^2) / sum((y_train - mean(y_train))^2)

mse_test <- mean((y_test - pred_test)^2)
rmse_test <- sqrt(mse_test)
r2_test <- 1 - sum((y_test - pred_test)^2) / sum((y_test - mean(y_test))^2)

cat("\nElastic Net Model Performance Metrics (Train):")
cat("\nMSE:", round(mse_train, 4))
cat("\nRMSE:", round(rmse_train, 4))
cat("\nR-squared:", round(r2_train, 4))

cat("\n\nElastic Net Model Performance Metrics (Test):")
cat("\nMSE:", round(mse_test, 4))
cat("\nRMSE:", round(rmse_test, 4))
cat("\nR-squared:", round(r2_test, 4))

# Plot actual vs predicted for test set
plot_data_glmnet <- data.frame(
  actual = y_test,
  predicted = as.vector(pred_test)
)

ggplot(plot_data_glmnet, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Elastic Net: Actual vs Predicted Scores (Test Set)",
    x = "Actual Score",
    y = "Predicted Score"
  )

# Plot coefficient magnitudes
coef_df <- as.data.frame(as.matrix(coef(cvfit, s = best_lambda)))
coef_df$feature <- rownames(coef_df)
colnames(coef_df)[1] <- "coefficient"
coef_df <- coef_df %>% filter(feature != "(Intercept)") %>% arrange(desc(abs(coefficient)))

ggplot(coef_df[1:30,], aes(x = reorder(feature, abs(coefficient)), y = coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 30 Coefficient Magnitudes (Elastic Net)",
    x = "Feature",
    y = "Coefficient"
  )

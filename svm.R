# Load required libraries
library(e1071)
library(tidyverse)

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

# Prepare data for SVM similar to ridge regression approach
svm_data <- as.data.frame(features) %>%
  mutate(melody_id = melody_ids) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Split response and predictors
y <- svm_data$mean_score
X <- svm_data %>% select(-c(mean_score, melody_id))

# Perform cross-validation to tune parameters
set.seed(123)
tune_out <- tune(svm, mean_score ~ ., data = data.frame(X, mean_score = y),
                 kernel = "radial",
                 ranges = list(
                   cost = c(0.1, 1, 10),
                   gamma = c(0.1, 1, 10)
                 ))

print(tune_out)

# Fit SVM with optimal parameters
svm_model <- svm(mean_score ~ ., data = data.frame(X, mean_score = y),
                 kernel = "radial",
                 cost = tune_out$best.parameters$cost,
                 gamma = tune_out$best.parameters$gamma)

# Calculate R-squared on training data
y_pred <- predict(svm_model, X)
r2 <- cor(y, y_pred)^2
cat("\nR-squared:", round(r2, 3))

# Perform k-fold cross-validation
k <- 10
set.seed(123)
folds <- sample(1:k, nrow(X), replace = TRUE)
cv_rmse <- numeric(k)
cv_r2 <- numeric(k)

for(i in 1:k) {
  # Split data
  train_indices <- folds != i
  train_data <- data.frame(X[train_indices,], mean_score = y[train_indices])
  test_data <- data.frame(X[!train_indices,], mean_score = y[!train_indices])
  
  # Fit model
  cv_model <- svm(mean_score ~ ., data = train_data,
                  kernel = "radial",
                  cost = tune_out$best.parameters$cost,
                  gamma = tune_out$best.parameters$gamma)
  
  # Make predictions
  pred_y <- predict(cv_model, test_data)
  test_y <- test_data$mean_score
  
  # Calculate metrics
  cv_rmse[i] <- sqrt(mean((test_y - pred_y)^2))
  cv_r2[i] <- cor(test_y, pred_y)^2
}

cat("\nCross-validated RMSE:", round(mean(cv_rmse), 3))
cat("\nCross-validated R-squared:", round(mean(cv_r2), 3))

# Get feature importance through permutation
importance_scores <- sapply(names(X), function(feature) {
  X_permuted <- X
  X_permuted[[feature]] <- sample(X_permuted[[feature]])
  pred_permuted <- predict(svm_model, X_permuted)
  return(cor(y, y_pred)^2 - cor(y, pred_permuted)^2)
})

# Create importance dataframe
importance_df <- data.frame(
  feature = names(X),
  importance = importance_scores
) %>%
  arrange(desc(abs(importance)))

# Plot top 20 features by importance
ggplot(importance_df %>% head(20),
       aes(x = reorder(feature, abs(importance)), y = importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Features by SVM Importance",
    x = "Feature",
    y = "Importance (Decrease in R-squared)"
  )

# Calculate R-squared on training data
y_pred <- predict(svm_model, X)
r2 <- cor(y, y_pred)^2
cat("\nTraining R-squared:", round(r2, 3))

# Calculate RMSE on training data
rmse <- sqrt(mean((y - y_pred)^2))
cat("\nTraining RMSE:", round(rmse, 3))

# Create scatter plot of predicted vs actual values
pred_vs_actual <- data.frame(
  Actual = y,
  Predicted = y_pred
)

ggplot(pred_vs_actual, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "SVM Predictions vs Actual Values",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  coord_fixed(ratio = 1)

# Calculate accuracy metrics
mae <- mean(abs(y - y_pred))
mse <- mean((y - y_pred)^2)
rmse <- sqrt(mse)
r2 <- cor(y, y_pred)^2

cat("\nModel Accuracy Metrics:")
cat("\nMean Absolute Error (MAE):", round(mae, 3))
cat("\nMean Squared Error (MSE):", round(mse, 3)) 
cat("\nRoot Mean Squared Error (RMSE):", round(rmse, 3))
cat("\nR-squared (R2):", round(r2, 3))

summary(svm_model)

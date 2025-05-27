# Read the CSV files
original_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
miq_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")

# Load required libraries
library(gbm)
library(tidyverse)
library(caret)
library(ggplot2)

# Keep melody ID column
melody_id <- original_features$melody_id

# Drop non-numeric columns before subtraction
original_features <- original_features[sapply(original_features, is.numeric)]
miq_features <- miq_features[sapply(miq_features, is.numeric)]

# Calculate differences between dataframes
feature_diffs <- miq_features - original_features

# Find zero variance columns
zero_var_cols <- feature_diffs %>%
  summarise(across(everything(), var)) %>%
  pivot_longer(everything()) %>%
  filter(value == 0) %>%
  pull(name)

# Print dropped columns
if(length(zero_var_cols) > 0) {
  cat("Dropping zero variance features:", paste(zero_var_cols, collapse=", "), "\n")
}

# Drop zero variance columns and tempo features
feature_diffs <- feature_diffs %>%
  select(-starts_with("duration_features.tempo")) %>%
  select(-any_of(zero_var_cols))

# Handle infinite and missing values
feature_diffs <- feature_diffs %>%
  mutate(across(everything(), ~replace(., is.infinite(.), NA))) %>%
  mutate(across(everything(), ~replace(., is.na(.), 0)))

# Create 80-20 split before scaling and PCA
set.seed(123)  # for reproducibility
n_samples <- nrow(feature_diffs)
train_idx <- sample(1:n_samples, size = floor(0.8 * n_samples))
train_data <- feature_diffs[train_idx, ]
test_data <- feature_diffs[-train_idx, ]

# Scale the features using only training data
scaler <- preProcess(train_data, method = c("center", "scale"))
train_data_scaled <- predict(scaler, train_data)
test_data_scaled <- predict(scaler, test_data)

# Perform PCA on training data
pca <- prcomp(train_data_scaled)

# Transform test data using the PCA from training
test_data_pca <- predict(pca, test_data_scaled)

# Print summary of PCA (based on training data)
print(summary(pca))

# Plot scree plot
plot(pca$sdev^2/sum(pca$sdev^2), type="b", 
     xlab="Principal Component", 
     ylab="Proportion of Variance Explained",
     main="Scree Plot")

# Load in the original item bank file
item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses
df <- 
  read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")

# Extract first 15 PCs for both train and test sets
train_pc_scores <- pca$x[, 1:15]
test_pc_scores <- test_data_pca[, 1:15]

# Prepare data for GBM - Training set
train_gbm_data <- as.data.frame(train_pc_scores) %>%
  mutate(melody_id = melody_id[train_idx]) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Prepare data for GBM - Test set
test_gbm_data <- as.data.frame(test_pc_scores) %>%
  mutate(melody_id = melody_id[-train_idx]) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Split response and predictors for training
X_train <- train_gbm_data %>% select(-c(mean_score, melody_id))
y_train <- train_gbm_data$mean_score

# Split response and predictors for testing
X_test <- test_gbm_data %>% select(-c(mean_score, melody_id))
y_test <- test_gbm_data$mean_score

# Create parameter grid for GBM
gbm_grid <- expand.grid(
  n.trees = c(100, 500, 1000),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(5, 10)
)

# Set up cross-validation control
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  search = "grid",
  savePredictions = TRUE
)

# Train GBM with cross-validation
gbm_cv <- train(
  x = X_train,
  y = y_train,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_grid,
  verbose = FALSE
)

# Get best model
best_model <- gbm_cv$finalModel

# Make predictions
y_pred_train <- predict(gbm_cv, X_train)
y_pred_test <- predict(gbm_cv, X_test)

# Calculate metrics
train_r2 <- cor(y_train, y_pred_train)^2
test_r2 <- cor(y_test, y_pred_test)^2
train_rmse <- sqrt(mean((y_train - y_pred_train)^2))
test_rmse <- sqrt(mean((y_test - y_pred_test)^2))
train_mae <- mean(abs(y_train - y_pred_train))
test_mae <- mean(abs(y_test - y_pred_test))

# Print metrics
cat("\nTraining Metrics:")
cat("\nR-squared:", round(train_r2, 3))
cat("\nRMSE:", round(train_rmse, 3))
cat("\nMAE:", round(train_mae, 3))

cat("\n\nTest Metrics:")
cat("\nR-squared:", round(test_r2, 3))
cat("\nRMSE:", round(test_rmse, 3))
cat("\nMAE:", round(test_mae, 3))

cat("\n\nBest GBM parameters:")
cat("\nNumber of trees:", gbm_cv$bestTune$n.trees)
cat("\nInteraction depth:", gbm_cv$bestTune$interaction.depth)
cat("\nLearning rate:", gbm_cv$bestTune$shrinkage)
cat("\nMin observations in node:", gbm_cv$bestTune$n.minobsinnode)

# Get variable importance
var_imp <- summary(best_model, n.trees = best_model$n.trees)

# Create scatter plot of predicted vs actual values
pred_vs_actual <- data.frame(
  Actual = y_test,
  Predicted = y_pred_test
)

# Create main prediction plot
p1 <- ggplot(pred_vs_actual, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "GBM Predictions vs Actual Values",
    subtitle = "Using 15 PCA components",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  coord_fixed(ratio = 1)

# Create residual plot
residuals <- y_pred_test - y_test
p2 <- ggplot(data.frame(Predicted = y_pred_test, Residuals = residuals), 
             aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Residual Plot",
    x = "Predicted Values",
    y = "Residuals"
  )

# Plot variable importance for top 10 components
importance_df <- as.data.frame(var_imp)
importance_df$Component <- rownames(importance_df)
p3 <- ggplot(importance_df %>% head(10), 
             aes(x = reorder(Component, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 10 Most Important Components",
    x = "PCA Component",
    y = "Relative Importance (%)"
  )

# Arrange plots in a grid
gridExtra::grid.arrange(p1, p2, p3, ncol = 2)

# Calculate and print additional insights
cat("\n\nModel Insights:")
cat("\nMean absolute error by score range:")
pred_vs_actual$score_range <- cut(pred_vs_actual$Actual, 
                                 breaks = c(0, 0.25, 0.5, 0.75, 1),
                                 labels = c("0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"))
mae_by_range <- pred_vs_actual %>%
  group_by(score_range) %>%
  summarise(mae = mean(abs(Predicted - Actual)))
print(mae_by_range)
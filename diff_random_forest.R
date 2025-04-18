# Read the CSV files
original_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
miq_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")

# Load required libraries first
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# Keep melody ID column
melody_id <- original_features$melody_id

# Drop non-numeric columns before subtraction
original_features <- original_features[sapply(original_features, is.numeric)]
miq_features <- miq_features[sapply(miq_features, is.numeric)]

# Calculate differences between dataframes
feature_diffs <- as.data.frame(miq_features - original_features)

# Find zero variance columns
zero_var_cols <- names(feature_diffs)[apply(feature_diffs, 2, var) == 0]

# Print dropped columns
if(length(zero_var_cols) > 0) {
  cat("Dropping zero variance features:", paste(zero_var_cols, collapse=", "), "\n")
}

# Remove tempo features and zero variance columns using base R
tempo_cols <- grep("duration_features.tempo", names(feature_diffs), value = TRUE)
cols_to_remove <- unique(c(tempo_cols, zero_var_cols))
feature_diffs <- feature_diffs[, !names(feature_diffs) %in% cols_to_remove]

# Handle infinite and missing values
feature_diffs <- as.data.frame(lapply(feature_diffs, function(x) {
  x[is.infinite(x)] <- NA
  x[is.na(x)] <- 0
  x
}))

# Scale the features
feature_diffs <- scale(feature_diffs)

# Perform PCA
pca <- prcomp(feature_diffs)

# Print summary of PCA
print(summary(pca))

# Plot scree plot
pdf("R/rf_scree_plot.pdf")
plot(pca$sdev^2/sum(pca$sdev^2), type="b", 
     xlab="Principal Component", 
     ylab="Proportion of Variance Explained",
     main="Scree Plot")
dev.off()

# Load required libraries
library(randomForest)
library(tidyverse)
library(caret)
library(ggplot2)

# Load in the original item bank file
item_bank <- read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") %>%
  rename(item_id = id) %>%
  select(-discrimination, -difficulty, -guessing, -inattention)

# Read in the participant responses
df <- read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) %>%
  filter(test == "mdt") %>%
  left_join(item_bank, by = "item_id")

# Extract first 15 PCs
pc_scores <- pca$x[, 1:15]

# Prepare data for Random Forest
rf_data <- as.data.frame(pc_scores) %>%
  mutate(melody_id = melody_id) %>%
  left_join(
    df %>% 
      select(item_id, score),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Convert scores to factor with explicit levels
rf_data$score <- factor(rf_data$score, levels = c("0", "1"), labels = c("No", "Yes"))

# Split response and predictors
y <- rf_data$score
X <- rf_data %>% select(-c(score, melody_id))

# Create train/test split
set.seed(123)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx,]
X_test <- X[-train_idx,]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Create parameter grid for Random Forest
rf_grid <- expand.grid(
  mtry = c(3, 5, 7, 9)
)

# Set up cross-validation control
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  search = "grid",
  savePredictions = TRUE,
  classProbs = TRUE
)

# Train Random Forest with cross-validation
set.seed(123)
rf_cv <- train(
  x = X_train,
  y = y_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 1000,
  importance = TRUE
)

# Get best model
best_model <- rf_cv$finalModel

# Make predictions
y_pred_train <- predict(rf_cv, X_train)
y_pred_test <- predict(rf_cv, X_test)

# Calculate metrics
train_accuracy <- confusionMatrix(y_pred_train, y_train)$overall["Accuracy"]
test_accuracy <- confusionMatrix(y_pred_test, y_test)$overall["Accuracy"]
train_kappa <- confusionMatrix(y_pred_train, y_train)$overall["Kappa"]
test_kappa <- confusionMatrix(y_pred_test, y_test)$overall["Kappa"]

# Print metrics
cat("\nTraining Metrics:")
cat("\nAccuracy:", round(train_accuracy, 3))
cat("\nKappa:", round(train_kappa, 3))

cat("\n\nTest Metrics:")
cat("\nAccuracy:", round(test_accuracy, 3))
cat("\nKappa:", round(test_kappa, 3))

cat("\n\nBest Random Forest parameters:")
cat("\nmtry:", rf_cv$bestTune$mtry)
cat("\nNumber of trees:", best_model$ntree)

# Get variable importance
var_imp <- importance(best_model)
var_imp_df <- data.frame(
  Component = rownames(var_imp),
  Importance = var_imp[, "MeanDecreaseGini"]
)

# Create confusion matrix plot
cm <- confusionMatrix(y_pred_test, y_test)
cm_plot <- ggplot(data = as.data.frame(cm$table)) +
  geom_tile(aes(x = Prediction, y = Reference, fill = Freq)) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(
    title = "Confusion Matrix",
    x = "Predicted",
    y = "Actual"
  )

# Create ROC curve
roc_data <- data.frame(
  actual = as.numeric(as.character(y_test)) - 1,  # Convert back to 0/1 for ROC
  predicted = predict(rf_cv, X_test, type = "prob")[, "Yes"]
)

roc_plot <- ggplot(roc_data, aes(d = actual, m = predicted)) +
  geom_roc() +
  style_roc() +
  theme_minimal() +
  labs(
    title = "ROC Curve",
    x = "False Positive Rate",
    y = "True Positive Rate"
  )

# Plot variable importance
importance_plot <- ggplot(var_imp_df %>% head(10), 
                          aes(x = reorder(Component, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 10 Most Important Components",
    x = "PCA Component",
    y = "Mean Decrease in Gini"
  )

# Save plots
pdf("R/rf_confusion_matrix.pdf")
print(cm_plot)
dev.off()

pdf("R/rf_roc_curve.pdf")
print(roc_plot)
dev.off()

pdf("R/rf_importance_plot.pdf")
print(importance_plot)
dev.off()

# Calculate and print additional insights
cat("\n\nModel Insights:")
cat("\nConfusion Matrix:")
print(cm$table)

# Calculate accuracy by score range
pred_vs_actual <- data.frame(
  Actual = as.numeric(as.character(y_test)) - 1,  # Convert back to 0/1
  Predicted = as.numeric(as.character(y_pred_test)) - 1  # Convert back to 0/1
)

accuracy_by_range <- pred_vs_actual %>%
  group_by(Actual) %>%
  summarise(accuracy = mean(Actual == Predicted))
print(accuracy_by_range) 
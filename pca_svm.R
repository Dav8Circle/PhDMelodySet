# Load required libraries
library(e1071)
library(tidyverse)
library(caret)
library(gbm)
library(ggplot2)
library(doParallel)

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

# --- Remove PCA and use raw features ---
# features is already scaled and cleaned above
raw_features <- as.data.frame(features)
raw_features$melody_id <- melody_ids

# Prepare data for GBM
raw_gbm_data <- raw_features %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Create train-test split (80-20)
set.seed(123)
train_idx <- createDataPartition(raw_gbm_data$mean_score, p = 0.8, list = FALSE)
train_data <- raw_gbm_data[train_idx, ]
test_data <- raw_gbm_data[-train_idx, ]

# Split response and predictors for training
X_train <- train_data %>% select(-c(mean_score, melody_id))
y_train <- train_data$mean_score
X_test <- test_data %>% select(-c(mean_score, melody_id))
y_test <- test_data$mean_score

# Train GBM with cross-validation
set.seed(123)
gbm_cv <- train(
  x = X_train,
  y = y_train,
  method = "gbm",
  trControl = ctrl,
  tuneGrid = gbm_grid,
  verbose = FALSE,
  distribution = "gaussian"
)

# Get best model
best_model <- gbm_cv$finalModel

# Calculate predictions for both train and test sets
train_pred <- predict(gbm_cv, X_train)
test_pred <- predict(gbm_cv, X_test)

# Calculate metrics for both sets
train_r2 <- cor(y_train, train_pred)^2
train_mae <- mean(abs(y_train - train_pred))
train_rmse <- sqrt(mean((y_train - train_pred)^2))

test_r2 <- cor(y_test, test_pred)^2
test_mae <- mean(abs(y_test - test_pred))
test_rmse <- sqrt(mean((y_test - test_pred)^2))

# Get variable importance
var_imp <- summary(best_model, n.trees = best_model$n.trees)

# Stop parallel processing
stopCluster(cl)

# Print results
cat("\n\nRaw Feature Model Results:")
cat("\nTraining Metrics:")
cat("\nR-squared:", round(train_r2, 3))
cat("\nRMSE:", round(train_rmse, 3))
cat("\nMAE:", round(train_mae, 3))
cat("\nTest Metrics:")
cat("\nR-squared:", round(test_r2, 3))
cat("\nRMSE:", round(test_rmse, 3))
cat("\nMAE:", round(test_mae, 3))
cat("\nBest parameters:")
cat("\nNumber of trees:", gbm_cv$bestTune$n.trees)
cat("\nInteraction depth:", gbm_cv$bestTune$interaction.depth)
cat("\nLearning rate:", gbm_cv$bestTune$shrinkage)
cat("\nMin observations in node:", gbm_cv$bestTune$n.minobsinnode)

# Create scatter plot of predicted vs actual values for test set
pred_vs_actual <- data.frame(
  Actual = y_test,
  Predicted = test_pred
)

# Create main prediction plot
p1 <- ggplot(pred_vs_actual, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "GBM Predictions vs Actual Values (Test Set, Raw Features)",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  coord_fixed(ratio = 1)

# Create residual plot for test set
residuals <- test_pred - y_test
p2 <- ggplot(data.frame(Predicted = test_pred, Residuals = residuals), 
             aes(x = Predicted, y = Residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Residual Plot (Test Set, Raw Features)",
    x = "Predicted Values",
    y = "Residuals"
  )

# Plot variable importance for top 20 features
importance_df <- as.data.frame(var_imp)
importance_df$Feature <- rownames(importance_df)
p3 <- ggplot(importance_df %>% head(20), 
             aes(x = reorder(Feature, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Most Important Raw Features",
    x = "Feature",
    y = "Relative Importance (%)"
  )

# Arrange plots in a grid
gridExtra::grid.arrange(p1, p2, p3, ncol = 2)

# Print final model summary
cat("\n\nBest Model Summary:")
cat("\nNumber of PCA components used:", 0)
cat("\nVariance explained by PCA components:", 0, "%")
cat("\nBest GBM parameters:")
cat("\nNumber of trees:", gbm_cv$bestTune$n.trees)
cat("\nInteraction depth:", gbm_cv$bestTune$interaction.depth)
cat("\nLearning rate:", gbm_cv$bestTune$shrinkage)
cat("\nMin observations in node:", gbm_cv$bestTune$n.minobsinnode)
cat("\n\nFinal Model Metrics:")
cat("\nR-squared:", round(test_r2, 3))
cat("\nRMSE:", round(test_rmse, 3))
cat("\nMAE:", round(test_mae, 3))

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

loadings <- pca_result$rotation[,1:17]

# Create data frame of loadings with variable names
loadings_df <- data.frame(
  variable = rownames(loadings),
  PC1 = loadings[,1],
  PC2 = loadings[,2],
  PC3 = loadings[,3],
  PC4 = loadings[,4],
  PC5 = loadings[,5],
  PC6 = loadings[,6],
  PC7 = loadings[,7],
  PC8 = loadings[,8],
  PC9 = loadings[,9],
  PC10 = loadings[,10],
  PC11 = loadings[,11],
  PC12 = loadings[,12],
  PC13 = loadings[,13],
  PC14 = loadings[,14],
  PC15 = loadings[,15],
  PC16 = loadings[,16],
  PC17 = loadings[,17]
)

loadings_df <- loadings_df %>%
  mutate(category = case_when(
    str_starts(variable, "pitch") ~ "pitch_features",
    str_starts(variable, "interval") ~ "interval_features",
    str_starts(variable, "contour") ~ "contour_features", 
    str_starts(variable, "duration") ~ "duration_features",
    str_starts(variable, "tonality") ~ "tonality_features",
    str_starts(variable, "narmour") ~ "narmour_features",
    str_starts(variable, "melodic_movement") ~ "melodic_movement_features",
    str_starts(variable, "mtype") ~ "mtype_features",
    str_starts(variable, "corpus") ~ "corpus_features"
  ))

library(plotly)
plot_ly(loadings_df, 
        x = ~PC1, 
        y = ~PC9,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", 
                   "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC1:", round(PC1, 3),
                      "<br>PC9:", round(PC9, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC1"),
         yaxis = list(title = "PC9"))

# library(PCAtest)
# PC_sig <- PCAtest(features, nboot=1000, nperm=1000, plot=FALSE)

on.exit(stopCluster(cl))

# Load required libraries
library(glmnet)
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

# Prepare data for ridge regression similar to random forest approach
ridge_data <- as.data.frame(features) %>%
  mutate(melody_id = melody_ids) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Split response and predictors 
y <- ridge_data$mean_score
X <- as.matrix(ridge_data %>% select(-c(mean_score, melody_id)))

# Perform cross-validation to find optimal lambda
set.seed(123)
cv_ridge <- cv.glmnet(X, y, alpha = 0, nfolds = 10)

# Plot cross-validation results
plot(cv_ridge)

# Print optimal lambda values
cat("Optimal lambda (minimum MSE):", cv_ridge$lambda.min, "\n")
cat("Optimal lambda (1-SE rule):", cv_ridge$lambda.1se, "\n")

# Fit ridge model with optimal lambda
ridge_model <- glmnet(X, y, alpha = 0, lambda = cv_ridge$lambda.min)

# Get coefficients
coef_df <- data.frame(
  feature = rownames(coef(ridge_model)),
  coefficient = as.vector(coef(ridge_model))
) %>%
  filter(feature != "(Intercept)") %>%
  arrange(desc(abs(coefficient)))

# Plot top 20 features by absolute coefficient size
ggplot(coef_df %>% head(20),
       aes(x = reorder(feature, abs(coefficient)), y = coefficient)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Features by Ridge Regression Coefficient Magnitude",
    x = "Feature",
    y = "Coefficient"
  )

# Calculate R-squared on training data
y_pred <- predict(ridge_model, X)
r2 <- cor(y, y_pred)^2
cat("\nR-squared:", round(r2, 3))

# Perform k-fold cross-validation to get out-of-sample performance
k <- 10
set.seed(123)
folds <- sample(1:k, nrow(X), replace = TRUE)
cv_rmse <- numeric(k)
cv_r2 <- numeric(k)

for(i in 1:k) {
  # Split data
  train_X <- X[folds != i,]
  train_y <- y[folds != i]
  test_X <- X[folds == i,]
  test_y <- y[folds == i]
  
  # Fit model
  cv_model <- glmnet(train_X, train_y, alpha = 0, lambda = cv_ridge$lambda.min)
  
  # Make predictions
  pred_y <- predict(cv_model, test_X)
  
  # Calculate metrics
  cv_rmse[i] <- sqrt(mean((test_y - pred_y)^2))
  cv_r2[i] <- cor(test_y, pred_y)^2
}

cat("\nCross-validated RMSE:", round(mean(cv_rmse), 3))
cat("\nCross-validated R-squared:", round(mean(cv_r2), 3))

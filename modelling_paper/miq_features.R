# Load in the original item bank file
item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses (first 1e6, as too many will throttle
# my cpu)
df <- 
  read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")

# Now loading the features calculated from my feature set - and filtering just
# for numeric items
features <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv") %>%
  select(-matches("\\{.*\\}")) %>%
  select(where(is.numeric))

# Store melody_ids before removing them for PCA
melody_ids <- features$melody_id
features <- features %>% select(-melody_id)

# Find zero variance columns
zero_var_cols <- features %>%
  summarise(across(everything(), var)) %>%
  pivot_longer(everything()) %>%
  filter(value == 0) %>%
  pull(name)

# Print dropped columns
if(length(zero_var_cols) > 0) {
  cat("Dropping zero variance features:", paste(zero_var_cols, collapse=", "), "\n")
}

# Drop zero variance columns and tempo features
features <- features %>%
  select(-starts_with("duration_features.tempo")) %>%
  select(-any_of(zero_var_cols))

# Let's start with a simple PCA
library(PCAtest)
library(stringr)
features <- scale(features)
# Drop any features that contain NaN values
features <- features[, colSums(is.na(features)) == 0]

# Check for any remaining zero variance features after scaling
zero_var_after_scale <- apply(features, 2, function(x) var(x) == 0)
if(any(zero_var_after_scale)) {
  cat("Features with zero variance after scaling:", 
      paste(names(features)[zero_var_after_scale], collapse=", "), 
      "\n")
  # Remove any remaining zero variance features
  features <- features[, !zero_var_after_scale]
}

pca.res <- PCAtest(features, nperm=1000, nboot=1000, plot=TRUE)

pca.data <- prcomp(features)
loadings <- pca.data$rotation[,1:5]

# Create data frame of loadings with variable names
loadings_df <- data.frame(
  variable = rownames(loadings),
  PC1 = loadings[,1],
  PC2 = loadings[,2],
  PC3 = loadings[,3],
  PC4 = loadings[,4],
  PC5 = loadings[,5]
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


# Plot loadings with plotly for interactive hover labels
library(plotly)
plot_ly(loadings_df, 
        x = ~PC1, 
        y = ~PC2,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", 
                   "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC1:", round(PC1, 3),
                      "<br>PC2:", round(PC2, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC1"),
         yaxis = list(title = "PC2"))

# Get first 5 PCs from PCA scores and add melody_ids back
pca_scores <- as.data.frame(pca.data$x[,1:5])
names(pca_scores) <- c("PC1", "PC2", "PC3", "PC4", "PC5") 
pca_scores$melody_id <- melody_ids

# Join PCA scores with participant data
df_with_pcs <- left_join(df, pca_scores, by = c("item_id" = "melody_id"))

# Handle NA values before model fitting
df_with_pcs <- na.omit(df_with_pcs)

df_with_pcs <- df_with_pcs %>%
  group_by(item_id) %>%
  mutate(
    score_item = mean(score))

# Fit linear model using first 5 PCs to predict score
pc_model <- lm(score_item ~ PC1 + PC2 + PC3 + PC4 + PC5 + difficulty
               + ability_WL, data = df_with_pcs)
summary(pc_model)

# Create formula explicitly
formula <- score ~ PC1 + PC2 + PC3 + PC4 + PC5 + (1|user_id)

# Fit mixed effects logistic regression with explicit formula
mod <- glmer(formula, 
            family = binomial, 
            data = df_with_pcs,
            control = glmerControl(optimizer = "bobyqa"))  # More stable optimizer

# Print model summary
summary(mod)

# Calculate R-squared values using r2beta
library(r2glmm)
r2_values <- r2beta(mod, partial = TRUE)
print(r2_values)

# Calculate AIC and BIC
AIC_val <- AIC(mod)
BIC_val <- BIC(mod)
cat("\nAIC:", round(AIC_val, 1))
cat("\nBIC:", round(BIC_val, 1))

# Cross-validate model predictions
library(caret)
set.seed(123)

# Create 10-fold CV with stratification by user_id
train_control <- trainControl(
  method = "cv",
  number = 10,
  index = createFolds(df_with_pcs$user_id, k = 10)  # Stratify by user
)

# Train model with cross-validation
cv_model <- train(
  formula,
  data = df_with_pcs,
  method = "glm",
  family = "binomial",
  trControl = train_control
)

# Print cross-validation results
print(cv_model)

# Get predictions
df_with_pcs$predicted_prob <- predict(mod, type = "response")
df_with_pcs$predicted_class <- ifelse(df_with_pcs$predicted_prob > 0.5, 1, 0)

# Calculate confusion matrix
conf_matrix <- table(df_with_pcs$score, df_with_pcs$predicted_class)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- conf_matrix[2,2] / sum(conf_matrix[,2])
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\nAccuracy:", round(accuracy, 3))
cat("\nPrecision:", round(precision, 3))
cat("\nRecall:", round(recall, 3))
cat("\nF1 Score:", round(f1_score, 3))

# Plot ROC curve
library(pROC)
roc_curve <- roc(df_with_pcs$score, df_with_pcs$predicted_prob)
plot(roc_curve, main = "ROC Curve")
auc(roc_curve)

# Plot calibration curve
library(ggplot2)
ggplot(df_with_pcs, aes(x = predicted_prob, y = score)) +
  geom_point(alpha = 0.1) +
  geom_smooth(method = "loess", color = "red") +
  geom_abline(intercept = 0, slope = 1, color = "blue", linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Calibration Plot",
    x = "Predicted Probability",
    y = "Actual Score"
  )

# Random Forest approach to predict score using all features
library(randomForest)

# Prepare data for random forest
# Convert features matrix to data frame and combine with scores
rf_data <- as.data.frame(features) %>%
  mutate(melody_id = melody_ids) %>%
  left_join(
    df %>% 
    group_by(item_id) %>%
    summarise(mean_score = mean(score)),
    by = c("melody_id" = "item_id")
  ) %>%
  na.omit()

# Split response and predictors
y <- rf_data$mean_score
X <- rf_data %>% select(-c(mean_score, melody_id))  # Keep melody_id in rf_data but exclude from X

# Train random forest model
set.seed(123)
rf_model <- randomForest(
  x = X,
  y = y,
  ntree = 500,
  importance = TRUE
)

# Print model summary
print(rf_model)

# Get variable importance measures
importance_scores <- importance(rf_model)
importance_df <- data.frame(
  feature = rownames(importance_scores),
  importance = importance_scores[,"%IncMSE"]
) %>%
  arrange(desc(importance))

# Plot top 20 most important features
library(ggplot2)
ggplot(importance_df %>% head(20), 
       aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Most Important Features for Predicting Score",
    x = "Feature",
    y = "Importance (%IncMSE)"
  )

# Perform PLS regression
library(pls)
library(plsVarSel)  # Add this package for VIP scores

# Scale the predictors
X_scaled <- scale(as.matrix(X))

# Fit PLS model with cross-validation to determine optimal components
set.seed(123)
pls_model <- plsr(y ~ X_scaled, 
                  validation = "CV",
                  ncomp = 20)

# Plot RMSEP to examine number of components
plot(RMSEP(pls_model), main = "RMSEP vs Number of Components")

# Get optimal number of components based on RMSEP
rmsep_values <- RMSEP(pls_model)$val[1,,]  # Get all RMSEP values
ncomp_opt <- which.min(rmsep_values) - 1
cat("Optimal number of components:", ncomp_opt, "\n")

# Refit model with optimal components
pls_final <- plsr(y ~ X_scaled, ncomp = ncomp_opt)

# Get PLS scores
pls_scores <- scores(pls_final)
pls_scores_df <- as.data.frame(pls_scores)

# Get variable importance in projection (VIP) scores
R2 <- R2(pls_final)
W <- loading.weights(pls_final)
P <- loadings(pls_final)
T <- scores(pls_final)
VIP_scores <- VIP(pls_final, opt.comp = ncomp_opt)  # Added opt.comp parameter

# Create VIP scores data frame
vip_df <- data.frame(
  feature = colnames(X),
  VIP = VIP_scores
) %>%
  arrange(desc(VIP))

# Plot top 20 features by VIP score
ggplot(vip_df %>% head(20),
       aes(x = reorder(feature, VIP), y = VIP)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Features by VIP Score",
    x = "Feature",
    y = "VIP Score"
  )

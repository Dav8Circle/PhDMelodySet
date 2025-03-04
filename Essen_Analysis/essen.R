library(tidyverse)

# Read in features CSV
features <- read_csv("essen_features.csv")
# Read in JSON file for melody names
library(jsonlite)
melodies <- fromJSON("essen_midi_sequences.json")

# Extract ID and melody name columns
melody_names <- data.frame(
  ID = melodies$ID,
  Original_Melody = melodies$`Original Melody`
)

# Join features with melody names
features <- merge(features, melody_names, by.x="melody_id", by.y="ID")

# Keep numeric columns plus melody_id and Original_Melody
features <- features %>%
  select(-matches("\\{.*\\}")) %>%
  select(melody_id, Original_Melody, where(is.numeric))

# Extract country from Original_Melody by taking only the letter parts
features <- features %>%
  mutate(country = str_extract(Original_Melody, "[A-Za-z]+"))

# Keep only han, shanx (grouped together) and deut (limited to 2000)
features <- features %>%
  filter(country %in% c("han", "shanx", "deut")) %>%
  mutate(country = ifelse(country %in% c("han", "shanx"), "china", country))

# Randomly sample 2000 German melodies
set.seed(8)
deut_samples <- features %>%
  filter(country == "deut") %>%
  sample_n(2000)

# Combine with all Chinese melodies
features <- features %>%
  filter(country == "china") %>%
  bind_rows(deut_samples)

# Remove melody_id and Original_Melody columns for modeling
model_data <- features %>%
  select(-melody_id, -Original_Melody) %>%
  select_if(~!is.factor(.))

# Split features into predictors and response
X <- model_data %>% select(-country)
y <- model_data$country


# Remove any columns with NA/infinite values
X <- X %>% select_if(~ !any(is.na(.) | is.infinite(.)))
# Split into training and test sets
set.seed(8)
train_idx <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_idx,]
X_test <- X[-train_idx,]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Print dataset sizes
print(paste("Training set size:", nrow(X_train)))
print(paste("Test set size:", nrow(X_test)))
print(paste("Number of features:", ncol(X_train)))
print(paste("Number of unique countries:", length(unique(y))))
print("\nClass distribution in training set:")
print(table(y_train))

# Train random forest model
library(randomForest)
rf_model <- randomForest(x = X_train, 
                        y = as.factor(y_train),
                        ntree = 500,
                        importance = TRUE)

# Print model details
print("\nRandom Forest Model Summary:")
print(rf_model)

# Make predictions and evaluate
predictions <- predict(rf_model, X_test)
conf_matrix <- table(Predicted = predictions, Actual = y_test)

# Calculate performance metrics
accuracy <- mean(predictions == y_test)
precision <- diag(conf_matrix) / colSums(conf_matrix)
recall <- diag(conf_matrix) / rowSums(conf_matrix)
f1 <- 2 * (precision * recall) / (precision + recall)

print("\nModel Performance Metrics:")
print(paste("Overall Accuracy:", round(accuracy, 3)))
print("\nPer-Class Metrics:")
metrics_df <- data.frame(
  Precision = round(precision, 3),
  Recall = round(recall, 3),
  F1_Score = round(f1, 3)
)
print(metrics_df)

print("\nConfusion Matrix:")
print(conf_matrix)

# Feature importance analysis
importance_scores <- importance(rf_model)
var_importance <- data.frame(
  Feature = rownames(importance_scores),
  Importance = importance_scores[,1],
  Gini = importance_scores[,2]
)
var_importance <- var_importance %>%
  arrange(desc(Importance)) %>%
  head(20)

print("\nTop 20 Most Important Features:")
print(var_importance)

# Create more informative importance plot
ggplot(var_importance, aes(x = reorder(Feature, Importance))) +
  geom_segment(aes(xend = Feature, y = 0, yend = Importance), color = "grey50") +
  geom_point(aes(y = Importance), size = 3, color = "darkblue") +
  coord_flip() +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  ) +
  labs(
    x = "Feature",
    y = "Mean Decrease in Accuracy",
    title = "Top 20 Most Important Features for Predicting Country",
    subtitle = paste("Overall Model Accuracy:", round(accuracy, 3))
  )

# Plot class distribution
class_dist <- data.frame(table(y_train))
colnames(class_dist) <- c("Country", "Count")

ggplot(class_dist, aes(x = reorder(Country, -Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    x = "Country",
    y = "Number of Samples",
    title = "Distribution of Countries in Training Set"
  )

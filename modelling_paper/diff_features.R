# Read the CSV files
original_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
miq_features <- read.csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")

item_bank <- 
  read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/item-bank.csv") |>
  rename(item_id = id)

# Read in the participant responses (first 1e6, as too many will throttle
# my cpu)
df <- 
  read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")


# Load required libraries
library(gbm)
library(tidyverse)
library(caret)
library(ggplot2)

# Keep melody ID column
melody_ids <- original_features$melody_id

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

# library(PCAtest)
# pca_res <- PCAtest(feature_diffs, plot=FALSE)

pca.data <- prcomp(feature_diffs)
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

# Get first 5 PCs from PCA scores and add melody_ids back
pca_scores <- as.data.frame(pca.data$x[,1:5])
names(pca_scores) <- c("PC1", "PC2", "PC3", "PC4", "PC5") 
pca_scores$melody_id <- melody_ids

# Join PCA scores with participant data
df_with_pcs <- left_join(df, pca_scores, by = c("item_id" = "melody_id"))

# Handle NA values before model fitting
df_with_pcs <- na.omit(df_with_pcs)

# Print column names to check what we have
print("Column names in df_with_pcs:")
print(names(df_with_pcs))

# Calculate mean scores per item while preserving participant information
df_with_pcs <- df_with_pcs %>%
  group_by(item_id) %>%
  mutate(score_item = mean(score)) %>%
  ungroup()

library(lme4)
# Fit linear model using first 5 PCs to predict score
pc_model <- lmer(score_item ~ PC1 + PC2 + PC3 + PC4 + PC5 + (1|user_id),
                 data = df_with_pcs)
summary(pc_model)
# Calculate R-squared for the mixed model
# Method from Nakagawa & Schielzeth (2013)
library(MuMIn)
r2_values <- r.squaredGLMM(pc_model)
print("R-squared values for mixed model:")
print(paste("Marginal R2 (fixed effects):", round(r2_values[1], 3)))
print(paste("Conditional R2 (fixed + random effects):", round(r2_values[2], 3)))

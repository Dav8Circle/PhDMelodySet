# Read in the features data
item_features <- read.csv("item_features.csv")

# Remove melody_id column and any non-numeric columns
features_numeric <- item_features[, sapply(item_features, is.numeric)]
# Remove any rows with NA values
features_numeric <- na.omit(features_numeric)

# Scale the features before PCA
features_numeric <- features_numeric[, !names(features_numeric) %in% "tonality_features.inscale"]
features_scaled <- scale(features_numeric)
View(features_scaled)

# Perform PCA with bootstrapping and permutation tests
pca_res <- PCAtest(features_scaled, nboot = 1000, nperm = 1000, alpha = 0.05,
                   varcorr = TRUE, plot = TRUE)
View(pca_res)
# Print summary of results
print(summary(pca_res))

df2 <- features_numeric
pca_data <- prcomp(df2, scale = TRUE, center = TRUE, retx = T)

library(FactoMineR)
library(factoextra)
fviz_pca_var(pca_data,
             repel = TRUE,
             axes = c(1,2),
             show.legend.text = F,
             geom = "text")

# Create scree plot of PCA results

# Define feature categories based on column names from essen_features.csv
feature_categories <- data.frame(
  feature = colnames(df2),
  category = case_when(
    grepl("pitch_range|pitch_mean|pitch_std|pitch_mode", colnames(df2)) ~ "Pitch Statistics",
    grepl("entropy|information", colnames(df2)) ~ "Information Theory",
    grepl("interval", colnames(df2)) ~ "Intervals",
    grepl("contour", colnames(df2)) ~ "Contour",
    grepl("tonality", colnames(df2)) ~ "Tonality",
    grepl("rhythm|duration", colnames(df2)) ~ "Rhythm",
    TRUE ~ "Other"
  )
)

# Create PCA variable plot colored by category
fviz_pca_var(pca_data,
             col.var = feature_categories$category, 
             repel = TRUE,
             axes = c(1,2),
             title = "PCA Variables by Feature Type") +
  scale_color_brewer(palette = "Set2") +
  theme_minimal() +
  theme(legend.title = element_text(face = "bold"))



fviz_eig(pca_data, 
         addlabels = TRUE,
         ylim = c(0, 50),
         main = "Scree Plot of PCA Components",
         ylab = "Percentage of explained variances",
         xlab = "Principal Components")


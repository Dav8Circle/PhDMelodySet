library(tidyverse)
library(PCAtest)
setwd("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/Essen_Analysis")

# Read in features CSV
features <- read_csv("essen3.csv")
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

features_numeric <- features %>%
  select(-c(Original_Melody, country))

# Remove columns with zero variance
var_zero <- apply(features_numeric, 2, var)
features_numeric <- features_numeric[, var_zero != 0]
# Print features that were dropped due to zero variance
dropped_features <- names(var_zero[var_zero == 0])
cat("Features dropped due to zero variance:\n")
print(dropped_features)

# Remove melody_id 
features_numeric <- features_numeric %>% 
  select(-melody_id)

features_scaled <- scale(features_numeric)

pca_res <- PCAtest(features_scaled, nboot=1000, nperm=1000, alpha=0.05,
                   varcorr=TRUE, plot=FALSE)

pca_data <- prcomp(features_scaled)
# Get variable loadings for first 3 PCs
loadings <- pca_data$rotation[,1:5]

# Create data frame of loadings with variable names
loadings_df <- data.frame(
  variable = rownames(loadings),
  PC1 = loadings[,1],
  PC2 = loadings[,2],
  PC3 = loadings[,3],
  PC4 = loadings[,4],
  PC5 = loadings[,5]
)

# Add feature categories based on variable names
loadings_df <- loadings_df %>%
  mutate(category = case_when(
    grepl("pitch", variable) ~ "pitch_features",
    grepl("interval", variable) ~ "interval_features", 
    grepl("contour", variable) ~ "contour_features",
    grepl("duration", variable) ~ "duration_features",
    grepl("tonality", variable) ~ "tonality_features",
    grepl("narmour", variable) ~ "narmour_features",
    grepl("melodic_movement", variable) ~ "melodic_movement_features",
    grepl("mtype", variable) ~ "mtype_features",
    grepl("corpus", variable) ~ "corpus_features"
  ))

# Bind the PCA scores with the melody_ids
melody_scores <- bind_cols(
  melody_id = melody_ids,
  pca_data$x
)

selected_melodies <- 
  melody_scores %>%
  arrange(desc(PC4)) %>%
  slice(1:3) %>%
  left_join(melodies %>% select(melody_id = ID, melody_name = "Original Melody"), by = "melody_id") %>%
  select(melody_id, melody_name, PC4)
for (name in selected_melodies$melody_name) {
  file <- file.path("'/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)'", 
                    paste0(name, ".mid"))
  system2("open", file)
  Sys.sleep(3)
}

top_n_features_pc1 <-
  loadings_df %>%
  select(variable, PC1) %>%
  arrange(desc(abs(PC1))) %>%
  slice(1:10)

top_n_features_pc2 <-
  loadings_df %>%
  select(variable, PC2) %>%
  arrange(desc(abs(PC2))) %>%
  slice(1:10)


selected_melodies <- 
  features_numeric %>%
  arrange(desc(duration_features.note_density)) %>%
  select(melody_id, duration_features.note_density) %>%
  left_join(melodies %>% select(melody_id = ID, melody_name = "Original Melody"), by = "melody_id") %>%
  slice(1:1)

for (name in selected_melodies$melody_name) {
  file <- file.path("'/Users/davidwhyatt/Downloads/01_Essen Folksong Database (.mid-conversions)'", 
                    paste0(name, ".mid"))
  system2("open", file)
  Sys.sleep(3)
}



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

# Filter for just duration, pitch and interval categories
loadings_df_subset <- loadings_df %>%
  filter(category %in% c("duration_features", "contour_features", "interval_features", "pitch_features"))

# Plot loadings with plotly for interactive hover labels
plot_ly(loadings_df_subset, 
        x = ~PC1, 
        y = ~PC2,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#E41A1C", "#4DAF4A", "#984EA3", "#F781BF"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC1:", round(PC1, 3),
                      "<br>PC2:", round(PC2, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC1"),
         yaxis = list(title = "PC2"))


# Plot loadings with plotly for interactive hover labels
library(plotly)
plot_ly(loadings_df, 
        x = ~PC2, 
        y = ~PC3,
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
         xaxis = list(title = "PC2"),
         yaxis = list(title = "PC3"))

# Filter for just mtype/corpus and duration categories
loadings_df_subset <- loadings_df %>%
  filter(category %in% c("mtype_features", "corpus_features", 
                         "duration_features")) %>%
  mutate(category = ifelse(category %in% c("mtype_features", "corpus_features"), 
                          "complexity_features", category))

# Plot loadings with plotly for interactive hover labels
plot_ly(loadings_df_subset, 
        x = ~PC2, 
        y = ~PC3,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#377EB8", "#4DAF4A"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC1:", round(PC1, 3),
                      "<br>PC2:", round(PC2, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC2"),
         yaxis = list(title = "PC3"))


# Plot loadings with plotly for interactive hover labels
plot_ly(loadings_df, 
        x = ~PC3, 
        y = ~PC4,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", 
                   "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC3:", round(PC3, 3),
                      "<br>PC4:", round(PC4, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC3"),
         yaxis = list(title = "PC4"))

# Filter for just mtype, corpus and melodic_movement categories
loadings_df_subset <- loadings_df %>%
  filter(category %in% c("mtype_features", "corpus_features", 
                         "melodic_movement_features"))

# Plot loadings with plotly for interactive hover labels
plot_ly(loadings_df_subset, 
        x = ~PC3, 
        y = ~PC4,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#377EB8", "#FF700F", "#377EB8"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC3:", round(PC3, 3),
                      "<br>PC4:", round(PC4, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC3"),
         yaxis = list(title = "PC4"))

# Plot loadings with plotly for interactive hover labels
plot_ly(loadings_df, 
        x = ~PC4, 
        y = ~PC5,
        color = ~category,
        type = 'scatter',
        mode = 'markers',
        marker = list(size = 10),  # Increased marker size
        colors = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", 
                   "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"),
        text = ~paste("Variable:", variable,
                      "<br>Category:", category,
                      "<br>PC4:", round(PC4, 3),
                      "<br>PC5:", round(PC5, 3)),
        hoverinfo = 'text') %>%
  layout(title = "PCA Loadings by Feature Category",
         xaxis = list(title = "PC4"),
         yaxis = list(title = "PC5"))


# Create screeplot of first 10 principal components
fviz_eig(pca_data, 
         choice = "eigenvalue",
         ncp = 10,
         addlabels = TRUE,
         main = "Screeplot of First 10 Principal Components")



# Create biplot of observations and first 2 components
fviz_pca_biplot(pca_data,
                repel = TRUE,
                axes = c(1,2),
                geom.ind = "point",  # Show points for observations
                geom.var = "arrow",  # Show arrows for variables
                title = "Biplot of Observations and Variables",
                col.var = "red",   # Variable arrows in black
                alpha.var = 0.5,     # Semi-transparent arrows
                pointsize = 0.5,     # Smaller points for observations
                label = "var")       # Label the variable arrows

library(FactoMineR)
library(factoextra)
fviz_pca_var(pca_data,
             repel = TRUE,
             axes = c(1,2),
             show.legend.text = F,
             geom = "point")


# Hierarchical clustering ####
cor_features <- cor(features_numeric)
dist_features <- 1 - abs(cor_features)
hclust_features <- hclust(as.dist(dist_features))

dev.off()
jpeg("hclust.jpeg", width = 1200, height = 1200, quality=100)
plot(hclust_features)
dev.off()

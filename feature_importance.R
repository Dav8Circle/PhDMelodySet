library(plotly)
library(dplyr)

# Read the feature importance data
feature_importance <- read.csv("feature_importance.csv")

# Create interactive plot
p <- plot_ly(feature_importance, 
             x = ~reorder(feature, desc(importance)), 
             y = ~importance,
             type = "bar",
             text = ~round(importance, 4),
             textposition = "auto",
             hoverinfo = "text",
             texttemplate = "%{text}") %>%
  layout(title = "Feature Importance",
         xaxis = list(title = "Features",
                     tickangle = 45),
         yaxis = list(title = "Importance"),
         margin = list(b = 150)) %>%
  config(displayModeBar = FALSE)

p

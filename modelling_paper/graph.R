# Import the file again
df <- readxl::read_xlsx("~/Downloads/Diss Data.xlsx")
df <- df[!grepl("disregard", df$County, ignore.case=TRUE), ]
df <- df[!grepl("dsiregard", df$County, ignore.case=TRUE), ]
df <- df[!grepl("greater london", df$County, ignore.case=TRUE), ]

# Convert table to data frame for ggplot
county_df <- as.data.frame(table(df$County))
names(county_df) <- c("County", "Count")
county_df <- county_df[order(match(county_df$County, LETTERS), decreasing=TRUE),] # Z-A sort

# Create plot with ggplot2
library(ggplot2)
library(ggrepel)

# Create and save plot
ggplot(county_df[order(county_df$County),], aes(x=County, y=Count, fill=County)) +
  geom_bar(stat="identity") +
  coord_flip() + # Make horizontal
  theme_minimal() +
  theme(
    axis.text.y = element_text(size=6),
    axis.text.x = element_text(size=7),
    legend.position="none",
    plot.margin = margin(l = 0, unit = "pt") # Remove left margin
  ) +
  labs(
    title="Number of Records by County",
    x="County",
    y="Count"
  )

ggsave("county_records.png", width=10, height=8)

# Generate sample data
set.seed(42)  # For reproducibility
n <- 100
x <- rnorm(n, mean = 0, sd = 1)
y <- 2 * x + 1 + rnorm(n, mean = 0, sd = 0.5)

# Save x and y values to a file
write.table(data.frame(x = x, y = y), "R/xy_values.txt", row.names = FALSE, col.names = TRUE)

# Fit linear regression model
model <- lm(y ~ x)

# Print model summary
print(summary(model))

# Create visualization
pdf("R/regression_plot.pdf")
plot(x, y, main = "Linear Regression in R",
     xlab = "X", ylab = "Y",
     pch = 16, col = "blue")
abline(model, col = "red", lwd = 2)
dev.off()

# Save results to a file
sink("R/r_results.txt")
cat("R Linear Regression Results\n")
cat("=========================\n\n")
cat("Model Coefficients:\n")
print(coef(model))
cat("\nR-squared:", summary(model)$r.squared, "\n")
cat("Adjusted R-squared:", summary(model)$adj.r.squared, "\n")
cat("F-statistic:", summary(model)$fstatistic[1], "\n")
cat("P-value:", pf(summary(model)$fstatistic[1], 
                   summary(model)$fstatistic[2], 
                   summary(model)$fstatistic[3], 
                   lower.tail = FALSE), "\n")
sink()
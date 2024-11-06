library(tidyverse)
library(lme4)
library(psyphy)
library(nlme)

item_bank <- 
  read_csv("~/Downloads/PhD MELODY SET/item-bank.csv") |>
  rename(item_id = id)

df <- 
  read_csv("~/Downloads/PhD MELODY SET/miq_trials.csv", n_max = 1e6) |> 
  filter(test == "mdt") |> 
  left_join(item_bank |> select(- c("discrimination", "difficulty", "guessing", "inattention")), by = "item_id")



df <- na.omit(df)

scores_by_item <- 
  df |> 
  group_by(item_id) |> 
  summarise(
    score = mean(score)
  ) |> 
  left_join(item_bank, by = "item_id")


reg1 <- lm(score ~ item_id + oddity + difficulty + 
             displacement + length_crotchets + contour_dif + in_key,
           data = scores_by_item)


# Overall effect of oddity
scores_by_item |> 
  group_by(oddity) |> 
  summarise(
    score = mean(score)
  )


# As oddity has such a dramatic effect on mean score for a given item,
# lets instead consider it a factor in the regression 

reg2 <- lm(score ~ item_id + as.factor(oddity) + difficulty + 
             displacement + length_crotchets + contour_dif + in_key,
           data = scores_by_item)

plot(reg2)

# Problem: doesn't account for the adaptive nature of the test

reg3 <- lm(score ~ item_id + as.factor(oddity) + difficulty:length_crotchets
         + displacement + contour_dif + in_key, data = scores_by_item)

model1 <- glmer(score ~ difficulty + (1 | user_id), data=df, 
                family = binomial(mafc.logit(3)), verbose = 100)
model2 <- glmer(score ~ difficulty:length_crotchets + (1 | user_id), data=df,
                family = binomial(mafc.logit(3)), verbose = 100)

# Stratify by length
scores_by_item |> 
  group_by(
    oddity,
    num_notes
  ) |> 
  summarise(
    score = mean(score)
  ) |> 
  ggplot(
    aes(x = num_notes, y = score, colour = factor(oddity))
  ) + 
  geom_line()


scores_by_item |> 
  group_by(oddity, in_key) |> 
  summarise(
    score = mean(score)
  ) |> 
  ggplot(aes(x = in_key, colour = factor(oddity), fill = factor(oddity), y = score)) + 
  geom_bar(stat = "identity", position = "dodge")

df |> 
  group_by(answer) |> 
  summarise(answer = n() / nrow(df))

# Test whether proportions differ
prop_df <- df |> 
  group_by(answer) |> 
  summarise(proportion = n() / nrow(df))
observed <- table(df$answer)  # Count of each category in 'answer'
expected <- rep(nrow(df) / length(observed), length(observed))  # Expected proportions if all groups are equal

# Perform the chi-square test
chisq_test <- chisq.test(observed, p = expected, rescale.p = TRUE)
chisq_test

df |> names()

# df$score
# df$item_id
# df$user_id


# mod <-
#   glmer(score ~ difficulty + (1 | item_id) + (1|user_id),
#         data = df,
#         family = binomial(mafc.logit(3)), verbose = 100)

# 
# mod <- 
#   glmer(score ~ difficulty + (1 | item_id) + (1|user_id),
#         data = df,
#         family = binomial())

# bias free estimation

library(dplyr)

# Load data (make sure to use correct file path)
data <- read_csv("Downloads/PhD MELODY SET/miq_trials.csv", n_max=1e6)

# Create new variables and recode y
data <- data %>%
  mutate(
    trial = row_number(),
    y1 = 2 - ytemp1,
    y2 = 2 - ytemp2,
    z = x1 - x2,
    z1 = 1 - 2 * x1 - x2,
    z2 = 1 - x1 - 2 * x2
  ) %>%
  gather(key = "i", value = "y", y1, y2) %>%  # Reshape the data into long format
  mutate(
    i1 = ifelse(i == "y1", 1, 0),
    i2 = ifelse(i == "y2", 1, 0)
  ) %>%
  select(trial, z, z1, z2, i1, i2, y)  # Select the necessary variables
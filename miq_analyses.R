library(tidyverse)
library(lme4)
library(psyphy)
library(nlme)
library(lmtest)

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

# express scores as mean for each item and store alongside other predictors

df2 <- df %>%
  group_by(item_id) %>%
  mutate(
    score_item = mean(score))

# As oddity has such a dramatic effect on mean score for a given item,
# lets instead consider it a factor in the regression 

reg2 <- lm(score ~ item_id + as.factor(oddity) + difficulty + 
             displacement + length_crotchets + contour_dif + in_key,
           data = scores_by_item)

plot(reg2)

# Problem: doesn't account for the adaptive nature of the test

reg3 <- lm(score_item ~ item_id + as.factor(oddity) + difficulty:length_crotchets 
           + displacement + contour_dif + in_key, data = df2)

# since it's adaptive and we're updating these parameters using WL, maybe
# this is a good predictor?

reg4 <- lm(score_item ~ item_id + as.factor(oddity) + difficulty:length_crotchets
           + displacement + contour_dif + in_key + ability_WL, data=df2)


# check the effect of the interaction term
reg5 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif +
             in_key + ability_WL, data=df2)

# reinsert the difficulty predictor
reg6 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
             in_key + difficulty + ability_WL, data=df2)


names(df2)

# add the participant predictor - result is N.S. and doesn't really make sense
# to do

reg7 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
             in_key + difficulty + user_id + ability_WL, data=df2)

reg8 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
             in_key + change_note + difficulty + ability_WL, data=df2)

reg8a <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
              in_key + change_note + ability_WL, data=df2)

reg8b <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
              in_key + change_note + ability_WL + num_notes, data=df2)

# Produce a plot of reg8a against the data it's fitted to
plot(df2$score_item, predict(reg8a))

anova(reg8, reg8a, reg8b)
anova(reg8a, reg8b)

# 8b preferred

# Visualise the difference between reg8a and reg8b
plot(df2$score_item, predict(reg8a))


reg9 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
             in_key + change_note + ability_WL, data=df2)

coxtest(reg8, reg8a)
jtest(reg8, reg9)

reg10 <- lm(score_item ~ as.factor(oddity) + displacement + contour_dif + 
              change_note + num_notes + ability_WL, data=df2)

model1 <- lmer(score_item ~ difficulty + (1 | user_id), data=df2,
               verbose = 100)

model2 <- lmer(score_item ~ difficulty:length_crotchets + (1 | user_id), data=df2, verbose = 100)

model3 <- lmer(score_item ~ as.factor(oddity) + ability_WL + (1 | user_id), data=df2, verbose = 100)

model4 <- lmer(score_item ~ as.factor(oddity) + ability_WL + num_notes + (1 | user_id), data=df2)

model5 <- glmer(score ~ as.factor(oddity) + ability_WL + num_notes + (1 | user_id), data=df2,
                family = binomial(mafc.logit(3)), verbose = 100)

anova(model1, model2, model3, model4, model5)


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

# T-test whether the mean score for oddity 1 is different from oddity 2
t.test(df |> filter(oddity == 1) |> pull(score), df |> filter(oddity == 2) |> pull(score))

# T-test whether the mean score for oddity 2 is different from oddity 3
t.test(df |> filter(oddity == 2) |> pull(score), df |> filter(oddity == 3) |> pull(score))

# T-test whether the mean score for oddity 1 is different from oddity 3
t.test(df |> filter(oddity == 1) |> pull(score), df |> filter(oddity == 3) |> pull(score))

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


# Another way of looking at this kind of data - 
# Instead of creating linear models, which filter everything through
# a pre-existing set of hypothesised features, 
# we can just compute average scores for each item and compare
# the hardest and easiest items by listening to them.

tmp <- 
  df2 %>% 
  group_by(item_id) %>% 
  summarise(
    item_index = unique(item_index), 
    difficulty = unique(difficulty), 
    file_name = unique(file_name),
    score = mean(score), 
    n = n()
  ) %>% 
  filter(item_index == 10) %>% 
  arrange(score)

tmp$file_name %>% first()
tmp$file_name %>% last()



# TODO - IDyOM features?
# TODO - FANTASTIC features?
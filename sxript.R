# Library ====
library(skimr)
library(tidyverse)
library(tidymodels)
library(GGally)
library(funModeling)
library(janitor)
# install.packages("corrr")
library(corrr)
library(corrplot)
library(vip)
library(rpart)
library(rpart.plot)
library(ranger)
library(klaR)
library(discrim)
library(kknn)
# install.packages("doParallel")
library(doParallel)


# Loading the data set ====
car_claims = read.csv("train.csv")

# Decision Variable ====
car_claims$is_claim = ifelse(car_claims$is_claim == 1, "Yes", "No")

# Managing Factor Variables ====
name = car_claims %>% 
  select_if(is.character) %>% 
  names()

car_claims[, name] = lapply(car_claims[, name], factor)

yes_no = car_claims %>% 
  select(starts_with("is")) %>% 
  names()

yes_no

car_claims$is_esc = fct_relevel(car_claims$is_esc, "Yes")
car_claims$is_adjustable_steering = fct_relevel(car_claims$is_adjustable_steering, "Yes")
car_claims$is_tpms = fct_relevel(car_claims$is_tpms, "Yes")
car_claims$is_parking_sensors = fct_relevel(car_claims$is_parking_sensors, "Yes")
car_claims$is_parking_camera = fct_relevel(car_claims$is_parking_camera, "Yes")
car_claims$is_front_fog_lights = fct_relevel(car_claims$is_front_fog_lights, "Yes")
car_claims$is_rear_window_wiper = fct_relevel(car_claims$is_rear_window_wiper, "Yes")
car_claims$is_rear_window_washer = fct_relevel(car_claims$is_rear_window_washer, "Yes")
car_claims$is_rear_window_defogger = fct_relevel(car_claims$is_rear_window_defogger, "Yes")
car_claims$is_brake_assist = fct_relevel(car_claims$is_brake_assist, "Yes")
car_claims$is_power_door_locks = fct_relevel(car_claims$is_power_door_locks, "Yes")
car_claims$is_central_locking = fct_relevel(car_claims$is_central_locking, "Yes")
car_claims$is_power_steering = fct_relevel(car_claims$is_power_steering, "Yes")
car_claims$is_driver_seat_height_adjustable = fct_relevel(car_claims$is_driver_seat_height_adjustable, "Yes")
car_claims$is_day_night_rear_view_mirror = fct_relevel(car_claims$is_day_night_rear_view_mirror, "Yes")
car_claims$is_ecw = fct_relevel(car_claims$is_ecw, "Yes")
car_claims$is_speed_alert = fct_relevel(car_claims$is_speed_alert, "Yes")
car_claims$is_claim = fct_relevel(car_claims$is_claim, "Yes")

summary(car_claims)

# Analyzing numeric and non-numeric variables ====
## Non-Numeric Analysis ====
freq(car_claims)

## Numeric Analysis ====
plot_num(car_claims)

# Split the data ====
set.seed(123)
claims_split = initial_split(car_claims, prop = 0.75, strata = is_claim)
claims_split

claims_training = claims_split %>% training()
claims_testing = claims_split %>% testing()

set.seed(123)
claims_folds = vfold_cv(claims_training, v = 5)
# Decision Tree ====
## Recipe ====
claims_tree_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "id variable") %>%
  step_normalize(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_dummy(all_nominal(), -has_role("id variable"), -all_outcomes())

claims_tree_recipe %>% prep() %>% bake(new_data = claims_training) %>% view()

## Tree Model ====
tree_model = decision_tree(cost_complexity = tune(),
                           tree_depth = tune(),
                           min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

## Tree Workflow ====
tree_wf = workflow() %>% 
  add_model(tree_model) %>% 
  add_recipe(claims_tree_recipe)

## Hyper parameter tuning ====
### Creating grid ====
tree_grid = grid_regular(cost_complexity(),
                         tree_depth(),
                         min_n(),
                         levels = 2)
### Tuning Grid ====
set.seed(123)
tree_tuning = tree_wf %>% 
  tune_grid(resamples = claims_folds,
            grid = tree_grid)

### Seeing Results ====
tree_tuning %>% show_best("roc_auc")

best_tree = tree_tuning %>% select_best("roc_auc")
best_tree

# ## Fit model ====
# tree_wf_fit = tree_wf %>% 
#   fit(data = claims_training)
# 
# tree_fit = tree_wf_fit %>% 
#   pull_workflow_fit()
# 
# vip(tree_fit)
# 
# rpart.plot(tree_fit$fit, roundint = FALSE)

## Finalize workflow ====
final_tree_workflow = tree_wf %>% finalize_workflow(best_tree)

## Visualize Results ====
tree_wf_fit = final_tree_workflow %>% 
  fit(data = claims_training)

### Important Variables ====
vip(tree_wf_fit)

rpart.plot(tree_wf_fit$fit, roundint = F)
rpart(tree_wf_fit$fit)

## Last fit ====
tree_last_fit = final_tree_workflow %>% 
  last_fit(claims_split)

tree_last_fit %>% collect_metrics()

tree_last_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = is_claim, estimate = .pred_Yes) %>% 
  autoplot()

## Confusion Matrix ====
tree_predictions = tree_last_fit %>% 
  collect_predictions()

conf_mat(tree_predictions, truth = is_claim, estimate = .pred_class)

# # Random Forest (Revisit later) ====
# ## Random Recipe ====
# random_recipe = recipe(is_claim ~ ., data = claims_training) %>%
#   update_role(policy_id, new_role = "id variable") %>%
#   step_normalize(all_numeric(), -all_outcomes(), -has_role("id variable")) %>%
#   step_dummy(all_nominal(), -all_outcomes(), -has_role("id variable"))
# 
# ## Random Model ====
# random_model = rand_forest(mtry = tune(),
#                            trees = tune(),
#                            min_n = tune()) %>%
#   set_engine("ranger", importance = "impurity") %>%
#   set_mode("classification")
# 
# ## Random Workflow ====
# random_wf = workflow() %>%
#   add_model(random_model) %>%
#   add_recipe(random_recipe)
# 
# 
# ## Hyper parameter tuning ====
# set.seed(123)
# random_grid = grid_random(mtry() %>%  range_set(c(6, 34)),
#                             trees(),
#                             min_n(),
#                             size = 2)
# 
# random_tuning = random_wf %>%
#   tune_grid(resamples = claims_folds,
#             grid = random_grid)



# Logistic Regression ====
## Logistic Recipe ====
# After looking at all the numeric variables, I can safely say that only population_density and age_of_car need square root distribution while others need Yeo Johnson distribution. Therefore, we will incorporate it into our recipe.

logistic_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "id variable") %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("id variable")) %>% 
  step_corr(all_numeric(), -all_outcomes(), -has_role("id variable"))

## Logistic Model ====
logistic_model = logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

## Logistic workflow ====
logistic_wf = workflow() %>% 
  add_model(logistic_model) %>% 
  add_recipe(logistic_recipe)

## Logistic Tune ====
# logistic_tuning = logistic_wf %>% 
#   tune_grid(resamples = claims_folds)
# 
# remove(logistic_tuning)

## Logistic fit ====
logistic_fit = logistic_wf %>% 
  fit(data = claims_training)

summary(logistic_fit)

vip(logistic_fit$pre)

logistics_last_fit = logistic_wf %>% 
  last_fit(claims_split)

logistics_last_fit %>% collect_metrics()

logistics_last_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)

# Lasso Logistic Regression ====
## Lasso Recipe ====
lasso_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "id variable") %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("id variable"))

## Lasso Logistic Model ====
lasso_log_reg = logistic_reg(penalty = 0.1, mixture = 1) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

## Lasso Logistic Workflow ====
lasso_log_wf = workflow() %>% 
  add_model(lasso_log_reg) %>% 
  add_recipe(lasso_recipe)

## Lasso fit ====
lasso_log_last_fit = lasso_log_wf %>% 
  last_fit(claims_split)

lasso_log_last_fit %>% collect_metrics()

lasso_log_last_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)


# Linear Discriminant Analysis ====
## Linear Discriminant Analysis (LDA) Recipe ====
LDA_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "id variable") %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_normalize(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("id variable"))

## LDA model ====
lda_model = discrim_regularized(frac_common_cov = 1) %>% 
  set_engine("klaR") %>% 
  set_mode("classification")

## LDA workflow ====
lda_wf = workflow() %>% 
  add_recipe(LDA_recipe) %>% 
  add_model(lda_model)

## Evaluate model ====
last_fit_lda = lda_wf %>% 
  last_fit(split = claims_split)

last_fit_lda %>% collect_metrics()

last_fit_lda %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)

# Quadratic Discriminant Analysis ====
## Quadratic Discriminant Analysis (QDA) Model ====
 qda_model = discrim_regularized(frac_common_cov = 0) %>% 
  set_engine("klaR") %>% 
  set_mode("classification")

## QDA Workflow ====
qda_wf = workflow() %>% 
  add_model(qda_model) %>% 
  add_recipe(LDA_recipe)

## Last fit QDA ====
last_fit_qda = qda_wf %>% 
  last_fit(split = claims_split)

last_fit_qda %>% collect_metrics()

last_fit_qda %>% collect_predictions() %>% conf_mat(truth = is_claim, estimate = .pred_class)

# K-Nearest Neighbors ====
## KNN Model ====
knn_model = nearest_neighbor(neighbors = 2) %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

## KNN workflow ====
knn_wf = workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(LDA_recipe)

## KNN last fit ====
knn_last_fit = knn_wf %>% 
  last_fit(claims_split)

knn_last_fit %>% 
  collect_metrics()

knn_last_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)

# Random Forest without folds ====
## RF1 Recipe ====
RF1_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "id variable") %>% 
  step_other(model, max_torque, max_power, engine_type, threshold = 0.1, other = "other values") %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_normalize(all_numeric(), -all_outcomes(), -has_role("id variable")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("id variable"))


## RF1 Model ====
rf_model = rand_forest(mtry = tune(),
                       trees = tune(),
                       min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

## RF1 Grid search ====
set.seed(123)

rf_grid = grid_random(mtry() %>% range_set(c(4,40)),
                      trees(),
                      min_n(),
                      size = 10
                      )

rf_grid

## RF1 Workflow ====
rf_workflow = workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(RF1_recipe)

## RF1 Tuning ====
rf_tuning = rf_workflow %>% 
  tune_grid(grid = rf_grid,
            resamples = claims_folds)

rf_tuning %>% show_best("roc_auc")

best_rf = rf_tuning %>% 
  select_best(metric = "roc_auc")

## RF1 finalize workflow ====
final_rf_workflow = rf_workflow %>% 
  finalize_workflow(best_rf)

## RF workflow fit ====
rf_wf_fit = final_rf_workflow %>% 
  fit(data = claims_training)

rf_fit = rf_wf_fit %>% 
  extract_fit_parsnip()

vip(rf_fit)

## RF last fit ====
rf_last_fit = final_rf_workflow %>% 
  last_fit(claims_split)

rf_last_fit %>% collect_metrics()

rf_last_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)

rf_last_fit %>% 
  collect_predictions() %>% 
  roc_curve(truth = is_claim, estimate = .pred_Yes) %>% 
  autoplot()


# Refining the model ====
funModeling::plot_num(car_claims)

car_claims %>% 
  ggplot(aes(x = ncap_rating)) + 
  geom_density()

recipe(is_claim ~ . , data = car_claims) %>% 
  step_YeoJohnson(all_numeric(), -all_outcomes()) %>% 
  prep() %>% bake(new_data = car_claims) %>% 
  ggplot(aes(x = ncap_rating)) + geom_density()

## Recipe for variable distribution ====
refine_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "ID") %>% 
  step_sqrt(age_of_car, population_density, height, gross_weight) %>% 
  step_YeoJohnson(age_of_policyholder) %>% 
  step_normalize(all_numeric(), -all_outcomes(), -has_role("ID")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("ID"))

refine_recipe %>% 
  prep() %>% 
  bake(new_data = car_claims) %>% 
  ggplot(aes(x = age_of_policyholder)) +
  geom_density()

## Refine QDA Analysis ====
refine_qda_workflow = workflow() %>% 
  add_model(qda_model) %>% 
  add_recipe(refine_recipe)

refine_qda_last_fit = refine_qda_workflow %>% 
  last_fit(split = claims_split)

refine_qda_last_fit %>% collect_metrics()

refine_qda_last_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = is_claim, estimate = .pred_class)

# Without Distribution analysis ====
without_recipe = recipe(is_claim ~ ., data = claims_training) %>% 
  update_role(policy_id, new_role = "ID") %>% 
  step_normalize(all_numeric(), -all_outcomes(), -has_role("ID")) %>% 
  step_dummy(all_nominal(), -all_outcomes(), -has_role("ID"))

without_qda_model = workflow() %>% 
  add_model(qda_model) %>% 
  add_recipe(without_recipe)

without_qda_lastfit = without_qda_model %>% 
  last_fit(split = claims_split)

# Testing on unseen data ====
car_claims_test = read.csv("test.csv")


names(car_claims_test)
names(car_claims)

qda_fit = qda_wf %>% 
  fit(data = car_claims)

predict_claims = predict(qda_fit, car_claims_test)

table(predict_claims)


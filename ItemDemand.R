# Item Demand

# setwd("C:/Users/farri/OneDrive/Documents/PA/ItemDemand")

# LIBRARIES=====================================================================

library(tidyverse)
library(vroom)
library(tidymodels)
library(discrim)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(recipes)
library(embed) 
library(naivebayes)
library(kknn)
library(themis) # for smote
library(timetk)
library(forecast)
library(modeltime) 
library(bonsai)

test <- vroom("test.csv")
train <- vroom("train.csv")
sampleSubmission <- vroom("sample_submission.csv")

# 1=============================================================================

## Filter down to just 1 store item for exploration and model building5
storeItem_1 <- train %>%
filter(store==1, item==1)
storeItemTest_1 <- test %>%
  filter(store==1, item==1)

storeItem_2 <- train %>%
  filter(store==2, item==2)
storeItem_3 <- train %>%
  filter(store==3, item==3)
storeItem_4 <- train %>%
  filter(store==4, item==5)


storeItem_1 %>%
  plot_time_series(date, sales, .interactive=FALSE)

storeItem_1 %>%
pull(sales)
forecast::ggAcf(train$sales, lag.max=28)
ggAcf(storeItem_1$sales, lag.max=28)
ggAcf(storeItem_2$sales, lag.max=28)
ggAcf(storeItem_3$sales, lag.max=28)
ggAcf(storeItem_4$sales, lag.max=28)

library(patchwork)
(ggAcf(storeItem_1$sales, lag.max=28) + ggAcf(storeItem_2$sales, lag.max=28)) / (ggAcf(storeItem_3$sales, lag.max=28) + ggAcf(storeItem_4$sales, lag.max=28)) #4 panel plot



View(storeItem_1)

# KNN===========================================================================

Recipe_KNN <- recipe(sales ~ ., data = storeItem_1) %>%
  step_date(date, features="dow")

knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("regression") %>%
  set_engine("kknn") 

knn_wf <- workflow() %>%
  add_recipe(Recipe_KNN) %>%
  add_model(knn_model)%>%
  fit(data=storeItem_1) 

grid <- grid_regular(neighbors(),
                     levels = 5) 

folds <- vfold_cv(storeItem_1, v = 5, repeats=1)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=grid,
            metrics=metric_set(smape)) #Or leave metrics NUL

bestTune <- CV_results %>%
  select_best("smape")

wf_prep <- workflow() %>%
  add_recipe(Recipe_KNN) %>%
  add_model(knn_model) %>%
  fit(data=train) 

# Plots=========================================================================


          cv_split1 <- time_series_split(storeItem_1, assess="3 months", cumulative = TRUE)
          
          cv_split1 %>%
          tk_time_series_cv_plan() %>% #Put into a data frame
            plot_time_series_cv_plan(date, sales, .interactive=FALSE)
          
          
          es_model <- exp_smoothing() %>%
          set_engine("ets") %>%
          fit(sales~date, data = training(cv_split1))
          
          ## Cross-validate to tune model
          cv_results <- modeltime_calibrate(es_model,
                                            new_data = testing(cv_split1))
          
          ## Visualize CV results
          p1 <- cv_results %>%
          modeltime_forecast(
                             new_data = testing(cv_split1),
                             actual_data = storeItem_1
          ) %>%
          plot_modeltime_forecast(.interactive=TRUE)

# ========== PLOT 2
          cv_split2 <- time_series_split(storeItem_4, assess="3 months", cumulative = TRUE)
          
          cv_split2 %>%
            tk_time_series_cv_plan() %>% #Put into a data frame
            plot_time_series_cv_plan(date, sales, .interactive=FALSE)
          
          
          es_model <- exp_smoothing() %>%
            set_engine("ets") %>%
            fit(sales~date, data = training(cv_split2))
          
          ## Cross-validate to tune model
          cv_results2 <- modeltime_calibrate(es_model,
                                            new_data = testing(cv_split2))
          
          ## Visualize CV results
          p2 <- cv_results %>%
            modeltime_forecast(
              new_data = testing(cv_split2),
              actual_data = storeItem_4
            ) %>%
            plot_modeltime_forecast(.interactive=TRUE)

# PLOT 3
## Evaluate the accuracy
cv_results %>%
modeltime_accuracy() %>%
table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit <- cv_results %>%
modeltime_refit(data = storeItem_1)

es_preds <- es_fullfit %>%
modeltime_forecast(h = "3 months") %>%
rename(date=.index, sales=.value) %>%
select(date, sales) %>%
full_join(., y=test, by="date") %>%
select(id, sales)

p3 <- es_fullfit %>%
modeltime_forecast(h = "3 months", actual_data = storeItem_1) %>%
plot_modeltime_forecast(.interactive=FALSE)

# PLOT 4
## Evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit <- cv_results2 %>%
  modeltime_refit(data = storeItem_4)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p4 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem_4) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1,p2,p3,p4,nrows = 2)

# ARIMA=========================================================================

## Create the CV split for time series
cv_split1 <- time_series_split(storeItem_1, assess="3 months", cumulative = TRUE)
cv_split2 <- time_series_split(storeItem_4, assess="3 months", cumulative = TRUE)

## Create a recipe for the linear model part
arima_recipe <- recipe(sales ~ ., data = storeItem_1) %>%
  step_date(date, features="dow") # For the linear model part

## Define the ARIMA Model
arima_model <- arima_reg(seasonal_period=2,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
set_engine("auto_arima")

## Merge into a single workflow and fit to the training data
arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=storeItem_1) 

  
# Calibrate (tune) the models (find p,d,q,P,D,Q)
cv_results <- modeltime_calibrate(arima_wf,
                                    new_data = testing(cv_split1))

## Visualize results (use previous code)


## Now that you have calibrated (tuned) refit to whole dataset
fullfit <- cv_results %>%
modeltime_refit(data=train)

## Predict for all the observations in storeItemTest
fullfit %>%
modeltime_forecast(
                   new_data = storeItemTest,
                   actual_data = storeItemTrain
) %>%
plot_modeltime_forecast(.interactive=TRUE)


# ========== PLOT 1
cv_split1 <- time_series_split(storeItem_1, assess="3 months", cumulative = TRUE)

arima_recipe <- recipe(sales ~ ., data = storeItem_1) %>%
  step_date(date, features="dow") # For the linear model part

arima_model <- arima_reg(seasonal_period=2,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=storeItem_1) 

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split1))

p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = storeItem_1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# ========== PLOT 2
cv_split2 <- time_series_split(storeItem_4, assess="3 months", cumulative = TRUE)

arima_recipe <- recipe(sales ~ ., data = storeItem_4) %>%
  step_date(date, features="dow") # For the linear model part

arima_model <- arima_reg(seasonal_period=2,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=storeItem_4) 

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split2))

p2 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem_4
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# ================================================PLOT 3

fullfit <- cv_results %>%
  modeltime_refit(data=storeItem_1)

p3 <- fullfit %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = storeItem_1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# PLOT 4
fullfit <- cv_results %>%
  modeltime_refit(data=storeItem_4)

p4 <- fullfit %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem_4
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plotly::subplot(p1,p2,p3,p4,nrows = 2)

# PROPHET=======================================================================

cv_split1 <- time_series_split(storeItem_1, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
set_engine(engine = "prophet") %>%
fit(sales ~ date, data = training(cv_split1))

## Calibrate (i.e. tune) workflow
cv_results1 <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split1))

## Refit best model to entire data and predict

best_model <- cv_results1 %>%
  modeltime_refit(data = storeItem_1)

p3 <- best_model %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem_1) %>%
  plot_modeltime_forecast(.interactive=FALSE)

p1 <- cv_results1 %>%
  modeltime_forecast(
    new_data = testing(cv_split1),
    actual_data = storeItem_1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)


# 2 & 4==============
cv_split2 <- time_series_split(storeItem_4, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split2))

## Calibrate (i.e. tune) workflow
cv_results2 <- modeltime_calibrate(prophet_model,
                                   new_data = testing(cv_split2))

## Refit best model to entire data and predict

best_model2 <- cv_results2 %>%
  modeltime_refit(data = storeItem_4)

p4 <- best_model2 %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem_4) %>%
  plot_modeltime_forecast(.interactive=FALSE)

p2 <- cv_results1 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem_4
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

















# Model Testing ================================================================

item <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
itemTest <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")
n.stores <- max(item$store)
n.items <- max(item$item)

## Define the workflow
item_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "year")) %>%
  step_mutate(Season = case_when(
    month(date) %in% c(3, 4, 5) ~ "Spring",
    month(date) %in% c(6, 7, 8) ~ "Summer",
    month(date) %in% c(9, 10, 11) ~ "Fall",
    month(date) %in% c(12, 1, 2) ~ "Winter")) %>% 
  step_rm(date, item, store)
# step_normalize(all_numeric_predictors())


boosted_model <- boost_tree(tree_depth=1, #Determined by random store-item combos
                            trees=1000,
                            learn_rate=0.1) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(boosted_model)

for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i)
    test <- itemTest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- boost_wf %>%
      fit(data=train)
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
  }
}

vroom_write(x=all_preds, "./submission.csv", delim=",")
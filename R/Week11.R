#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))#error in here so I use the button on the Files tab
library(tidyverse)
library(haven)
library(caret)
library(parallel)
library(doParallel)

#Data Import and Cleaning
gss<-read_sav("../data/GSS2016.sav")
gss_tbl<-gss %>% 
  drop_na(MOSTHRS) %>% 
  rename(`work hours`=MOSTHRS) %>% 
  select(-c(HRS1, HRS2)) %>% 
  select(where(~ mean(is.na(.)) < 0.75))
gss_tbl$`work hours`<-as.numeric(gss_tbl$`work hours`)

#Visualization
ggplot(gss_tbl, aes(x = `work hours`)) +
  geom_histogram()+
  labs(title = "Distribution of Work Hours",
       x = "Work Hours",
       y = "Frequency")

#Analysis
set.seed(0220)
holdout_indices <- createDataPartition(gss_tbl$`work hours`,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,]
training_tbl <- gss_tbl[-holdout_indices,]

training_folds <- createFolds(training_tbl$`work hours`)

#OLS regression model
ols_model <- train(`work hours` ~ ., 
                   data = training_tbl, 
                   method="lm",
                   na.action = na.pass,
                   preProcess = c("center","scale","zv","nzv","medianImpute"),
                   trControl = trainControl(method="cv", 
                                            number=10, 
                                            verboseIter=T, 
                                            indexOut = training_folds)
)

# Elastic Net model
elastic_net_model <- train(`work hours` ~ ., 
                           data = training_tbl, 
                           method="glmnet",
                           na.action = na.pass,
                           preProcess = c("center","scale","zv","nzv","medianImpute"),
                           trControl = trainControl(method="cv", 
                                                    number=10, 
                                                    verboseIter=T, 
                                                    indexOut = training_folds)
)

# Random Forest model
rf_model<-train(`work hours` ~ ., 
                training_tbl,  
                method="ranger",
                na.action = na.pass,
                tuneLength = 1,
                preProcess = c("center","scale","zv","nzv","medianImpute"),
                trControl = trainControl(method="cv", 
                                         number=10, 
                                         verboseIter=T, 
                                         indexOut = training_folds))


#eXtreme Gradient Boosting model
xgb_model <- train(`work hours` ~ ., 
                   training_tbl,  
                   method="xgbLinear",
                   na.action = na.pass,
                   tuneLength = 1,
                   preProcess = c("center","scale","zv","nzv","medianImpute"),
                   trControl = trainControl(method="cv", 
                                            number=10, 
                                            verboseIter=T, 
                                            indexOut = training_folds))



cv_ols <- max(ols_model$results$Rsquared)
cv_net<-max(elastic_net_model$results$Rsquared)
cv_rf<-max(rf_model$results$Rsquared)
cv_xgb<-max(xgb_model$results$Rsquared)


#CV prediction
ols_pred <- predict(ols_model, training_tbl, na.action=na.pass)
net_pred <- predict(elastic_net_model, training_tbl, na.action=na.pass)
rf_pred <- predict(rf_model, training_tbl, na.action=na.pass)
xgb_pred <- predict(xgb_model, training_tbl, na.action=na.pass)

# Evaluate holdout CV
ols_rmse<-cor(training_tbl$`work hours`, ols_pred)^2
net_rmse<-cor(training_tbl$`work hours`, net_pred)^2
rf_rmse<-cor(training_tbl$`work hours`, rf_pred)^2
xgb_rmse<-cor(training_tbl$`work hours`, xgb_pred)^2


#create a table 1
# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    make_it_pretty(cv_ols),
    make_it_pretty(cv_net),
    make_it_pretty(cv_rf),
    make_it_pretty(cv_xgb)
  ),
  ho_rqs = c(
    make_it_pretty(ols_rmse),
    make_it_pretty(net_rmse),
    make_it_pretty(rf_rmse),
    make_it_pretty(xgb_rmse)
  )
)

table1_tbl

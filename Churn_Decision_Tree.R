

Credit = read.csv("C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 10/Churn_Modelling.csv")
names(Credit)
Credit <- Credit %>% select(-RowNumber:-Surname)
Credit$Gender = as.factor(Credit$Gender)
Credit$Geography = as.factor(Credit$Geography)
summary(Credit)

sam = sample(1:10000,7000,replace=F)

C.train = Credit[sam,]
C.test = Credit[-sam,]
require(Matrix)

Train.X = sparse.model.matrix(Exited~.-1,data=C.train)
Test.X = sparse.model.matrix(Exited~.-1,data=C.test)
Train.Y = as.list(C.train$Exited)
Test.Y = as.list(C.test$Exited)

require(xgboost)

dtrain = xgb.DMatrix(data=Train.X,label=Train.Y)
dtest = xgb.DMatrix(data=Test.X,label=Test.Y)

watchlist1 = list(train=dtrain,test=dtest)

mod1 = xgb.train(data=dtrain,
                 max.depth=4,eta=.2,nthread=6,nround=50,
                 watchlist=watchlist1,objective="binary:logistic",
                 eval_metric="error",eval_metric="logloss",eval_metric="auc")

cv.train = xgb.cv(data=dtrain,nfold=5,nthread=6,
                  max.depth=3,eta=.05,nround=200,
                  objective="binary:logistic",eval_metric="auc")

finmod = xgboost(data=dtrain,nthread=6,max.depth=3,
                 eta=.05,nround=200,
                 objective="binary:logistic",
                 eval_metric="auc")

phat = predict(finmod,Test.X)
require(ROCR)

pred = prediction(phat,C.test$Exited)
perf = performance(pred,"tpr","fpr")
plot(perf)

perf2 = performance(pred,"auc")
perf2



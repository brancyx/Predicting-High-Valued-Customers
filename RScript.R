#========================================================================================
# Project: Improving WhiteRock's Client Onboarding
# Aim: As an analytics consultant for WhiteRock, improve its intermediaries' client onboarding processes
# Model: Classification and Regression Tree
# Inputs: Gender, Geography, CreditScore, EstimatedSalary, Age 
# Output: CustomerValue (High or Low)
# Data/ data source : Churn_Modelling.csv
# Author: Brandon Chen, Joelle Lee, Jerry Lee, Shwe Sin, Janzy Chua
# Date: 01.11.2020
#========================================================================================

#-------- Import Data & Libraries --------
library(data.table)
library(nnet)
library(caTools)
library(ggplot2)
library(rpart)
library(rpart.plot)
setwd("BC2406/Team Assignment and Project")
churn.dt <- fread("Churn_Modelling.csv", stringsAsFactors = TRUE)


#-------- Data Cleaning --------

# Checking for NA values
summary(churn.dt) # No logical errors identified
sum(is.na(churn.dt)) # Returned 0 NA values

# Changing binary values '0' to 'no' and '1' to 'yes' for easier visualization
churn.dt$HasCrCard <- replace(churn.dt$HasCrCard, churn.dt$HasCrCard == 1, "Yes" )
churn.dt$HasCrCard <- replace(churn.dt$HasCrCard, churn.dt$HasCrCard == 0, "No" )
churn.dt$IsActiveMember <- replace(churn.dt$IsActiveMember, churn.dt$IsActiveMember == 1, "Yes" )
churn.dt$IsActiveMember <- replace(churn.dt$IsActiveMember, churn.dt$IsActiveMember == 0, "No" )

# Add CustomerValue column (if all conditions met, Customer has 'High' value)
churn.dt[,CustomerValue := ifelse(NumOfProducts > 1 & HasCrCard == "Yes" & IsActiveMember == "Yes", 'High', 'Low')]

# Convert data type of relevant variables to factors
churn.dt$CustomerValue<-factor(churn.dt$CustomerValue)
churn.dt$Gender <- factor(churn.dt$Gender)
churn.dt$Geography <- factor(churn.dt$Geography)
churn.dt$HasCrCard<-factor(churn.dt$HasCrCard)
churn.dt$IsActiveMember<-factor(churn.dt$IsActiveMember)
churn.dt$Exited <-factor(churn.dt$Exited)


#-------- Data Exploration & Insights --------

##---- General Insights
# Relationship between Active status and Age
j1 <- ggplot(churn.dt, aes(Age, fill = IsActiveMember))
j1 + geom_bar(position = "stack") + labs(title = "Relationship between Active Status and Age") # More active members from age group 30 to 40 years old

# Relationship between CreditCard and Age
j2 <- ggplot(churn.dt, aes(Age, fill = HasCrCard))
j2 + geom_bar(position = "stack") + labs(title = "Relationship between Credit Card and Age") # More customers with credit card from age group 30 to 40 years old

##---- Characteristics of high value customers
highcust.dt <- churn.dt[CustomerValue == "High"]

# Scatter plot that can show the difference between high value and low value age and salary
d1 <- ggplot(highcust.dt, aes(Age, EstimatedSalary, color = CustomerValue))
d1 + geom_point() + scale_color_brewer(palette =  "Accent") + labs(title="Scatterplot of Estimated Salary against Age", subtitle = "Only high value customers") # More 'high' valued customers from age group 30 to 40 years old

# Distribution of geography across age by frequency count
high1 <- ggplot(highcust.dt, aes(Age, fill = Geography))
high1 + geom_histogram(bins = 8, position = "dodge") + xlim(10, 90) + scale_fill_brewer(palette = "Set2") +  labs(title = 'Age distribution of high valued customers', subtitle = "segregated by geography ")

# Distribution of geography across age by proportion
high1 + geom_histogram(bins = 5, position = "fill") + scale_fill_brewer(palette = "Set2") + labs(title = 'Age distribution of high valued customers', subtitle = "segregated by geography ")

# Credit score distribution of high valued customers
high3 <- ggplot(highcust.dt, aes(CreditScore, fill = CustomerValue))
high3 + geom_histogram(bins = 15, color = "black", fill = "#FE874C") + labs(title = 'Credit score distribution of high valued customers')



#-------- Logistic Regression Predictions --------

# Initial Logistic Regression Analysis
levels(churn.dt$CustomerValue) # 'High' is baseline
cust.m1 <- glm(CustomerValue ~ EstimatedSalary+Geography+Gender+Age+CreditScore, family = binomial,data = churn.dt)
summary(cust.m1) # 'Age' is identified to be statistically significant

# Train-Test Split (Ratio of Trainset:Testset is 70:30)
set.seed(2019)
train <- sample.split(Y = churn.dt$CustomerValue, SplitRatio = 0.7)
trainset <- subset(churn.dt, train == T)
testset <- subset(churn.dt, train == F)
summary(trainset$CustomerValue) # Ensure trainset has the same proportion as initial dataset
summary(testset$CustomerValue) # Ensure testset has the same proportion as initial dataset

# Rebalancing of Trainset
majority = trainset[CustomerValue == "Low"] # 'Low' value customers make up the majority
minority = trainset[CustomerValue == "High"] # While 'High' value customers make up minority
chosen = sample(seq(1:nrow(majority)), size = nrow(minority)) # Undersampling of 'Low' value customers
majority.chosen = majority[chosen]
trainset.bal = rbind(majority.chosen, minority) 
summary(trainset.bal) # Equal numbers of 'High' and 'Low' value customers

# Training model on balanced data with 'Age' as the only X variable
levels(trainset.bal$CustomerValue) # 'High' is baseline
train_cust <- glm(CustomerValue ~ Age, family = binomial,data = trainset.bal) 
summary(train_cust) # Z = 0.477209 - 0.012277(Age)

# Odds Ratio and Odds Ratio Confidence Interval
OR <- exp(coef(train_cust))
OR # Odds (CustomerValue = 'High') increase by a factor of 0.98779 with 1 unit increase in 'Age'
OR.CI <- exp(confint(train_cust))
OR.CI # Range does not include 1

# Output the probability from the logistic function for Trainset
predict_cust_train <- predict(train_cust, type="response")  
threshold <- 0.5 # Set the threshold for predicting Y = 'High' based on probability.
train_hat <- ifelse(predict_cust_train > threshold, "High", "Low") # Predict Y = "High" if probability > threshold, else predict Y = "Low".
table(Trainset.Actual = trainset.bal$CustomerValue, train_hat, deparse.level = 2) # Confusion matrix.
mean(train_hat == trainset.bal$CustomerValue) # Trainset Accuracy of 0.49531

# Output the probability from the logistic function for Testset
predict_cust_test <- predict(train_cust, newdata = testset,type="response")
test_hat <- ifelse(predict_cust_test > threshold, "High", "Low")
table(Testset.Actual = testset$CustomerValue, test_hat, deparse.level = 2) 
mean(test_hat == testset$CustomerValue) # Accuracy of 0.48133


#-------- CART Model Predictions --------

# Rebalancing
set.seed(2025)
cart_majority = churn.dt[CustomerValue == "Low"]
cart_minority = churn.dt[CustomerValue == "High"]
cart_chosen = sample(seq(1:nrow(cart_majority)), size = nrow(cart_minority))
cart_majority.chosen = cart_majority[cart_chosen]
cart_trainset.bal = rbind(cart_majority.chosen, cart_minority)
summary(cart_trainset.bal)

# Building CART Model (minimum number of values before splitting set at 20)
set.seed(2025)
cart1 <- rpart(CustomerValue ~ Age+Gender+CreditScore+EstimatedSalary+Geography, data = trainset.bal,
               method = 'class', control = rpart.control(minsplit = 20, cp = 0))

# Plotting the CP graph to visualize the lowest CP point
plotcp(cart1)

# Finding the optimal CP value
CVerror.cap <- cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xerror"] +
  cart1$cptable[which.min(cart1$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4
while (cart1$cptable[i,j] > CVerror.cap) {i <- i + 1}
cp.opt = ifelse(i > 1, sqrt(cart1$cptable[i,1] * cart1$cptable[i-1,1]), 1)
cp.opt ## Optimal CP value of 0.008456296

# Pruning the CART Model using optimal CP
cart2 <- prune(cart1, cp = cp.opt)
rpart.plot(cart2, nn= T, main = "Pruned Tree")

# Output the variable importance of the CART model
cart2$variable.importance ## Age, CreditScore, EstimatedSalary identified as important

# Model Predictions
Predicted_CustVal <- predict(cart2, type="class")
summary(Predicted_CustVal)

# Confusion Matrix
table(Actual_CustType = cart_trainset.bal$CustomerValue, Predicted_CustVal, deparse.level = 2)

# Accuracy of CART Model
mean(Predicted_CustVal == trainset.bal$CustomerValue) ## Accuracy of 0.5384825, better than Logistic Regression Model's accuracy







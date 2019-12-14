
#Loading the required libraries:
library(MASS)
library(car)
library(e1071)
library(caret)
library(ggplot2)
library(cowplot)
library(caTools)
library(DAAG)
library(tseries)
library(ggplot2)
library(glmnet)


#Loading files:
test_data <- read.csv("base_promo_testset.csv", stringsAsFactors = F)
train_data <- read.csv("base_promo_trainset.csv", stringsAsFactors = F)

#Data Check and Preparation:
#####################################################################################
str(test_data)
#converting target variable from NA to 0 in the test set.
test_data$ordered_units <- 0

str(train_data)

#convert character variable (MAG) to factor:
test_data$MAG <- as.factor(test_data$MAG)
train_data$MAG <- as.factor(train_data$MAG)

#convert integer varaibles to Numeric:
INTEGER_VARIABLES <- lapply(train_data, class) == "integer"
train_data[, INTEGER_VARIABLES] <- lapply(train_data[, INTEGER_VARIABLES], as.numeric)
sapply(train_data,class)

#similiarly for test:
INTEGER_VARIABLES <- lapply(test_data, class) == "integer"
test_data[, INTEGER_VARIABLES] <- lapply(test_data[, INTEGER_VARIABLES], as.numeric)
sapply(test_data,class)

#Dropping account_id as its same across all datapoints

train_data<- train_data[,-c(1)]
test_data<- test_data[,-c(1)]

#Missing values:
sapply(train_data, function(x) sum(is.na(x)))
sapply(test_data, function(x) sum(is.na(x)))


#Exploratory Data Analysis:
#####################################################################################

box_theme<- theme(axis.line=element_blank(),axis.title=element_blank(), 
                  axis.ticks=element_blank(), axis.text=element_blank())

box_theme_y<- theme(axis.line.y=element_blank(),axis.title.y=element_blank(), 
                    axis.ticks.y=element_blank(), axis.text.y=element_blank(),
                    legend.position="none")

#### Distribution of Ordered_Units in Numerical varaibles ###########################

plot_grid(ggplot(train_data, aes(x=ordered_units,y=product_id, fill=ordered_units))+ geom_point()+ 
            coord_flip() +theme(legend.position="none"),
          ggplot(train_data, aes(x=ordered_units,y=AG, fill=ordered_units))+ geom_point()+
            coord_flip() + box_theme_y,
          ggplot(train_data, aes(x=ordered_units,y=promo_flag, fill=ordered_units))+ geom_point()+
            coord_flip() + box_theme_y,
          align = "v",nrow = 1)

plot_grid(ggplot(train_data, aes(x=ordered_units,y=date_id, fill=ordered_units))+ geom_point()+ 
            coord_flip() +theme(legend.position="none"),
          ggplot(train_data, aes(x=ordered_units,y=promo_discount_perc, fill=ordered_units))+ geom_point()+
            coord_flip() + box_theme_y,
          ggplot(train_data, aes(x=ordered_units,y=base_demand, fill=ordered_units))+ geom_point()+
            coord_flip() + box_theme_y,
          align = "v",nrow = 1)

#### Distribution of Ordered_Units in Factor varaible (MAG) ###########################

bar_theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5), 
                   legend.position = "left")

plot_grid(ggplot(train_data, aes(x=MAG,fill=ordered_units))+ geom_bar()+ labs(y="ordered_units")+bar_theme1, align = "h")
#EO1 and E15 with the highest number of ordered units.


#Feature Standardisation:
#######################################################################################

#Creating Dummy variable for MAG in both test and train:

#for train:
dummy_1 <- data.frame(model.matrix( ~MAG, data = train_data))
dummy_1<-dummy_1[,-1]
train_data_final <- cbind(train_data[ , c(1,3,4,5,6,7,8)], dummy_1)

#for test:
dummy_2 <- data.frame(model.matrix( ~MAG, data = test_data))
dummy_2<-dummy_2[,-1]
test_data_final <- cbind(test_data[ , c(1,3,4,5,6,7,8)], dummy_2)

set.seed(100)

#Modelling:
#######################################################################################

#APPROACH:

#Since the Predicted variable(ordered_units) is numeric and there are more than one 
#independent numeric variables with random error and some showing linear dependence like base_demand,
#I'll try fitting a linear model, cross validate it also if required will try to regularize using 
#ridege and accodingly finalise the model.

# Developing the first model:

model_1 <-lm(ordered_units~.,data=train_data_final)
summary(model_1)
sort(vif(model_1))

#-------------------------------------------------------------------------------------------

# Applying the stepwise approach:

step <- stepAIC(model_1, direction="both")

#-------------------------------------------------------------------------------------------
# Run the step object
step

#-------------------------------------------------------------------------------------------
#creating the model returned by stepAIC and accordingly checking for multicollinearity amongs predictors:

model_2 <- lm(formula = ordered_units ~ product_id + date_id + promo_flag + 
     promo_discount_perc + base_demand + MAGE15 + MAGI36 + MAGM41 + 
     MAGW91, data = train_data_final)

summary(model_2)
sort(vif(model_2))

#removing promo_flag as its insignificant and has a high VIF and rebuilding the model:
#-------------------------------------------------------------------------------------------
model_3 <- lm(formula = ordered_units ~ product_id + date_id  + 
                promo_discount_perc + base_demand + MAGE15 + MAGI36 + MAGM41 + 
                MAGW91, data = train_data_final)

summary(model_3)
sort(vif(model_3))
#-------------------------------------------------------------------------------------------

#Further removing the product_id as its insignificant and intutively not very helpful:
model_4 <- lm(formula = ordered_units ~  date_id  + 
                promo_discount_perc + base_demand + MAGE15 + MAGI36 + MAGM41 + 
                MAGW91, data = train_data_final)

summary(model_4)
sort(vif(model_4))

#-------------------------------------------------------------------------------------------

#Further removing the MAGE15 as its insignificant :
model_5 <- lm(formula = ordered_units ~  date_id  + 
                promo_discount_perc + base_demand  + MAGI36 + MAGM41 + 
                MAGW91, data = train_data_final)

summary(model_5)
sort(vif(model_5))

#Adjusted R-squared:  0.6893 
#not much change in the adj-R^2 across all models(1-5) but
#model_5 has all significant predictors and no multicolinearity amongs predictors.

#cross validating on train sample, as no actual are provided for test:
#here giving the same formula as given in model_5:
temp_crossval <- cv.lm(data = train_data_final, form.lm = formula(ordered_units ~  date_id + promo_discount_perc + base_demand  + MAGI36 + MAGM41 + MAGW91),m = 10)

#CV results:
temp_crossval

#######################################################################################
#Trying the regularized model RIDGE-REGRESSION:-----------------------------------------------------------------------------------------

data_new <-read.csv("base_promo_trainset.csv")

# Creating a matrix "x" of all independent variables
# and storing dependent variable in "y".

x <- model.matrix(ordered_units~.,data=data_new)[,-1]
y <-data_new$ordered_units

# Divide you data in 70:30 

set.seed(1)
train= sample(1:nrow(x), 0.7*nrow(x))

# Store indices into test which is not present in train . 
test = (-train)
y.test = y[test]

# Cross Validation

cv.out <- cv.glmnet(x[train,],y[train],alpha=0)

# Plot cv.out

plot(cv.out)
#the plots tells the log(lambda) interval for minimum means squared error 
#by dotted line, also above it shows the number of features used, as its 11 no regularization is carried out.


# Optimal lamda 

minlamda <- cv.out$lambda.min

minlamda
#[1] 73.1

# Apply model on train dataset at lambda equal to minlamda
ridge.mod <- glmnet(x[train,],y[train],alpha=0,lambda =minlamda)

# Prediction on test dataset
ridge.pred <- predict(ridge.mod,s=minlamda,newx=x[test,])

# MSE with ridge 
mean((ridge.pred-y.test)^2)
#268075

#MSE without ridge = MSE of linear model when we did cross validation:
#262027

#since not much change in MSE therefor we choose linear model without regularization to predict:
final_model <- model_5

#######################################################################################
# OUTPUT:
#-------------------------------------------------------------------------------------------

# Now we have 6 variables in the model.
# Predicting the model on test dataset.

Predict_1 <- predict(final_model,test_data_final[,-c(7)])

#-------------------------------------------- FIN -----------------------------------------------
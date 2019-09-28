#this file contains all of the code, all of the functions I've written and used on this assignment 
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from numpy.linalg import inv
import sklearn.utils
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score

#read in csv file containing mpg data to dataframe mpg_df
mpg_df = pd.read_csv("auto-mpg.csv")
print(mpg_df.shape)
print(mpg_df)
print(mpg_df.describe())

#sort mpg values in order to find mpg thresholds
mpg_df = pd.read_csv('auto-mpg.csv').sort_values(['mpg'])
mpg_df.plot(kind='scatter', x='mpg', y='horsepower')



#split mpg_df into 3 parts, and then take averages 
#from the mins and maxes of each part to determine thresholds
mpg_dfs = np.split(mpg_df, [130, 261], axis=0)

max1 = max(mpg_dfs[0]['mpg'])
max2 = max(mpg_dfs[1]['mpg'])
min1 = min(mpg_dfs[1]['mpg'])
min2 = min(mpg_dfs[2]['mpg'])

thresh1 = (max1 + min1) / 2
thresh2 = (max2 + min2) / 2
thresh1
thresh2
#these thresholds will be used to classify each mpg value

#after thresholds have been found, we can return to original, unsorted mpg_df
mpg_df = pd.read_csv("auto-mpg.csv")

#scatter matrix of mpg_df before classification:
scatter_matrix(mpg_df, alpha=0.9, figsize=(20, 20))
plt.show()


#create a list and fill it with the class number of each mpg value using calculated thresholds
mpg_list = [0] * 392   
mpg_df_len = len(mpg_df)
i = 0
while i < mpg_df_len:
    if mpg_df['mpg'][i] <= thresh1:
        mpg_list[i] = 1
    elif mpg_df['mpg'][i] >= thresh2:
        mpg_list[i] = 3
    else:
        mpg_list[i] = 2
    i = i + 1

#take list of mpg classes, convert into series, then use series to
#create new column in mpg_df called 'class'
mpg_df['class'] = pd.Series(mpg_list)




#create list called colors and fill it with the color corresponding with
#the number of each class. 1(low mpg) = red, 2(medium mpg) = yellow, 3(high mpg) = green
print ('scatter matrix with color-coded mpg classes')
colors = [0] * mpg_df_len

i = 0
while i < mpg_df_len:
    if mpg_df['class'][i] == 1:
        colors[i] = 'red'
    elif mpg_df['class'][i] == 2:
        colors[i] = 'yellow'
    else:
        colors[i] = 'green'
    i = i + 1
    

#use colors list to plot scatter matrix with samples divided into classes    
pd.plotting.scatter_matrix(mpg_df, alpha=0.9, figsize=(20, 20), color=colors)
    


#this function returns the correlation between feature x and feature y
def correlation(df, x, y):
    corr = df[x].corr(df[y])
    
    return corr
    


#split shuffles and then divides the dataset into a training and testing set

def split(df):
    df = shuffle(mpg_df)
    df = df.reset_index(drop=True)   #reset the indices of the dataset after shuffling
    df = np.split(df, [200], axis=0)
    
    return df

#create the training and testing datasets
mpg_dfs = split(mpg_df)

mpg_train = mpg_dfs[0]
mpg_test = mpg_dfs[1]
mpg_test = mpg_test.reset_index(drop=True)


def Arrays(df, n, col): #turns each feature column from the dataframe into an array to be used later
    
    list = [0] * n  
    i = 0
    while i < n:
        list[i] = df[col][i]
        i = i + 1        
    
    array = np.array([list])
    array = array.astype(float)
    
    return array
    
 



#calculates OLS and returns w, array of weights 
def OLS(x, xt, y):
    w = xt @ x
    w = inv(w)
    w = w @ xt
    w = w @ y
        
    return w


#performs linear regression on given dataset
#parameters are polynomial order(0,1,2, or 3), training dataset x (one of 7 features), 
#testing dataset xtest (only used when calculating test mse, otherwise np.zeros() goes here), 
#the training dataset y (mpg), size of training dataset n, and size of testing dataset ntest.    
def LinearReg(order, x, xtest, y, n, ntest): 
 
    
    ones = [1] * n
    ones = np.array([ones]) #creates array of ones to be included in X matrix
    ones = ones.astype(float) 
    
    y = y.transpose()

    
    #different orders require slightly different algorithms for calculating w (weights)
    if order == 0:
        X = ones
        X = X.transpose()
        XT = X.transpose()
        w = OLS(X, XT, y) #call OLS function
    
    elif order == 1:
        X = np.append(ones, x, axis = 0)  #combine ones and x arrays to prpoduce X matrix
        X = X.transpose()                 #transpose X to get M x 2 matrix
        XT = X.transpose()
        w = OLS(X, XT, y)
        
    elif order == 2:
        xsqr = np.square(x)               #for second order we add on a column containing the square of x
        X = np.append(ones, x, axis = 0)
        X = np.append(X, xsqr, axis = 0)
        X = X.transpose()
        XT = X.transpose()
        w = OLS(X, XT, y)
        
    elif order == 3:
        xcube = np.power(x, 3)            #for third order we add yet another column containing the cube of x
        xsqr = np.square(x)
        X = np.append(ones, x, axis = 0)
        X = np.append(X, xsqr, axis = 0)
        X = np.append(X, xcube, axis = 0)
        X = X.transpose()
        XT = X.transpose()
        w = OLS(X, XT, y)
        
    else:
        return print('error: order must be between 0 and 3')
        
        
    #calculate y-hat, the predicted equation for the line of best fit
    
    yhat = X @ w
    
    #unless we're calculating the test mse, xtest is just an array of zeros and the function ends here and returns yhat
    if xtest.all() == 0:
        
        return yhat
    
    
    #if an array containing test data is entered as a parameter, we go through the same steps
    #as before, excluding calculating w.
    else:
        
        ones = [1] * ntest
        ones = np.array([ones]) #creates array of ones to be included in X matrix
        ones = ones.astype(float)
        
        if order == 0:
            X = ones
            X = X.transpose()
        
        
    
        elif order == 1:
            X = np.append(ones, xtest, axis = 0)  #combine ones and x arrays to prpoduce X matrix
            X = X.transpose()  #transpose X to get M x 2 matrix
        
        
        elif order == 2:
            xsqr = np.square(xtest)
            X = np.append(ones, xtest, axis = 0)
            X = np.append(X, xsqr, axis = 0)
            X = X.transpose()
        
        
        elif order == 3:
            xcube = np.power(xtest, 3)
            xsqr = np.square(xtest)
            X = np.append(ones, xtest, axis = 0)
            X = np.append(X, xsqr, axis = 0)
            X = np.append(X, xcube, axis = 0)
            X = X.transpose()
        
        else:
            return print('error: order must be between 0 and 3')
   
        yhat_test = X @ w
    
        return yhat_test
        

#creates array of zeros to be used in LinearReg()and MultLinearRef().  Number of zeros entered is arbitrary 
zeros = np.zeros(5)

#function call will return the set of predicted values based on training data and calculated training weights
LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), zeros, Arrays(mpg_train, 200, 'mpg'), 200, 192)
print(LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), zeros, Arrays(mpg_train, 200, 'mpg'), 200, 192))

#function call will return the set of predicted values based on the test data and calculated training weights
LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_train, 200, 'mpg'), 200, 192)
print(LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_train, 200, 'mpg'), 200, 192))


#plots scatterplot with 0th, 1st, 2nd, and 3rd order lines
def PlotLinReg(x, y, yhat0, yhat1, yhat2, yhat3, xlabel, ylabel):
    x = x.transpose()
    y = y.transpose()
    plt.plot(x, yhat0, color='g')
    plt.plot(x, yhat1, color='c')
    plt.plot(x, yhat2, 'o', color='yellow')
    plt.plot(x, yhat3, 'o', color='red')
    plt.scatter(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    
print("\nplot for (horsepower, mpg)")
PlotLinReg(Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'mpg'), LinearReg(0, Arrays(mpg_train, 200, 'horsepower'), zeros, Arrays(mpg_train, 200, 'mpg'), 200, 192), LinearReg(1, Arrays(mpg_train, 200, 'horsepower'), zeros, Arrays(mpg_train, 200, 'mpg'), 200, 192), LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), zeros,  Arrays(mpg_train, 200, 'mpg'), 200, 192), LinearReg(3, Arrays(mpg_train, 200, 'horsepower'), zeros,  Arrays(mpg_train, 200, 'mpg'), 200, 192), 'horsepower', 'mpg')    
    
    

def MultLinearReg(order, x1, x2, x3, x4, x5, x6, x7, x1test, x2test, x3test, x4test, x5test, x6test, x7test, y, n, ntest):
    
    ones = [1] * n
    ones = np.array([ones]) #creates array of ones to be included in X matrix
    ones = ones.astype(float) 
    
    #x = x.astype(float)       #converts arrays to float types so we can perform matrix operations
           #
    y = y.transpose()
    #y = y.astype(float)
    
    
    if order == 0:
        X = ones
        X = X.transpose()
        XT = X.transpose()
        w = OLS(X, XT, y)
    
    elif order == 1:
        X = np.append(ones, x1, axis = 0)  #combine ones and x arrays to prpoduce X matrix
        X = np.append(X, x2, axis = 0)
        X = np.append(X, x3, axis = 0)
        X = np.append(X, x4, axis = 0)
        X = np.append(X, x5, axis = 0)
        X = np.append(X, x6, axis = 0)
        X = np.append(X, x7, axis = 0)
        X = X.transpose()  #transpose X to get M x 2 matrix
        XT = X.transpose()
        w = OLS(X, XT, y)
        
    elif order == 2:
        x1sqr = np.square(x1)
        x2sqr = np.square(x2)
        x3sqr = np.square(x3)
        x4sqr = np.square(x4)
        x5sqr = np.square(x5)
        x6sqr = np.square(x6)
        x7sqr = np.square(x7)        
                    
        X = np.append(ones, x1, axis = 0)
        X = np.append(X, x2, axis = 0)
        X = np.append(X, x3, axis = 0)
        X = np.append(X, x4, axis = 0)
        X = np.append(X, x5, axis = 0)
        X = np.append(X, x6, axis = 0)
        X = np.append(X, x7, axis = 0)
        X = np.append(X, x1sqr, axis = 0)
        X = np.append(X, x2sqr, axis = 0)
        X = np.append(X, x3sqr, axis = 0)
        X = np.append(X, x4sqr, axis = 0)
        X = np.append(X, x5sqr, axis = 0)
        X = np.append(X, x6sqr, axis = 0)
        X = np.append(X, x7sqr, axis = 0)
        
        X = X.transpose()
        XT = X.transpose()
        w = OLS(X, XT, y)
        

        
    else:
        return print('error: order must be between 0 and 2')
        
        
    #calculate y-hat, the predicted equation for the line of best fit
    
    yhat = X @ w
        
    if x1test.all() == 0:    
        return yhat   

    else:
        
        ones = [1] * ntest
        ones = np.array([ones]) #creates array of ones to be included in X matrix
        ones = ones.astype(float)
        
        if order == 0:
            X = ones
            X = X.transpose()
            
    
        elif order == 1:
            X = np.append(ones, x1test, axis = 0)  #combine ones and x arrays to prpoduce X matrix
            X = np.append(X, x2test, axis = 0)
            X = np.append(X, x3test, axis = 0)
            X = np.append(X, x4test, axis = 0)
            X = np.append(X, x5test, axis = 0)
            X = np.append(X, x6test, axis = 0)
            X = np.append(X, x7test, axis = 0)
            X = X.transpose()  #transpose X to get M x 2 matrix
        
        
        elif order == 2:
            x1sqr = np.square(x1test)
            x2sqr = np.square(x2test)
            x3sqr = np.square(x3test)
            x4sqr = np.square(x4test)
            x5sqr = np.square(x5test)
            x6sqr = np.square(x6test)
            x7sqr = np.square(x7test)        
                    
            X = np.append(ones, x1test, axis = 0)
            X = np.append(X, x2test, axis = 0)
            X = np.append(X, x3test, axis = 0)
            X = np.append(X, x4test, axis = 0)
            X = np.append(X, x5test, axis = 0)
            X = np.append(X, x6test, axis = 0)
            X = np.append(X, x7test, axis = 0)
            X = np.append(X, x1sqr, axis = 0)
            X = np.append(X, x2sqr, axis = 0)
            X = np.append(X, x3sqr, axis = 0)
            X = np.append(X, x4sqr, axis = 0)
            X = np.append(X, x5sqr, axis = 0)
            X = np.append(X, x6sqr, axis = 0)
            X = np.append(X, x7sqr, axis = 0)
            X = X.transpose()
     
       
        else:
            return print('error: order must be between 0 and 2')
        
    yhat_test = X @ w
    
    return yhat_test
        

#function call will return the set of predicted values based on training data and calculated training weights
MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), zeros, Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192)    
print(MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), zeros, Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192))
#function call will return the set of predicted values based on testing data and calculated training weights
MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192)    
print(MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192))

        

#predicted values for calculating training MSE of Univariate Linear Regression. 'horsepower' can be replaced with any of the other 7 features
predict_trainval = LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), zeros, Arrays(mpg_train, 200, 'mpg'), 200, 192)

#predicted values for calculating testing MSE of Univariate Linear Regression.  'horsepower' can be replaced with any of the other 7 features
predict_testval = LinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_train, 200, 'mpg'), 200, 192)

#predicted values for calculating training MSE of the Multivariate Linear Regression
predict_trainval = MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), zeros, Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192)

#predicted values for calculating testing MSE of the Multivariate Linear Regression
predict_testval = MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), Arrays(mpg_test, 192, 'horsepower'), Arrays(mpg_test, 192, 'acceleration'), Arrays(mpg_test, 192, 'displacement'),Arrays(mpg_test, 192, 'cylinders'),Arrays(mpg_test, 192, 'weight'),Arrays(mpg_test, 192, 'model year'), Arrays(mpg_test, 192, 'origin'), Arrays(mpg_train, 200, 'mpg'), 200, 192)  

#true mpg values from training data
true_trainval = Arrays(mpg_train, 200, 'mpg')
true_trainval = true_trainval.transpose()

#true mpg values from test data
true_testval = Arrays(mpg_test, 192, 'mpg')
true_testval = true_testval.transpose()

#calculates mean square error
def MSE(target_mpg, predicted_mpg, n):
    mse = 0
#for loop performs summation of (true_values - predicted_values) ^ 2
    for i in range(n):              
        mse = mse + np.square(target_mpg[i] - predicted_mpg[i])
        
    mse = mse / n      #divide by number of samples
    return mse.item(0) #the mse value is stored in an array so we use .item to extract it

#calculate MSE
MSE(true_trainval, predict_trainval, 200)
MSE(true_testval, predict_testval, 192)
print("\nmean squared errors:")
print(MSE(true_trainval, predict_trainval, 200))
print(MSE(true_testval, predict_testval, 192))


#Logistic Regression

logreg = LogisticRegression()

#for Logistic Regression, the 'car name' feature needs to be dropped from the dataset
mpg_train = mpg_train.drop(['car name'], axis=1)
mpg_test = mpg_test.drop(['car name'], axis=1)

#create variables for each feature array for convenience
horsepower = Arrays(mpg_train, 200, 'horsepower')
horsepower = horsepower.transpose()

acceleration = Arrays(mpg_train, 200, 'acceleration')
acceleration = acceleration.transpose()

displacement = Arrays(mpg_train, 200, 'displacement')
displacement = displacement.transpose()

weight = Arrays(mpg_train, 200, 'weight')
weight = weight.transpose()

cylinders = Arrays(mpg_train, 200, 'cylinders')
cylinders = cylinders.transpose()

modelyear = Arrays(mpg_train, 200, 'model year')
modelyear = modelyear.transpose()

origin = Arrays(mpg_train, 200, 'origin')
origin = origin.transpose()

mpg = Arrays(mpg_train, 200, 'mpg')
mpg = mpg.transpose()

mpg_class = Arrays(mpg_train, 200, 'class')
mpg_class = mpg_class.transpose()
mpg_class = mpg_class.astype(int) #classes are discrete, so mpg_classtrain is changed to type: int

test_mpg_class = Arrays(mpg_test, 192, 'class')
test_mpg_class = test_mpg_class.transpose()
test_mpg_class = test_mpg_class.astype(int)

test_horsepower = Arrays(mpg_test, 192, 'horsepower')
test_horsepower = test_horsepower.transpose()

test_acceleration = Arrays(mpg_test, 192, 'acceleration')
test_acceleration = test_acceleration.transpose()

test_displacement = Arrays(mpg_test, 192, 'displacement')
test_displacement = test_displacement.transpose()

test_weight = Arrays(mpg_test, 192, 'weight')
test_weight = test_weight.transpose()

test_cylinders = Arrays(mpg_test, 192, 'cylinders')
test_cylinders = test_cylinders.transpose()

test_modelyear = Arrays(mpg_test, 192, 'model year')
test_modelyear = test_modelyear.transpose()

test_origin = Arrays(mpg_test, 192, 'origin')
test_origin = test_origin.transpose()

#fit the data using .fit method.  horsepower can be replaced with any of the other 7 features
logreg.fit(horsepower, mpg_class)

#horsepower can be replaces with any of the other 7 test or training features.  (Example: horsepower or test_horsepower, weight or test_weight)
predict_class = logreg.predict(horsepower)
predict_class = predict_class.transpose()

#reshape before use in precision_score
mpg_class = np.reshape(mpg_class, (200,))
test_mpg_class = np.reshape(test_mpg_class, (192,))

#calculates precision.  If predict_class contains a training set use mpg_class, if it contains a testing set, replace with test_mpg_class
precision_score(predict_class, mpg_class, average='macro')
print("\nprecision score:")
print(precision_score(predict_class, mpg_class, average='macro'))



#Problem 7
#We can create a series of arrays that hold the values of each feature of our new car:

new_hp = [180]
new_hp = np.array([new_hp])

new_cyl = [6]
new_cyl = np.array([new_cyl])

new_accel = [9]
new_accel = np.array([new_accel])

new_weight = [3700]
new_weight = np.array([new_weight])

new_dis = [350]
new_dis = np.array([new_dis])

new_origin = [1]
new_origin = np.array([new_origin])

new_year = [80]
new_year = np.array([new_year])

#and now call the MultLinearReg() function to get the predicted mpg:
MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), new_hp, new_accel, new_dis, new_cyl, new_weight, new_year, new_origin, Arrays(mpg_train, 200, 'mpg'), 200, 1)

print("\nIf a USA manufacturer (origin 1) had considered to introduce a model in 1980 with the fol-lowing characteristics: 6 cylinders, 350 cc displacement, 180 horsepower, 3700 lb weight,9m/sec2acceleration, we should expect an mpg rating of:")
print(MultLinearReg(2, Arrays(mpg_train, 200, 'horsepower'), Arrays(mpg_train, 200, 'acceleration'), Arrays(mpg_train, 200, 'displacement'),Arrays(mpg_train, 200, 'cylinders'),Arrays(mpg_train, 200, 'weight'),Arrays(mpg_train, 200, 'model year'), Arrays(mpg_train, 200, 'origin'), new_hp, new_accel, new_dis, new_cyl, new_weight, new_year, new_origin, Arrays(mpg_train, 200, 'mpg'), 200, 1).item(0))



#Logistic Regression on each of the features of the new car sample
print("\npredicted mpg classes of each feature of new car with: 6 cylinders, 350 cc displacement, 180 horsepower, 3700 lb weight, and 9m/sec2 acceleration")
logreg.fit(acceleration, mpg_class)
predict_class = logreg.predict(new_accel)
print(predict_class)
logreg.fit(weight, mpg_class)
predict_class = logreg.predict(new_weight)
print(predict_class)
logreg.fit(horsepower, mpg_class)
predict_class = logreg.predict(new_hp)
print(predict_class)
logreg.fit(displacement, mpg_class)
predict_class = logreg.predict(new_dis)
print(predict_class)
logreg.fit(origin, mpg_class)
predict_class = logreg.predict(new_origin)
print(predict_class)
logreg.fit(modelyear, mpg_class)
predict_class = logreg.predict(new_year)
print(predict_class)
logreg.fit(cylinders, mpg_class)
predict_class = logreg.predict(new_cyl)
print(predict_class)





















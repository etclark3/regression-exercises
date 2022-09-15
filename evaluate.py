import numpy as np
import pandas as pd        
        
import matplotlib.pyplot as plt
import seaborn as sns
        
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression




#- plot_residuals(y, yhat): creates a residual plot
#- regression_errors(y, yhat): returns the following values:
#    - sum of squared errors (SSE)
#    - explained sum of squares (ESS)
#    - total sum of squares (TSS)
#    - mean squared error (MSE)
#    - root mean squared error (RMSE)
#- baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
#- better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false

def plot_residuals(y, yhat):
    return y
    
def regression_errors(y, yhat):
    SSE = df.residual_sq.sum()
    MSE = SSE/len(df)
    RMSE = MSE**0.5
    MSE2 = mean_squared_error(df.tax_value, df.yhat)
    SSE2 = MSE2 * len(df)
    RMSE2 = MSE2**0.5
    ESS = sum((df.yhat - df.tax_value.mean())**2)
    TSS = ESS + SSE
    R2 = r2_score(df.y, yhat)
    
    return SSE
    
def baseline_mean_errors(y):
    SSE_baseline = train.baseline_residual_sq.sum()
    MSE_baseline = SSE_baseline/len(train)
    RMSE_baseline = MSE_baseline**0.5
    MSE2_baseline = mean_squared_error(train.tax_value, train.baseline)
    SSE2_baseline = MSE2_baseline * len(train)
    RMSE2_baseline = MSE2_baseline**0.5
    ESS_baseline = sum((train.baseline - train.tax_value.mean())**2)
    TSS_baseline = ESS_baseline + SSE_baseline
    R2_baseline = ESS_baseline/TSS_baseline

    return SSE_baseline

#def better_than_baseline(y, yhat):
#    if:
#        
#        print('The Baseline is better than the Model')
#    else:
#        print('The Model beats the Baseline')
        


def plot_residuals(y, yhat):
    residuals = yhat - y
    plt.figure(figsize=(12,8))
    sns.scatterplot(y, residuals)
    plt.show()

def regression_errors(y, yhat):
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = mean_squared_error(y, yhat, squared=False)
    ESS = sum((yhat - y.mean())**2)
    TSS = ESS + SSE
    R2 = ESS/TSS

    return SSE, ESS, TSS, MSE, RMSE, R2

def baseline_mean_errors(y):
    baseline = pd.Series(y.mean(), index=np.arange(len(y)))
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = mean_squared_error(y,baseline, squared=False)

    return MSE, SSE, RMSE

def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE, R2 = regression_errors(y, yhat)

    MSE_baseline, SSE_baseline, RMSE_baseline = baseline_mean_errors(y)

    SSE_baseline = MSE_baseline * len(y)

    if SSE < SSE_baseline:
        return True
        
    else:
        return False
# -------------------------------------------------------------------------    
'''Best Predictors'''


'''Determines the best predictors of your target and returns the column names of the best predictors and a sample dataframe'''
def select_kbest(X_train, y_train, k):
    # create the model
    kbest = SelectKBest(f_regression, k=k)
    # Fit the model
    kbest.fit(X_train, y_train)
    # df of the top predictors
    X_train_transformed = pd.DataFrame(kbest.transform(X_train),
                                       columns=X_train.columns[kbest.get_support()],
                                       index=X_train.index)
    
    return X_train.columns[kbest.get_support()],X_train_transformed.head(3)

def rfe(X_train, y_train, k):
    model = LinearRegression()
    # Make the model
    rfe = RFE(model, n_features_to_select=k)
    # Fit the model
    rfe.fit(X_train, y_train)
    # df of the top predictors
    X_train_transformed = pd.DataFrame(rfe.transform(X_train), 
                                       index= X_train.index, 
                                       columns=X_train.columns[rfe.support_])
    
    return X_train.columns[rfe.support_], X_train_transformed.head(3)


'''
def split_evaluate_train(df, model, y, k):
    train_test_split(df)
    for col in df[col]:
        if df[col].dtype == 'object':
            pd.get_dummies(data=df, columns={col})
    
    X_train, y_train = train[['']], train.y
    X_validate, y_validate = validate[['']], validate.y
    X_test, y_test = test[['']], test.y
    
    if model in SelectKBest():
        kbest.fit(X_train, y_train)
    return 

'''

def what_type(col):
    if col.dtype == 'float64':
        print('float64')
    elif col.dtype == 'object':
        print('object')
    elif col.dtype == 'int64':
        print('int64')

# Functionality
import pandas as pd
import numpy as np

# Provides functions for interacting with the operating system
import os

# For Zillow data null values
import random

# statistical modeling
import scipy.stats as stats

# To acquire MYSQL Data
import acquire
from env import username, password, host

# For data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# For modeling
import sklearn.metrics as mtc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Logisticssion

# --------------------------------------------------
'''Establishing a overarching SQL server contact method'''

# Function to pull data from SQL
def get_db_url(username, hostname, password, database):
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

# Function used to split data when modeling
def split(df):
    train, test = train_test_split(df, test_size=.2, random_state=248)
    train, validate = train_test_split(train, test_size=.25, random_state=248)
    print(f'df shape: {df.shape}')
    print(f'Train shape: {train.shape}')
    print(f'Validate shape: {validate.shape}')
    print(f'Test shape: {test.shape}')
    return train, validate, test

# --------------------------------------------------

'''Zillow data'''

# This function that will be apply to ('bedroomcnt','bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt','yearbuilt','taxamount','fips') on homes that are SFH
def wrangle_zillow(df):
    df = df[(df.propertylandusedesc == 'Single Family Residential') | (df.propertylandusedesc == 'Inferred Single Family Residential')]
    df = df[['bedroomcnt','bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt','yearbuilt','taxamount','fips']]
    df.bedroomcnt.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathroomcnt.fillna(random.randint(1.0, 3.0), inplace = True)
    df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median(), inplace = True)
    df.taxvaluedollarcnt.fillna(df.taxvaluedollarcnt.mode().max(), inplace = True)
    # For yearbuilt I'll use 1958 as it falls in the middle of the mean and mode and they are all fairly close in value
    df.yearbuilt.fillna(df.yearbuilt.median(), inplace = True)
    df.taxamount.fillna(df.taxamount.median(), inplace = True)
    print(df)
    return df

def wrangle_7(df):
    df = df[['bedroomcnt','bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt','yearbuilt','taxamount','fips']]
    df.bedroomcnt.fillna(random.randint(2.0, 5.0), inplace = True)
    df.bathroomcnt.fillna(random.randint(1.0, 3.0), inplace = True)
    df.calculatedfinishedsquarefeet.fillna(df.calculatedfinishedsquarefeet.median(), inplace = True)
    df.taxvaluedollarcnt.fillna(df.taxvaluedollarcnt.mode().max(), inplace = True)
    # For yearbuilt I'll use 1958 as it falls in the middle of the mean and mode and they are all fairly close in value
    df.yearbuilt.fillna(df.yearbuilt.median(), inplace = True)
    df.taxamount.fillna(df.taxamount.median(), inplace = True)
    print(df)
    return df


#def prep_zillow(df):
#    wrangle_zillow(df)
#    df.rename(columns={'bedroomcnt': 'bedrooms', 
#                       'bathroomcnt': 'bathrooms', 
#                       'calculatedfinishedsquarefeet': 'f_sqft', 
#                       'taxvaluedollarcnt': 'tax_value'}
#    return df

def scale_data(train, 
               validate, 
               test, 
               columns=['bedrooms', 'bathrooms', 'tax_value', 'f_sqft', 'taxamount'],
               return_scaler=False):
    '''
    Scales train, validate, test and returns scaled versions of each 
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # Make the scaler
    scaler = MinMaxScaler()
    # Fit it
    scaler.fit(train[columns])
    # Apply the scaler:
    train_scaled[columns] = pd.DataFrame(scaler.transform(train[columns]),
                                                  columns=train[columns].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns] = pd.DataFrame(scaler.transform(validate[columns]),
                                                  columns=validate[columns].columns.values).set_index([validate.index.values])
    
    test_scaled[columns] = pd.DataFrame(scaler.transform(test[columns]),
                                                 columns=test[columns].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

              
# --------------------------------------------------

'''Iris Data'''
              
def prep_iris(iris_db):
    iris_db = iris_db.rename(columns={'species_name':'species'})
    iris_db = iris_db.replace({'species' : { 'setosa' : 1, 'versicolor' : 2, 'virginica' : 3 }})
    iris_db = iris_db.drop(columns={'species_id', 'Unnamed: 0'})
    return iris_db

# --------------------------------------------------

'''Telco Data'''              
# I have two Telco .csv somehow, so I have separate functions to prep each

def prep_telco(telco_db):
    telco_db = telco_db.drop(columns=['contract_type', 'payment_type', 'internet_service_type'])
    
    telco_db = pd.get_dummies(data=telco_db, columns=['streaming_tv','streaming_movies', 'paperless_billing', 
                                                      'churn', 'gender', 'partner', 'dependents', 
                                                      'phone_service', 'online_backup', 'device_protection', 'tech_support', 
                                                      'online_security', 'multiple_lines'], drop_first=True)
    
    telco_db = telco_db.drop(columns=['streaming_tv_No internet service', 'online_backup_No internet service', 
                                           'device_protection_No internet service', 'tech_support_No internet service', 
                                           'online_security_No internet service', 'multiple_lines_No phone service', 
                                           'streaming_movies_No internet service'])
    return telco_db

def prep_t(telco_db):
    telco_db = telco_db.drop(columns=['Unnamed: 0'])
    
    telco_db = pd.get_dummies(data=telco_db, columns=['streaming_tv','streaming_movies', 'paperless_billing', 
                                                      'churn', 'gender', 'partner', 'dependents', 
                                                      'phone_service', 'online_backup', 'device_protection', 'tech_support', 
                                                      'online_security', 'multiple_lines'], drop_first=True)
    
    telco_db = telco_db.drop(columns=['streaming_tv_No internet service', 'online_backup_No internet service', 
                                           'device_protection_No internet service', 'tech_support_No internet service', 
                                           'online_security_No internet service', 'multiple_lines_No phone service', 
                                           'streaming_movies_No internet service'])
    return telco_db

# This function compares churn against all other columns
def telco_vis(train, col):
    plt.title('Relationship of churn and '+col)
    sns.barplot(x=col, y='churn_Yes', data=train)
    sns.barplot(x=col, y='churn_Yes', data=train).axhline(train.churn_Yes.mean())
    plt.show()
    
def telco_analysis(train, col):
    telco_vis(train, col)
    test(train, 'churn_Yes', col)
    
# This function is all encompassing of telco_vis and telco_analysis, 
def telco_test(df):
    for col in df.columns.tolist():
        print(telco_analysis(df, col))
        print('-------')
        print(pd.crosstab(df.churn_Yes, df[col]))
        print('-------')
        print(stats.chi2_contingency(pd.crosstab(df.churn_Yes, df[col])))
        print('-------')
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True))
        
# --------------------------------------------------

# Creates dummy variables for all features that are categorical
def prep_titanic(titanic_db):
    titanic_db = titanic_db.drop(columns=['class', 'embarked', 'Unnamed: 0','deck', 'passenger_id'])
    titanic_db = titanic_db.replace({'embark_town' : { 'Southampton' : 1, 'Cherbourg' : 2, 'Queenstown' : 3 }})
    titanic_db.rename(columns={'sex': 'isMale'}, inplace=True)
    titanic_db.age[titanic_db.age.isnull()] = 28
    titanic_db.isMale[titanic_db.isMale == 'male'] = 1
    titanic_db.isMale[titanic_db.isMale == 'female'] = 0
    titanic_db.embark_town[titanic_db.embark_town.isnull()] = 1.0
    return titanic_db

# --------------------------------------------------

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


# --------------------------------------------------

# To test columns of a df and see if there is a relationship between the two
def test(df, col, col2):
    alpha = 0.05
    H0 = col+ ' and ' +col2+ ' are independent'
    Ha = 'There is a relationship between ' +col2+ ' and '+col
    observed = pd.crosstab(df[col2], df[col])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject that', H0)
        print(Ha)
    else:
        print('We fail to reject that', H0)
        print('There appears to be no relationship between ' +col2+ ' and '+col)
        
# --------------------------------------------------
# print("The bold text is",'\033[1m' + 'Python' + '\033[0m')        
# '\033[1m' + 'TEXT' + '\033[0m'
# --------------------------------------------------

'''Student Grade data'''
def wrangle_grades():
    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df
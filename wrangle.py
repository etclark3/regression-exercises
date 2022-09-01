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
from sklearn.linear_model import LogisticRegression

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


def prep_zillow(df):
    wrangle_zillow(df)
    df.rename(columns={'bedroomcnt': 'bedrooms', 
                       'bathroomcnt': 'bathrooms', 
                       'calculatedfinishedsquarefeet': 'f_sqft', 
                       'taxvaluedollarcnt': 'tax_value'}
    return df

# --------------------------------------------------

def prep_iris(iris_db):
    iris_db = iris_db.rename(columns={'species_name':'species'})
    iris_db = iris_db.replace({'species' : { 'setosa' : 1, 'versicolor' : 2, 'virginica' : 3 }})
    iris_db = iris_db.drop(columns={'species_id', 'Unnamed: 0'})
    return iris_db

# --------------------------------------------------

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


# --------------------------------------------------


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
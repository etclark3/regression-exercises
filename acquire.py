# Standard imports
import pandas as pd
import numpy as np

# Provides functions for interacting with the operating system
import os

# To ping SQL server
from env import username, password, host

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# For stat models
import scipy.stats as stats

# For modeling
import sklearn.metrics as mtc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#---------------------------------
# Function to pull data from SQL
def get_db_url(username, hostname, password, database):
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url
#---------------------------------
iris_query = '''
select *
from measurements
'''

def iris_data():
    return pd.read_sql(iris_query, get_db_url(username, host, password, 'iris_db'))

def grab_iris_data():
    filename1 = "iris.csv"    
    if os.path.isfile(filename1):
        return pd.read_csv(filename1)
    else:
        iris = iris_data
        iris.to_csv(filename1)
        return iris
#---------------------------------
telco_query = '''
select *
from customers
'''

def telco_data():
    return pd.read_sql(telco_query, get_db_url(username, host, password, 'telco_churn'))

def grab_telco_data():
    filename2 = "telco_churn.csv"
    if os.path.isfile(filename2):
        return pd.read_csv(filename2)    
    else:
        telco = telco_data()
        telco.to_csv(filename2)
        return telco

def grab_telco():
    filename2 = "telco.csv"
    
    if os.path.isfile(filename2):
        return pd.read_csv(filename2)
    
    else:
        telco = telco_data()
        telco.to_csv(filename2)
        return telco
#---------------------------------
titanic_query = '''
select *
from passengers
'''

def titanic_db():
    return pd.read_sql(titanic_query, get_db_url(username, host, password, 'titanic_db'))

def grab_titanic_data():
    filename3 = "titanic.csv"
    
    if os.path.isfile(filename3):
        return pd.read_csv(filename3)
    
    else:
        titanic = titanic_db
        titanic.to_csv(filename3)

        return titanic
#---------------------------------
zillow2016_query = '''
select *
from properties_2016
'''

def zillow_2016():
    return pd.read_sql(zillow2016_query, get_db_url(username, host, password, 'zillow'))

def get_zillow_2016():
    filename4 = "zillow2016.csv"
    if os.path.isfile(filename4):
        return pd.read_csv(filename4)
    else:
        zillow2016 = zillow_2016()
        zillow2016.to_csv(filename4)

        return zillow2016

# Query all data in properties_2017 from SQL
zillow17_query = '''
select *
from properties_2017
join propertylandusetype
USING (propertylandusetypeid)
'''

# Read the database and bring it to python (
def sfh_2017():
    return pd.read_sql(sfh_query, get_db_url(username, host, password, 'zillow'))

def zillow_2017():
    return pd.read_sql(zillow17_query, get_db_url(username, host, password, 'zillow'))

# Will pull data for properties_2017 joined with propertylandusetype
def get_zillow_2017():
    # Define the file I want to use
    file = "zillow_2017_w_proptypes.csv"
    # If the file exists on my system, pull it
    if os.path.isfile(file):
        return pd.read_csv(file)
    # If it doesn't exist, pull the data from SQL
    else:
        zillow_2017_w_proptypes = zillow_2017()
        zillow_2017_w_proptypes.to_csv(file)

        return zillow_2017_w_proptypes    
#---------------------------------
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

## Read data to pandas dataframe

# From URL
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train_pd = pd.read_csv(train_url)

# Locally
test_url = "/Users/dtemple/PycharmProjects/commondatacases/data/test.csv"
test_pd=pd.read_csv(test_url)

## Explore data using describe, shape and head/tail, info
train_pd.describe()
train_pd.shape
train_pd.head(10)
train_pd.info()

# Use describe(include=['O']) to select just categorical object types (returns count, unique, top, freq)
train_pd.describe(include=['O'])

# value_counts for absolute numbers from results
train_pd["Survived"].value_counts()

# value_counts with normalize for percentages
train_pd["Survived"].value_counts(normalize = True)

# value counts for a variable
x=train_pd["Survived"][train_pd["Sex"]=='male'].value_counts(normalize=True)

# Create the column Child and assign to 'NaN'
train_pd["Child"] = float('NaN')


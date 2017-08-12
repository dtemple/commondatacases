# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

###### Read data to pandas dataframe #######

# From URL
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train_df = pd.read_csv(train_url)

# Locally
test_url = "/Users/dtemple/PycharmProjects/commondatacases/data/test.csv"
test_df=pd.read_csv(test_url)

##### Explore data using describe, shape and head/tail, info #######
train_df.describe()
train_df.shape
train_df.head(10)
train_df.info()

# Use describe(include=['O']) to select just categorical object types (returns count, unique, top, freq)
train_df.describe(include=['O'])

# value_counts for absolute numbers from results
train_df["Survived"].value_counts()

# value_counts with normalize for percentages
train_df["Survived"].value_counts(normalize = True)

# value counts for a variable
x=train_df["Survived"][train_df["Sex"]=='male'].value_counts(normalize=True)

# Create the column Child and assign to 'NaN'
train_df["Child"] = float('NaN')

# Assign it a value (everyone under 18 is a child (1) and everyone over is not (0))
train_df["Child"][train_df["Age"] < 18] = 1
train_df["Child"][train_df["Age"] >= 18] = 0

# Drop useless data
train_df=train_df.drop('Cabin', axis=1)


####### Using Pivot Tables and Group By #########

# Pivot just a subset of the data (note the as_index defaults to true) #
x=train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)
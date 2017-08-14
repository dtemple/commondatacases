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

## Use Group By on just a subset of the data (note the as_index defaults to true) ##
x=train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)

# Group by and then apply a function #
y=train_df[['Sex','SibSp','Parch']].groupby('Sex').sum()

# Heirarchical indexing in a Group by #
z=train_df[['Pclass','Sex','SibSp','Parch']].groupby(['Pclass','Sex']).sum()

## Pivot tables ##
# Pivot table a single value by count/mean/etc #
x=pd.pivot_table(train_df, values=['SibSp'],index=['Sex'], aggfunc='count')

# Pivot a set of columns #
x=pd.pivot_table(train_df[['Pclass','Sex','SibSp','Parch']], index=['Sex'], aggfunc='mean')

# Pivot a set of columns and break it down by the values of one column #
x=pd.pivot_table(train_df[['Pclass','Sex','SibSp','Parch']], index=['Sex'], aggfunc='mean', columns='Pclass')

#           SibSp                         Parch
#Pclass         1         2         3         1         2         3
#Sex
#female  0.553191  0.486842  0.895833  0.457447  0.605263  0.798611
#male    0.311475  0.342593  0.498559  0.278689  0.222222  0.224784


## Performing operations on dfs and pivots ##

# Add a calculated column to a pivot table #
x['sum']=x.sum(axis=1)

# get the % who survived #
x=train_df[['Survived','Embarked']].groupby(['Embarked'], as_index=False).sum() # The number of survivors
y=train_df[['Survived','Embarked']].groupby(['Embarked'], as_index=False).count() # The number of total people
y.rename(columns={'Survived':'total'}, inplace=True)

z=pd.merge(x,y)

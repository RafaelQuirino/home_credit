#==============================================================================
# Imports
#==============================================================================

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns



#==============================================================================
# Read in Data
#==============================================================================

DATAROOT = '/home/rafael/home-credit-default-risk/'



# List files available
print(os.listdir(DATAROOT))
print('\n\n')



# Training data
app_train = pd.read_csv(DATAROOT + '/application_train.csv')
print('Training data shape: ', app_train.shape)
print(app_train.head())
print('\n\n')



# Testing data features
app_test = pd.read_csv(DATAROOT + 'application_test.csv')
print('Testing data shape: ', app_test.shape)
print(app_test.head())
print('\n\n')



#==============================================================================
# Exploratory Data Analysis
#==============================================================================

#----------------------------------------------------------
# Examine the Distribution of the Target Column
#----------------------------------------------------------

print(app_train['TARGET'].value_counts())
print('\n\n')



print(app_train['TARGET'].astype(int).plot.hist())
print('\n\n')
# plt.show()



#----------------------------------------------------------
# Examine Missing Values
#----------------------------------------------------------

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Missing values statistics
missing_values = missing_values_table(app_train)
print(missing_values.head(20))
print('\n\n')



#----------------------------------------------------------
# Column Types
#----------------------------------------------------------

# Number of each type of column
print(app_train.dtypes.value_counts())
print('\n\n')



# Number of unique classes in each object column
print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))
print('\n\n')



#==============================================================================
# Encoding Categorical Variables
#==============================================================================

#----------------------------------------------------------
# Label Encoding and One-Hot Encoding
#----------------------------------------------------------

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
print('\n\n')



# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
print('\n\n')



#----------------------------------------------------------
# Aligning Training and Testing Data
#----------------------------------------------------------

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
print('\n\n')



#==============================================================================
# Back to Exploratory Data Analysis
#==============================================================================

#----------------------------------------------------------
# Anomalies
#----------------------------------------------------------

'''
One problem we always want to be on the lookout for when doing EDA is anomalies within the data. 
These may be due to mis-typed numbers, errors in measuring equipment, or they could be valid but extreme measurements. 
One way to support anomalies quantitatively is by looking at the statistics of a column using the describe method. 
The numbers in the DAYS_BIRTH column are negative because they are recorded relative to the current loan application. 
To see these stats in years, we can mutliple by -1 and divide by the number of days in a year:
'''
print((app_train['DAYS_BIRTH'] / -365).describe())
print('\n\n')



'''
Those ages look reasonable. There are no outliers for the age on either the high or low end. 
How about the days of employment?
'''



print(app_train['DAYS_EMPLOYED'].describe())
print('\n\n')



'''
That doesn't look right! The maximum value (besides being positive) is about 1000 years!
'''



app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employed Histogram');
plt.xlabel('Days Employed');
# plt.show()



'''
Just out of curiousity, let's subset the anomalous clients and see if they tend to have 
higher or low rates of default than the rest of the clients.
'''



anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('Anomalies count: %d' % len(anom))
print('Non-anomalies count: %d' % len(non_anom))
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))
print('\n\n')



'''
Well that is extremely interesting! It turns out that the anomalies have a lower rate of default.

Handling the anomalies depends on the exact situation, with no set rules. 
One of the safest approaches is just to set the anomalies to a missing value and then have them filled in 
(using Imputation) before machine learning. In this case, since all the anomalies have the exact same value, 
we want to fill them in with the same value in case all of these loans share something in common. 
The anomalous values seem to have some importance, so we want to tell the machine learning model 
if we did in fact fill in these values. 
As a solution, we will fill in the anomalous values with not a number (np.nan) and then create a new boolean column 
indicating whether or not the value was anomalous.
'''



# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employed Histogram');
plt.xlabel('Days Employed');
# plt.show()



'''
The distribution looks to be much more in line with what we would expect, and we also have created a new column 
to tell the model that these values were originally anomalous (becuase we will have to fill in the nans with some value, 
probably the median of the column). The other columns with DAYS in the dataframe look to be about what we expect 
with no obvious outliers.

As an extremely important note, anything we do to the training data we also have to do to the testing data. 
Let's make sure to create the new column and fill in the existing column with np.nan in the testing data.
'''



app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
print('\n\n')



#----------------------------------------------------------
# Correlations
#----------------------------------------------------------

'''
Now that we have dealt with the categorical variables and the outliers, let's continue with the EDA. 
One way to try and understand the data is by looking for correlations between the features and the target. 
We can calculate the Pearson correlation coefficient between every variable and the target using the .corr dataframe method.

The correlation coefficient is not the greatest method to represent "relevance" of a feature, 
but it does give us an idea of possible relationships within the data. Some general interpretations of the 
absolute value of the correlation coefficent are:

.00-.19 "very weak"
.20-.39 "weak"
.40-.59 "moderate"
.60-.79 "strong"
.80-1.0 "very strong"
'''

# Find correlations with the target and sort
correlations = app_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n')
print(correlations.tail(15))
print('\n\n')
print('\nMost Negative Correlations:\n')
print(correlations.head(15))
print('\n\n')




'''
Let's take a look at some of more significant correlations: the DAYS_BIRTH is the most positive correlation. 
(except for TARGET because the correlation of a variable with itself is always 1!) 
Looking at the documentation, DAYS_BIRTH is the age in days of the client at the time of the loan in negative days 
(for whatever reason!). The correlation is positive, but the value of this feature is actually negative, meaning that 
as the client gets older, they are less likely to default on their loan (ie the target == 0). That's a little confusing, 
so we will take the absolute value of the feature and then the correlation will be negative.
'''

#----------------------------------------------------------
# Effect of Age on Repayment
#----------------------------------------------------------

# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
print(app_train['DAYS_BIRTH'].corr(app_train['TARGET']))
print('\n\n')

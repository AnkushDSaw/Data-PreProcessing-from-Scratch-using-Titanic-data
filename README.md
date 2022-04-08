# Data-PreProcessing-from-Scratch-using-Titanic-data
Here I provide all the basic code for Data PreProcessing and EDA which can run on spyder / notebook. 
# Import Libaray
import pandas as pd
import numpy as np
# Understand your data 
# Understanding meaning of each column: Data Dictionary: Variable Description
# Survived - Survived (1) or died (0) Pclass - Passenger’s class (1 = 1st, 2 = 2nd, 3 = 3rd) Name - Passenger’s name Sex - Passenger’s sex Age - Passenger’s age SibSp -  Number of siblings/spouses aboard Parch - Number of parents/children aboard (Some children travelled only with a nanny, therefore parch=0 for them.) Ticket - Ticket number Fare - Fare Cabin - Cabin Embarked - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

# Analysing which columns are completely useless in predicting the survival and deleting them Note - Don't just delete the columns because you are not finding it useful. Or focus is not on deleting the columns. Our focus is on analysing how each column is affecting the result or the prediction and in accordance with that deciding whether to keep the column or to delete the column or fill the null values of the column by some values and if yes, then what values.

# Read the data
titanic =pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\7th,8th_April\TASK-13\DATASET\train.csv',header=0,dtype={'Age':np.float64})
titanic.head()
titanic.info()
titanic.describe()

# Name is not play any role for decideing the survival of person, so delete it
del titanic['Name']
titanic.head()

# Ticket is not play any role for decideing the survival of person, so delete it
del titanic['Ticket']
titanic.head()

# Cabin is not play any role for decideing the survival of person, so delete it
del titanic['Cabin']
titanic.head()


# Here Gender is only male and female so we replace it by 0 or 1.

# There are two method for doing this 1. If else  2. LabelEncoder
# ---------------1. method if else
def getGender(str):
    if str=='male':
        return 1
    else:
        return 0
# Adding new column with name Gender which hold 1 for male and 0 for female    
titanic['Gender']=titanic['Sex'].apply(getGender)

titanic.head()

# Now delete that sex column 
del titanic['Sex']
titanic.head()


# Shift survival column to last .
titanic=titanic[['PassengerId','Pclass','Age','SibSp','Parch', 'Fare', 'Embarked','Gender','Survived']]
titanic.head()

# Check null values in all columns
titanic.isnull().sum()

# Age have 177 null value so it replace by mean 

meanS=titanic[titanic.Survived==1].Age.mean()
meanS

meanNS=titanic[titanic.Survived==0].Age.mean()
meanNS

# Create a new column age which hold mean if NULL is there but only for Survived people

titanic['age']=np.where(pd.isnull(titanic.Age) & titanic['Survived']==1, meanS,titanic['Age'])
titanic.head(20)

# Check Null
titanic.isnull().sum()

# Now fill the age for Non Survived
titanic.age.fillna(meanNS,inplace=True) # Filling for null value - fillna
titanic.head(20)

# Check Null
titanic.isnull().sum()

# Now delete the Age column
del titanic['Age']
titanic.head()

# Again reorder the coulum
titanic=titanic[['PassengerId','Pclass','age','SibSp','Parch', 'Fare', 'Embarked','Gender','Survived']]
titanic.head()


# Check that Embarked column is important or not i.e Embarked play a role in survived 
# Find the no. of people survivde using diffrent Embarked

print(titanic['Embarked'].unique()) # ['S' 'C' 'Q' nan]

#----------------- Check for Survived pople-------------------

SurvivedS=titanic[titanic.Embarked=='S'][titanic.Survived==1].shape[0]
SurvivedS # 217 peopel

SurvivedC=titanic[titanic.Embarked=='C'][titanic.Survived==1].shape[0]
SurvivedC # 93 people

SurvivedQ=titanic[titanic.Embarked=='Q'][titanic.Survived==1].shape[0]
SurvivedQ  # 30 people                                


#----------------- Check for NoN Survived pople-------------------
NSurvivedS=titanic[titanic.Embarked=='S'][titanic.Survived==0].shape[0]
NSurvivedS # 427 peopel

NSurvivedC=titanic[titanic.Embarked=='C'][titanic.Survived==0].shape[0]
NSurvivedC # 75 people

NSurvivedQ=titanic[titanic.Embarked=='Q'][titanic.Survived==0].shape[0]
NSurvivedQ  # 47 people 

# From the above resutl , there are diffrent number of people as per Embarked as survived and non Survived


# Now check for NULL
titanic.isnull().sum() # only 2 NULL in Embarked which can be removed

titanic.dropna(inplace=True)

titanic.isnull().sum() # Now No NULL value

# Now remale the age and gender with 'Age' and 'Sex'
titanic.rename(columns={'age':'Age'},inplace=True)
titanic.head()

titanic.rename(columns={'Gender':'Sex'},inplace=True)
titanic.head()

# in Enbarked 'S' 'C' 'Q' replace with 1,2,3
titanic['Embarked'].unique() 

def getEmb(str):
    if str=='S':
        return 1
    elif str=='C':
        return 2
    else:
        return 3
titanic['Embarked']=titanic['Embarked'].apply(getEmb)

titanic.head()


# Drew the pichart for number of male and female aboard
import matplotlib.pyplot as plt
from matplotlib import style

males=(titanic['Sex']==1).sum()
females=(titanic['Sex']==0).sum()
print(males) # 577
print(females) #312

p=[males, females]
plt.pie(p, labels=['Male','Female'], colors=['green','yellow'], explode=(0.15,0),startangle=0)
plt.show()
    
# Pie chart with Survived and Non Survivde male and females
Male_Sur=titanic[titanic['Sex']==1][titanic.Survived==1].shape[0]
Male_NSur=titanic[titanic['Sex']==1][titanic.Survived==0].shape[0]

FeMale_Sur=titanic[titanic['Sex']==0][titanic.Survived==1].shape[0]
FeMale_NSur=titanic[titanic['Sex']==0][titanic.Survived==0].shape[0]
print(Male_Sur,Male_NSur, FeMale_Sur,FeMale_NSur)

pchart=[Male_Sur,Male_NSur, FeMale_Sur,FeMale_NSur]
plt.pie(pchart,labels=['Male_Sur','Male_NSur', 'FeMale_Sur','FeMale_NSur'],explode=[0,0.05,0,0.1],startangle=100, autopct="%.2f%%")
plt.show()



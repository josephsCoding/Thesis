
from __future__ import division
import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy import stats
import graphviz
from sklearn import feature_selection

data_path ='C:\Users\josep\Documents\University of Toronto\Summaries\\fromPythonCurveFitSimilarRowColumnAvgv3145.csv' 
data = graphlab.SFrame(data_path)

columns_to_remove = ['rowNumber', 'Identifier', 'Model', 'UserID', 'Has a Heat Pump', 'filename', 
                    'Number of Remote Sensors', 'Auxiliary Heat Fuel Type', 'Country', 'Province/State', 'fill']

for column in columns_to_remove:
    data.remove_column(column)
     
# Converting style of home to a categorical variable 
categoricalStyle = data['Style']
categoricalStyle = pd.get_dummies(categoricalStyle)
data = data.to_dataframe()
data = data.merge(categoricalStyle, left_index=True, right_index=True)
 
# Pick and choose the time period that we are interested in
#winter = ['2017_01','2017_02','2017_03' ] 
#spring = ['2017_04','2017_05','2017_06','2016_04','2016_05','2016_06' ] 
#summer = ['2017_07','2017_08','2017_09','2016_07','2016_08','2016_09' ] 
#fall = ['2016_10','2016_11','2016_12' ] 
#cities = ['Austin', 'San Diego', 'Nashville','Toronto', 'Chicago', 'Calgary', 'Miami']
#summer = ['2017_07']
winter = ['2017_01']

# Pick and choose the features that we are interested in

timeOfInterest = winter
climateZone = ['ASHRAE Climate Zone']
modelFeatures = ['Floor Area [ft2]', 'Number of Occupants', 'Number of Floors', 'Age of Home [years]']
houseStyles = ['Apartment','Condominium','Detached','Loft','MultiPlex','RowHouse','SemiDetached','Townhouse']
interestedFeatures = ['Floor Area [ft2]', 'Number of Occupants', 'Number of Floors', 'Age of Home [years]']

# Obtain environmental parameters
indoorRH = []
indoorTemp = []
outdoorRH = []
outdoorTemp = []
adjustments = []

for month in timeOfInterest:
    indoorRH.append('{month} Indoor RH'.format(month=month))
    indoorTemp.append('{month} Indoor Temp'.format(month = month))
    outdoorRH.append('{month} Outdoor RH'.format(month = month))
    outdoorTemp.append('{month} Outdoor Temp'.format(month = month))
    adjustments.append('{month} Adjust'.format(month = month))

# Bins are based on [twice per week, every other day, every day, twice per day]

%matplotlib inline

binsWinter = [0, 8, 16, 31, 62, data['2017_01 Adjust'].max()]
histWinter, binsWinter = np.histogram(data['2017_01 Adjust'], bins=binsWinter)
widthsWinter = 10
centerWinter = (binsWinter[:-1] + binsWinter[1:])/2

binsSummer = [0, 8, 16, 31, 62, int(data['2017_07 Adjust'].max())]
histSummer, binsSummer = np.histogram(data['2017_07 Adjust'], bins=binsSummer)
widthsSummer = 10
centerSummer = (binsSummer[:-1] + binsSummer[1:])/2

plt.bar(centerWinter, histWinter, align='center', width=widthsWinter)
plt.bar(centerSummer, histSummer, align='center', width=widthsSummer)
plt.show()

# Choose 1 Function to create a target column

createTargetColumnBinary: Creates 2 classes. Comfortable if number of comfortable months > uncomfortable months
<br>
createMultipleTargetColumn: Creates multiple classes based on predefined bins
<br>
createMultipleTargetColumn3Class: Creates 3 classes. [Comfortable, uncomfortable, undecided]. 
<br>
Undecided if there are 2 months and each month is split between comfortable and uncomfortable. 
<br>
Undecided if there are 3 months and there is a mix of months that are comfortable and uncomfortable

# Creating the target columns
# A house is comfortable if the number of thermostat adjustments for that month is less than the average thermostat adjustments
# for that month
# If the house is uncomfortable for more than half of the months, then the target column = 'Uncomfortable'
def createTargetColumnBinary(data, adjustments):
    dataTargetColumn = data[adjustments]
    target = []

    for i in range(0, len(dataTargetColumn)):
        comfortableCounter = 0
        uncomfortableCounter = 0
        dataPoint = dataTargetColumn.iloc[i, :]

        for j in range(0, len(adjustments)):
            if dataPoint[j] <= dataTargetColumn.iloc[:, j].mean():
                comfortableCounter += 1
            else:
                uncomfortableCounter += 1
            
        if comfortableCounter > uncomfortableCounter:
            target.append('Comfortable')
        else:
            target.append('Uncomfortable')

    return target

#testTarget = createTargetColumnBinary(data, adjustments)

# Explanation for how the classes are created can be found above

def createMultipleTargetColumn3Class(data, adjustments):
    dataTargetColumn = data[adjustments]
   
    target = []

    for i in range(0, len(dataTargetColumn)):
        comfortableCounter = 0
        uncomfortableCounter = 0
        dataPoint = dataTargetColumn.iloc[i, :]

        for j in range(0, len(adjustments)):
            if dataPoint[j] <= dataTargetColumn.iloc[:, j].mean():
                comfortableCounter += 1
            else:
                uncomfortableCounter += 1
            
        if comfortableCounter >= 3:  # uncomfortableCounter: # and comfortableCounter >=4:
            target.append('Comfortable')
        elif uncomfortableCounter >= 3:  # comfortableCounter: # uncomfortableCounter: # and comfortableCounter <=2:
            target.append('Uncomfortable')
        else:
            target.append('Undecided')

    return target

testTarget = createMultipleTargetColumn3Class(data, adjustments)

# Checking class instability 
print len([i for i, x in enumerate(testTarget) if x == 'Uncomfortable'])
print len([i for i, x in enumerate(testTarget) if x == 'Comfortable'])
print len([i for i, x in enumerate(testTarget) if x == 'Undecided'])

# creating final target column
target = []
for i in range(0, len(targetWinter)):
    target.append(int((targetWinter[i] + targetSummer[i])/2))

print "Number of data points in each class label"
for j in range(1, len(binsWinter)):
    test = [i for i, x in enumerate(target) if x == j]
    testWinter = [i for i, x in enumerate(targetWinter) if x == j]
    testSummer = [i for i, x in enumerate(targetSummer) if x == j]
    print len(testWinter)
    print len(testSummer)
    print len(test)

dataOriginal = pd.DataFrame(data, copy = True)
dataNormalized = pd.DataFrame(data, copy = True)

# Choosing which features are included in the model 
features =  interestedFeatures + houseStyles + climateZone + indoorTemp + indoorRH + outdoorTemp + outdoorRH

# Function to normalize the data (scale between 0 and 1)
def normalize(column):
    newColumn = []
    maxValue = max(column)
    minValue = min(column)

    for i in range(0, len(column)):

        normalizedValue = (column[i] - minValue) / (maxValue - minValue)
        newColumn.append(normalizedValue)

    return newColumn, maxValue, minValue
   
# Normalizing the data
combined = features
dataNormalized = dataNormalized.reset_index(drop=True)
dataUnnormalized = pd.DataFrame(dataNormalized, copy=True)  # creating a copy of the original with the same index
dataToNormalize = dataNormalized[combined]

maxVals = []
minVals = []
minMaxdf = pd.DataFrame({'Value': ['Min', 'Max']})

for i in list(dataToNormalize.columns.values):
    dataColumn = dataToNormalize[i]
    normalizedColumn, maxValue, minValue = normalize(dataColumn)
    dataToNormalize[i] = normalizedColumn
    maxVals.append(maxValue)
    
x = dataNormalized[features]
y = testTarget

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 0)

# Hyperparameter tuning using GridSearchCV

def GridSearch(parameters, x_train, y_train):
    shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=shuffle, scoring='f1_weighted')
    clf.fit(x_train, y_train)
    optimalParam = clf.best_params_
    print clf.cv_results_.get('std_test_score').mean()
    print clf.cv_results_.get('mean_test_score')
    print clf.cv_results_.get('mean_test_score').mean()
    print clf.best_score_
    return optimalParam
  
# Hyperparameter tuning using GridSearchCV
optimalParam = GridSearch({'max_depth': range(3, 20),
                           'criterion': ['gini', 'entropy']}, x_train, y_train)

optimalDepthGSCV = optimalParam.get('max_depth')
optimalCriterion = optimalParam.get('criterion')
optimalClassWeight = optimalParam.get('class_weight')
print "Optimal depth from GridSearchCV: " + str(optimalDepthGSCV) 
print "Optimal criterion from GridSearchCV: " + str(optimalCriterion)
print "optimal class weight from GridSearchCV: " + str(optimalClassWeight)
minVals.append(minValue)
minMaxdf[i] = [minValue, maxValue]
dataNormalized.update(dataToNormalize)

model = tree.DecisionTreeClassifier(class_weight=None, criterion=optimalCriterion, random_state=0,
                                    max_depth=optimalDepthGSCV, splitter='best').fit(x_train, y_train)

# Obtain score metrics
print "Test Set Accuracy: " + str(accuracy_score(y_test, prediction, normalize=True))
print "Training Set Accuracy: " + str(model.score(x_train, y_train))
print "Test Set Accuracy using score: " + str(model.score(x_test, y_test))
print f1_score(y_test, prediction, average=None)
f1_score(y_test, prediction, average='weighted', labels=['Comfortable', 'Uncomfortable'])

# Obtain feature importances
importance = [x for x in model.feature_importances_]
featureImportance = {'Feature': features, 'Importance': importance}
summary = pd.DataFrame(data=featureImportance)
summarySorted = summary.sort_values(by=['Importance'], ascending=False)
summarySorted

# Plot feature importances ranking
plt.figure(figsize=(15, 10))
plt.bar(range(0, len(summarySorted)), summarySorted['Importance'], align='center')
plt.xticks(range(0, len(summarySorted)), summarySorted['Feature'], rotation=90)
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.rc('xtick', labelsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('axes', labelsize=15)
plt.show()

# Confusion matrix to count false positives/negatives
# Rows = actual, columns = prediction
confusion_matrix(y_test, prediction, labels=['Comfortable', 'Undecided', 'Uncomfortable'])
    

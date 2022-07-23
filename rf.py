'''
CP486 Group Project
Student Names: Shaojun Zheng, Zhaoyu Liu
Github ID: ChurchillVII, STEVEHardware
GitHub repository URL: https://github.com/ChurchillVII/CP468-Project
'''
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#read data file
data = pd.read_csv('car.csv') 

print(data)

columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
#Assign integer values to non-numeric attributes
for column in columns:
    data[column] = data[column].replace('low', 1)
    data[column] = data[column].replace('med', 2)
    data[column] = data[column].replace('high', 3)
    data[column] = data[column].replace('vhigh', 4)
    data[column] = data[column].replace('more', 6)
    data[column] = data[column].replace('5more', 6)
    data[column] = data[column].replace('small', 1)
    data[column] = data[column].replace('big', 3)
    data[column] = data[column].replace('unacc', 0)
    data[column] = data[column].replace('acc', 1)
    data[column] = data[column].replace('good', 1)
    data[column] = data[column].replace('vgood', 1)
    data[column] = data[column].replace('0', 0)
    data[column] = data[column].replace('1', 1)
    data[column] = data[column].replace('2', 2)
    data[column] = data[column].replace('3', 3)
    data[column] = data[column].replace('4', 4)
    

print(data)

#split train set and test set at 80/20
x = data.iloc[:, 0:6]
y = data.iloc[:, 6]
y = y.values
x = x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



rf = RandomForestClassifier()

rf.fit(x_train, y_train)
rf_proba = rf.predict_proba(x_test)
rf_pred = rf.predict(x_test)

rf_proba = rf_proba[:,1]

rf_auc = roc_auc_score(y_test, rf_proba)
print("rf_auc:",rf_auc)

rf_fpr, rf_tpr, rf_thresholds = metrics.roc_curve(y_test, rf_proba)
print("rf_fpr:",rf_fpr)
print("rf_tpr:",rf_tpr)
print("rf_thresholds:",rf_thresholds)

cm = confusion_matrix(y_test, rf_pred)
print("confusion_matrix")
print (cm)
print("accuracy_score:",accuracy_score(y_test, rf_pred))
print("precision_score:",precision_score (y_test, rf_pred))
print("recall_score:",recall_score(y_test, rf_pred))
print("f1 score:",f1_score(y_test, rf_pred))


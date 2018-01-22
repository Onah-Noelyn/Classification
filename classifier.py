"""Classifier Selector

This software makes use of scikit learn to build a decision tree and logistic regression 
model with the aim of chosen the best classifier when it's precision and 
predictive capability are compared using the ROC curve

Reference:
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

"""

import csv
import pandas as pd
import numpy as np
import graphviz   #conda install 'python-graphviz' first, if not already
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression



#create an empty list for the dataset
data = [] 

#read in the text file containing the data
f1 = open('BCW_Data.txt', newline='')
dataset = csv.reader(f1)
for row in dataset:				     
    rowlist = []                    
    for value in row:
        rowlist.append(value) 
    data.append(rowlist)
f1.close() 
    
#convert this text file into a pandas dataframe and add column labels
data = pd.DataFrame(data, columns = ['S/N', 'Clump_Thickness', 
'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 'Marginal_Adhesion', 
'Single_Epe_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 
'Mitoses', 'Class'])

#info() is used to check the datatype 
data.info()

#info above indicated 'object' datatype
#convert to numerics
#using coerce to convert non-numerics to NaN - missing data    
data = data.apply(pd.to_numeric, errors='coerce')

#again check for any non-numeric
data.info()

#Bare_Nuclei count total showed some of it were non-numeric
#call to explore 
data['Bare_Nuclei']

#fill missing data with average, as above step indicted NaN cases
data = data.fillna(np.mean(data['Bare_Nuclei']))
data = data.astype(int) #convert any float to integer
data.info() #check if all datatype is int 

#change index to serial number and exclude from dataset
data.index = data['S/N']
data = data[['Clump_Thickness', 'Cell_Size_Uniformity', 
'Cell_Shape_Uniformity', 'Marginal_Adhesion', 'Single_Epe_Cell_Size', 
'Bare_Nuclei','Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']]


#descriptive statistics to understand data
summary = (data.describe().transpose())
print(summary.astype(int))
"""though mean and std is not 1 and 0, 
   normalisation is not needed before prediction 
   as all variables have same range"""

#prepare data for classification 
#where X = independent variables
X = data[['Clump_Thickness', 'Cell_Size_Uniformity', 
'Cell_Shape_Uniformity', 'Marginal_Adhesion', 'Single_Epe_Cell_Size', 
'Bare_Nuclei','Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]

#and y = dependent variable
y = data[['Class']]

#Binarize output
y = pd.Categorical(data.Class).codes


#CLASSIFICATION--------------------------------------------------------------

#STEP 1: Split data into test and train for supervised learning
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        test_size = 0.3, random_state = 123)

#STEP 2: Train the classifiers
#1. Decision Tree Classifier 
classifier = DecisionTreeClassifier(criterion = "gini", random_state = 123,
                               max_depth=3, min_samples_leaf=2)
classifier.fit(X_train, y_train)

#visualise classifier using a decision tree graph
feature_names = ['Clump_Thickness', 'Cell_Size_Uniformity', 
'Cell_Shape_Uniformity', 'Marginal_Adhesion', 'Single_Epe_Cell_Size', 'Bare_Nuclei', 
'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']
target_names = ['Benign', 'Malignant']
dot_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=feature_names,  
                         class_names=target_names,       
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)
#print(graph)
#graph.write_png('somefile.png')
"""tried to print or save the decision tree graph but no success
one work around is to call 'graph' in the ipython console"""


#2. Logistic Regression Classifier   
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


#STEP 3: predict new cases

#1. Decision Tree Predictor
DT_Predictor = classifier.predict(X_test)
#save the outcome from prediction as csv file
DT_Predictor = pd.DataFrame(DT_Predictor)
DT_Predictor.to_csv("C:/Users/Noelyn/Documents/python/decision tree/DT_Predicted.csv")

#2. Logistic Regression Predictor
LR_Predictor = logreg.predict(X_test)
#save the outcome from prediction as csv file
LR_Predictor = pd.DataFrame(LR_Predictor)
LR_Predictor.to_csv("C:/Users/Noelyn/Documents/python/decision tree/LR_Predicted.csv")


#STEP 4: Check Prediction Accuracy
#1. Using R-square: the higher the better
DT_rsquared=classifier.score(X_test,y_test)
LR_rsquared=logreg.score(X_test,y_test)
print('Score for DT_Predictor: ', DT_rsquared, 'and for LR_Predictor: ', LR_rsquared)

#2. confusion matrix: indicates the count of correct vs. incorrect predictions
print('DT confusion matrix is ', confusion_matrix(y_test, DT_Predictor))
print('LR confusion matrix is ',confusion_matrix(y_test, LR_Predictor))

#3. ROC
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
classifier_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr_DT, tpr_DT, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr_LR, tpr_LR, label='Decision Tree Classifier (area = %0.2f)' % classifier_roc_auc)

plt.plot([0,1], [0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




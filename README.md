# CP468-Project

Description:
In this project, we will randomly separate the data into two parts, 80% of the data becomes the training set, and the rest becomes the testing set. First, we will use Classification methods from the Scikit-learn library of Python, including the Gaussian Naive Bayes, K-nearest neighbors, and random forest classifier, to train models with the evaluation attributes of each sample. Then we will calculate the accuracy, precision, recall and f1 score of each model based on their prediction of the testing set. Finally, we will compare the results of different models and find which classification method has the best performance in predicting the given data set. 

Dataset:
The dataset we used is the Car Evaluation Data Set from UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

Installation:
Sklearn library for Python is requierd, The data is in 'car.csv'

Conclusion:
After training and testing our models, we found the random forest classification performed best in all aspects and the Gaussian Naive Bayes classification performed worst compared to other models. As we tested in the Cross-Validation process, the K-nearest neighbor model yields its maximum scores when k=11, which is very close to the result of the random forest. 



# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:57:52 2021

@author: 19145
"""
#drinking classifier
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('final project - smoking ready.csv')
X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
y = dataset.iloc[:,[-1]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.19, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print("")
print("Smoking test")
print ("accuracy score" , "{0:.0%}".format(ac))
print (cm)

#user input
print("Please answer the follow questions. All values must be integers.")
smoking = input("1/19 : do you drink? \nNever = 0, Socially = 1, A lot = 2\n")
reliability = input("2/19 : I am reliable at work and always complete all tasks given to me.: \nStrongly disagree 1-2-3-4-5 Strongly agree\n")
promises = input("3/19 : I always keep my promises.: \n Strongly disagree 1-2-3-4-5 Strongly agree\n")
thinkinga = input("4/19 : I look at things from all different angles before I go ahead.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
work = input("5/19 : I often study or work even in my spare time.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
criticism = input("6/19 : I will find a fault in myself if people don't like me.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
empathy = input("7/19 : I am empathetic person.:\n Strongly disagree 1-2-3-4-5 Strongly agree \n")
eattosur = input("8/19 : I eat because I have to. I don't enjoy food and eat as fast as I can.:\n Strongly disagree 1-2-3-4-5 Strongly agree \n")
borrow = input("9/19 : I look after things I have borrowed from others.:\n Strongly disagree 1-2-3-4-5 Strongly agree \n")
health = input("10/19 : I worry about my health.:\n Strongly disagree 1-2-3-4-5 Strongly agree \n")
changepast = input("11/19 : I wish I could change the past because of the things I have done.:\n Strongly disagree 1-2-3-4-5 Strongly agree \n")
dreams = input("12/19 : I always have good dreams.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
punc = input("13/19 : Timekeeping.:\n I am often early. = 5 - I am always on time. = 3 - I am often running late. =1  \n")
moodswings = input("14/19 : My moods change quickly.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
anger = input("15/19 : I can get angry very easily.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
happy = input("16/19 : I am 100% happy with my life.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
elvl = input("17/19 : I am always full of life and energy.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
personality = input("18/19 : I believe all my personality traits are positive.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
gup = input("19/19 : I find it very difficult to get up in the morning.:\n Strongly disagree 1-2-3-4-5 Strongly agree\n")
enterData= [[smoking,reliability,promises,thinkinga,work,criticism,empathy,eattosur,borrow,health,changepast,dreams,punc,moodswings,anger,happy,elvl,personality,gup]]
enterDataTransformed = sc.transform(enterData)
entered_prediction = classifier.predict(enterDataTransformed)
if entered_prediction == 0:
    print("Answer: You've never smoked.")
if entered_prediction == 1:
    print("Answer: You've tried smoking.")
if entered_prediction == 2:
    print("Answer: You currently smoke.")
if entered_prediction == 3:
    print("Answer: You are a former smoker.")
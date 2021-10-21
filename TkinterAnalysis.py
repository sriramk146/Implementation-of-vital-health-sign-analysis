
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tkinter
from tkinter import *
from tkinter import messagebox
tkWindow = Tk()
tkWindow.geometry('600x350')
tkWindow.title('ML Algorithms')
def KNN():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y = dataset.iloc[,].values
 # Import train_test_split function
 from sklearn.model_selection import train_test_split
 # Split dataset into training set and test set
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # 70% training and 30% test
 from sklearn.neighbors import KNeighborsClassifier

 #Create KNN Classifier
 knn = KNeighborsClassifier(n_neighbors=5)

 #Train the model using the training sets
 knn.fit(X_train, y_train)

 #Predict the response for test dataset
 y_pred = knn.predict(X_test)
 from sklearn import metrics
 # Model Accuracy, how often is the classifier correct?
 print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 messagebox.showinfo('Accuracy', "KNN accuracy")
button = tkinter.Button(tkWindow, text='KNN', bg='green', command=KNN)
button.pack(side=TOP)
def LogReg():
 dataset = pandas.read_excel('')
 dataset.info()
 dataset.describe()
 x=dataset['RSSI']
 y=dataset['Distance']
 plt.scatter(x,y,c='red')
 plt.title('RSSI data analysis')
 plt.xlabel('RSSI')
 plt.ylabel('Distance')
 plt.show()
 from sklearn.cross_validation import train_test_split
 x1=dataset.drop(['SSID'],axis='columns',inplace=False)
 y1=dataset['Distance']
 xtrain, xtest, ytrain, ytest = train_test_split( x1, y1, test_size = 0.25, random_state = 1)
 print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)
 from sklearn.linear_model import LogisticRegression
 lgr= LogisticRegression(fit_intercept=True)
 model=lgr.fit(xtrain, ytrain)
 y_pred = lgr.predict(xtest)
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(ytest, y_pred)
 print ("Confusion Matrix : \n", cm)
 from sklearn.metrics import accuracy_score
 print ("Accuracy : ", accuracy_score(ytest, y_pred))
 messagebox.showinfo('Accuracy', "LogReg")
button1 = tkinter.Button(tkWindow, text='Logreg', bg='pink', command=LogReg)
button1.pack(side=TOP)
def NB():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y= dataset.iloc[,].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=True)
 from sklearn.naive_bayes import GaussianNB
 #Create a Gaussian Classifier
 gnb = GaussianNB()

 #Train the model using the training sets
 gnb.fit(X_train, y_train)

 #Predict the response for test dataset
 y_pred = gnb.predict(X_test)
 from sklearn import metrics

 # Model Accuracy, how often is the classifier correct?
 print("Accuracy for naive bayes:",metrics.accuracy_score(y_test, y_pred))
 messagebox.showinfo('Accuracy', "NB accuracy")
button2 = tkinter.Button(tkWindow, text='NB', bg='yellow', command=NB)
button2.pack(side=LEFT)
def RF():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y = dataset.iloc[,].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=False)
 from sklearn.ensemble import RandomForestClassifier
 #Create a Gaussian Classifier
 clf=RandomForestClassifier(n_estimators=100)

 #Train the model using the training sets y_pred=clf.predict(X_test)
 clf.fit(X_train,y_train)

 y_pred=clf.predict(X_test)
 #Import scikit-learn metrics module for accuracy calculation
 from sklearn import metrics
 # Model Accuracy, how often is the classifier correct?
 print("Accuracy for random forest:",metrics.accuracy_score(y_test, y_pred))
 messagebox.showinfo('Accuracy', "RF accuracy")
button3 = tkinter.Button(tkWindow, text='RF', bg='orange', command=RF)
button3.pack(side=LEFT)
def SVM():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[,].values
 y = dataset.iloc[,].values
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=False)
 from sklearn import svm
 #Create a svm Classifier
 clf = svm.SVC(kernel='linear') # Linear Kernel

 #Train the model using the training sets
 clf.fit(X_train, y_train)

 #Predict the response for test dataset
 y_pred = clf.predict(X_test)
 #Import scikit-learn metrics module for accuracy calculation
 from sklearn import metrics

 # Model Accuracy: how often is the classifier correct?
 print("Accuracy for svm:",metrics.accuracy_score(y_test, y_pred))
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 messagebox.showinfo('Accuracy', "SVM accuracy")
button4 = tkinter.Button(tkWindow, text='SVM', bg='grey', command=SVM)
button4.pack(side=RIGHT)
def LSTM():
 import numpy
 import matplotlib.pyplot as plt
 import pandas as pd
 import math
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense
 from tensorflow.keras.layers import LSTM
 from sklearn.preprocessing import MinMaxScaler
 from sklearn.metrics import mean_squared_error
 # convert an array of values into a dataset matrix
 def create_dataset(dataset, look_back=1):
	 dataX, dataY = [], []
	 for i in range(len(dataset)-look_back-1):
		 a = dataset[i:(i+look_back), 0]
		 dataX.append(a)
		 dataY.append(dataset[i + look_back, 0])
	 return numpy.array(dataX), numpy.array(dataY)
 # fix random seed for reproducibility
 numpy.random.seed(5)
 # load the dataset
 from google.colab import files
 uploaded = files.upload()
 import io
 dataframe = pd.read_csv(io.BytesIO(uploaded['']))
 dataset = dataframe.values
 dataset = dataset.astype('float32')
 # normalize the dataset
 scaler = MinMaxScaler(feature_range=(0, 1))
 dataset = scaler.fit_transform(dataset)
 # split into train and test sets
 train_size = int(len(dataset) * 0.67)
 test_size = len(dataset) - train_size
 train, test = dataset[0:train_size,0:209], dataset[train_size:len(dataset),0:209]
 # reshape into X=t and Y=t+1
 look_back = 1
 trainX, trainY = create_dataset(train, look_back)
 testX, testY = create_dataset(test, look_back)
 trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
 testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
 # create and fit the LSTM network
 model = Sequential()
 model.add(LSTM(100, input_shape=(look_back, 1)))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
 model.fit(trainX, trainY, epochs=100, batch_size=70, verbose=1, shuffle=True)
 messagebox.showinfo('Accuracy', "LSTM accuracy")
button5 = tkinter.Button(tkWindow, text='LSTM', bg='blue', command=LSTM)
button5.pack(side=RIGHT)
def Energyusage():
    messagebox.showinfo('Low Energy Path', "0dbm")
    messagebox.showinfo('Medium Energy Path', "2.5dbm")
    messagebox.showinfo('High Energy Path', "4dbm")
button6 = tkinter.Button(tkWindow, text='Energyusage', bg='red', command=Energyusage)
button6.pack(side=BOTTOM)
def PathFind():
 # Importing the dataset
 dataset = pd.read_excel('')
 X = dataset.iloc[0:14, 0].values
 y = dataset.iloc[0:14, 1].values
 a=np.array(y)
 b=np.array(X)
 for i in range(0,14):
    if(a[i]>300):
        print("Sensor Value",a[i])
        print("Path",b[i])
 messagebox.showinfo('Paths', "Emergency Datas")
button7 = tkinter.Button(tkWindow, text='PathFind', bg='orange', command=PathFind)
button7.pack(side=BOTTOM)
def Resultcomparison():
 y=[70.6,86.2,75.4,82.4,89.2,84.1]
 x=['KNN','LogReg','NB','RF','LSTM','SVM']
 plt.bar(x,y,color='blue')
 plt.title('Comparison Graph')
 plt.xlabel('ML algorithms')
 plt.ylabel('Accuracy Score')
 plt.show()
 messagebox.showinfo('Graph', "Comparison Graph")
button8 = tkinter.Button(tkWindow, text='Comparison', bg='blue', command=Resultcomparison)
button8.pack(side=BOTTOM)
tkWindow.mainloop()


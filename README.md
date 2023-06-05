# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library packages.
2.Import the dataset to operate on.
3.Split the dataset into required segments.
4.Predict the required output.
5.Run the program
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVINESH M
RegisterNumber: 212222230064 
*/
import chardet 
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.isnull().sum()
data.head()
data.info()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### data.head():
![243392413-2e4f16be-3aa6-458b-a8b3-8fe44e06db1a](https://github.com/kavinesh8476/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118466561/b832a901-0f8d-4783-9eec-c5e222c15643)


### data.info():
![243393052-00a6d72d-e64a-4334-b1c8-b978bc5b9335](https://github.com/kavinesh8476/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118466561/43a24ff5-9859-4609-855c-3e9bdb7da615)


### data.isnull().sum():
![243394255-a26144eb-819b-4726-8501-1a2db538ee84](https://github.com/kavinesh8476/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118466561/f1c8cfd3-ba60-4009-a92c-843f4e66da12)

### Y_prediction value:
![243394155-f4797ac7-09b2-4ece-877a-28619baf0fc8](https://github.com/kavinesh8476/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118466561/3d5d5d3d-f577-428e-84b0-c6daac5200b3)

### Accuracy value:
![243394462-4777dd1f-2e78-4db3-8568-1f352d201539](https://github.com/kavinesh8476/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118466561/dea58504-52a7-486f-8a88-566d8c921773)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

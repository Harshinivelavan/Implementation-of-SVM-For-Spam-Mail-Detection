# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARSHINI.V
RegisterNumber:  212224040109
*/



import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

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
![Screenshot 2025-05-17 092242](https://github.com/user-attachments/assets/61ba54f0-ab12-43eb-a8e2-848ddd716317)

![Screenshot 2025-05-17 092254](https://github.com/user-attachments/assets/38b1d458-9be1-4647-ab4c-274e48bce3eb)

![Screenshot 2025-05-17 092302](https://github.com/user-attachments/assets/cf264704-b57c-4b6e-9137-f7e9b3f641a2)

![Screenshot 2025-05-17 092310](https://github.com/user-attachments/assets/35257e2b-b83f-4ef9-9bb1-caace6e0ac6b)

![Screenshot 2025-05-17 092316](https://github.com/user-attachments/assets/858eba77-2958-440a-9b16-7e2dc0063daf)







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

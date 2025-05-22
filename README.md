# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result. 
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAHINI A
RegisterNumber:  212223230165
*/
import pandas as pd
data=pd.read_csv("/content/spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
# Data:
![Screenshot 2025-05-22 133009](https://github.com/user-attachments/assets/6749b844-df0e-4a9a-942f-4b8b81416402)

# Data.Shape():
![image](https://github.com/user-attachments/assets/8d0d4b73-9eb6-4d57-832b-41f979fb7004)

# x.shape:
![image](https://github.com/user-attachments/assets/69fd6f2a-2576-4342-a86f-71f16fa5e9a4)

# y.shape:
![image](https://github.com/user-attachments/assets/a2cbf6d3-a3e7-4a57-b5f1-a2fb2840b578)

# x.train:
![image](https://github.com/user-attachments/assets/c9875928-b2c4-47f7-bdee-96710d73a9b0)

# x_train.shape():
![image](https://github.com/user-attachments/assets/a39acc64-218a-454a-adc4-6e25f97bd5cf)

# y_pred:
![image](https://github.com/user-attachments/assets/09978815-db5f-46bc-a679-9bc10d078b96)

# acc(accuracy):
![Screenshot 2025-05-22 133332](https://github.com/user-attachments/assets/36e8685c-eaf7-4bc9-af22-0b8ccb961f8f)

# con(confusion matrix):
![image](https://github.com/user-attachments/assets/6c87873e-7f32-4da5-bbf3-5c7cfe90cfb6)

# cl(classification report):
![image](https://github.com/user-attachments/assets/3269f638-8afe-4f68-8637-6ca5592a1daa)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

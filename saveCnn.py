# this is for loading the dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('feature-envy.csv',error_bad_lines=False)
#for i in dataset.columns:
#    dataset.i = pd.to_numeric(dataset.i, error='ignore')


# view dataset
X= dataset.iloc[:,5:88]
y= dataset.iloc[:,87]
print(X.head(5))
print(y.head(3))
# standardization
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)
#print(X)
from sklearn import preprocessing
# Get column names first
names = dataset.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
#scaled_df = scaler.fit_transform(dataset)
#scaled_df = pd.DataFrame(scaled_df, columns=names)
print("hnaaaa")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("hnaaaa")


# model setting
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

classifier=load_model('feature-envymodel.h5')
#classifier.fit(X_train,y_train, batch_size=10, epochs=1000)
train_log=classifier.fit(X_train,y_train, batch_size=10, epochs=500)
test_log=classifier.fit(X_test,y_test, batch_size=10, epochs=500)
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,500),train_log.history["loss"],label="train_loss",color="r")
plt.plot(np.arange(0,500),train_log.history["acc"],label="train_acc",color="g")
plt.plot(np.arange(0,500),test_log.history["loss"],label="test_loss",color="y")
plt.plot(np.arange(0,500),test_log.history["acc"],label="test_acc",color="b")
plt.title("Training Loss and Accuracy on BPNN classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Loss_Accuracy_alexnet_{:d}e.png".format(500))
eval_model=classifier.evaluate(X_train, y_train)
eval_model
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)
# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy_score
from sklearn.metrics import accuracy_score
print("accuracy:",accuracy_score(y_test,y_pred))
#print(accuracy_score(y_test,y_pred,normalize=False))

# roc_auc_score
#from sklearn.metrics import roc_auc_score
#rc = roc_auc_score(y_test,y_pred)
#print(rc)

# recall_score
#from sklearn.metrics import recall_score
#print(recall_score(y_test,y_pred,average='macro'))
#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=200)
eval_model=classifier.evaluate(X_train, y_train)
eval_model
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)
# confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy_score
from sklearn.metrics import accuracy_score
print("accuracy:",accuracy_score(y_test,y_pred))
#print(accuracy_score(y_test,y_pred,normalize=False))

# roc area
from sklearn.metrics import roc_auc_score
print('ROC:',roc_auc_score(y_test,y_pred))

# recall_score
#from sklearn.metrics import recall_score
#print(recall_score(y_test,y_pred,average='macro'))

# f1_score
from sklearn.metrics import f1_score
print('f1_score:',f1_score(y_test,y_pred,average='macro'))

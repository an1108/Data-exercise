import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import model_selection
from sklearn.metrics import roc_curve




df = pd.read_csv('flights.csv')

ak = pd.read_csv('weather.csv') 


df = pd.merge(df,ak[['origin','year','month','day','hour','visib','wind_speed']],
                  how = 'left',
                  left_on = ['origin','year','month','day','hour'],
                  right_on = ['origin','year','month','day','hour']) #adding the column visib and wind_speed from weather into df

df['is_late'] = np.nan #empty column that will contain if the flight is late or not

df.loc[df['arr_delay'] > 15,['is_late']] = 1 #assigns 1 if the flight is late
df.loc[df['arr_delay'] <= 15,['is_late']] = 0 #assigns 0 if the flight is not late




k  = pd.notnull(df[['arr_delay','visib','wind_speed','distance','is_late']]) #not consider any entries that have NaN as the entry in the columns specified
j = df[k] #create a new dataframe that only contains the data 

x = j[['distance','visib','wind_speed']].to_numpy().astype(int) #create a numpy array with each column being distance, visibility and wind speed

y = j['is_late'].to_numpy().astype(int) #numpy array with the outcomes of the flight being late or not



model = LogisticRegression() #create a logistic regression

x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = 0.2) #creating a train and test set with 80% to train and 20% to test
model = model.fit(x_train,y_train)

result = model.score(x_test,y_test) #testing the accuracy of the model on the test data
print(result)

probs = model.predict_proba(x_test) #predict the probabilities 
probs = probs[:,1] #keep the probability of the positive outcome


fpr, tpr, thresholds = roc_curve(y_test, probs,pos_label=1) #calculating the ROC curve
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.show()

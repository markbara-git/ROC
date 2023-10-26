import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#path to input csv
input = "input\\input-we.csv"

#get the related columns
data = pd.read_csv(input,usecols=['geopixel', 
                                  'cve_css_total_score',
                                  'osi_model_layer1_security_posture_assessment',
                                  'osi_model_layer2_security_posture_assessment',
                                  'osi_model_layer3_security_posture_assessment',
                                  'os_os_security_assessment'])

#prepare the 'cve_css_total_score'
data['cve_css_total_score'] = data['cve_css_total_score'].values * 0.1

#group data by 'goepixel'
data = data.groupby('geopixel').mean().reset_index()

#append the 'score' column based on 'cve_css_total_score
data['score'] = data['cve_css_total_score'].apply(lambda x: 1 if x >= 0.5 else 0)

#append the 'score2' column based on the 'os_os_security_assessment'
data['score2'] = data['os_os_security_assessment'].apply(lambda x: 1 if x >= 0.5 else 0)

#save modified data into CSV file
#data.to_csv('input\\slim.csv',index=False)
#print(data)

X = data[['score2',
          'osi_model_layer1_security_posture_assessment',
          'osi_model_layer2_security_posture_assessment',
          'osi_model_layer3_security_posture_assessment',
          'os_os_security_assessment']]
y = data['score']

#train data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

#Logistic Regresion
log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)

y_pred = log_regression.predict(X)

#create confusion matrix
confusionMatrix = confusion_matrix(y,y_pred)

#define metrics
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


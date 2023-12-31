import pandas as pd
import os
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#path to input csv
input = "input\\input-0-r.csv"
output = "output\\"

#get the related columns
data = pd.read_csv(input,usecols=['geopixel', 
                                  'osi_model_layer6_security_posture_assessment',
                                  'ia_confidentiality_assessment',
                                  'ia_authenticity_assessment',
                                  'ia_integrity_assessment',
                                  'ia_availability_assessment',
                                  'device_total_security_score'])



#prepare the values in columns
data['osi_model_layer6_security_posture_assessment'] = data['osi_model_layer6_security_posture_assessment'].values * 0.1
data['ia_confidentiality_assessment'] = data['ia_confidentiality_assessment'].values * 0.1
data['ia_authenticity_assessment'] = data['ia_authenticity_assessment'].values * 0.1
data['ia_integrity_assessment'] = data['ia_integrity_assessment'].values * 0.1
data['ia_availability_assessment'] = data['ia_availability_assessment'].values * 0.1
data['device_total_security_score'] = data['device_total_security_score'].values * 0.1

#Replace Na/NaN with 0.5 value
data = data.fillna(0.5)

#group data by 'goepixel'
data = data.groupby('geopixel')

#lists to store data
geolist = []
auclist = []

#spliting by geopixel into seperate CSV files
for name, group in data:
    group.to_csv(output+f'{name}.csv',index=False)
    geolist.append(output+f'{name}.csv')

#proccesing each geopixel
for geo in geolist:
    data = pd.read_csv(geo)
    try:
        #append the 'score' column based on 'cve_css_total_score
        data['score'] = data['device_total_security_score'].apply(lambda x: 1 if x >= 0.5 else 0)

        #append the 'score2' column based on the 'os_os_security_assessment'
        data['score2'] = data['osi_model_layer6_security_posture_assessment'].apply(lambda x: 1 if x >= 0.5 else 0)
        
        X = data[['score2',
                  'ia_confidentiality_assessment',
                  'ia_authenticity_assessment',
                  'ia_integrity_assessment',
                  'ia_availability_assessment']]
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
        auclist.append(auc)        
    except:
        #if not enough data then set the AUC value to -1
        auclist.append(-1)
    
    #deleting temp csv    
    os.remove(geo)

#formating the geolist
geolist = [s[:-4] for s in geolist]
geolist = [s[7:] for s in geolist]

#creating the CSV output
df = pd.DataFrame({'geopixel': geolist, 'auc': auclist})
df.to_csv(output+'output-r.csv',index=False)


#create ROC curve
#plt.plot(fpr,tpr,label="AUC="+str(auc))
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.legend(loc=4)
#plt.show()


import pandas as pd
import os
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

#path to input csv
input = "input\\input-0.csv"
output = "output\\"
testsize = 0.75

#get the related columns
data = pd.read_csv(input,usecols=['geopixel'])
data = data.fillna("1_1")

data_rest = pd.read_csv(input,usecols=['osi_model_layer2_security_posture_assessment',
                                       'osi_model_layer4_security_posture_assessment',
                                       'osi_model_layer6_security_posture_assessment',
                                       'ia_confidentiality_assessment',
                                       'ia_integrity_assessment',
                                       'ia_availability_assessment'])

data_rest = data_rest.fillna(5)

#normalize the values in columns
data_rest['osi_model_layer2_security_posture_assessment'] = data_rest['osi_model_layer2_security_posture_assessment'] * 0.1
data_rest['osi_model_layer4_security_posture_assessment'] = data_rest['osi_model_layer4_security_posture_assessment'] * 0.1
data_rest['osi_model_layer6_security_posture_assessment'] = data_rest['osi_model_layer6_security_posture_assessment'] * 0.1
data_rest['ia_confidentiality_assessment'] = data_rest['ia_confidentiality_assessment'] * 0.1
data_rest['ia_integrity_assessment'] = data_rest['ia_integrity_assessment'] * 0.1
data_rest['ia_availability_assessment'] = data_rest['ia_availability_assessment'] * 0.1

#combine both data
data = pd.concat([data,data_rest], axis=1)

#remove not needed dataframe
del data_rest

#data.to_csv(output+'output-0-10.csv',index=False)

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
        data['score'] = data['osi_model_layer4_security_posture_assessment'].apply(lambda x: 1 if x >= 0.5 else 0)

        #append the 'score2' column based on the 'os_os_security_assessment'
        data['score2'] = data['osi_model_layer2_security_posture_assessment'].apply(lambda x: 1 if x >= 0.5 else 0)
        
        X = data[['score2',
                  'osi_model_layer2_security_posture_assessment',
                  'osi_model_layer4_security_posture_assessment',
                  'osi_model_layer6_security_posture_assessment',
                  'ia_confidentiality_assessment',
                  'ia_integrity_assessment',
                  'ia_availability_assessment']]
        y = data['score']
        
        #train data
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testsize, random_state=int(time.time()))

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
        #if not enough data then set the AUC value to 0
        auclist.append(-1)
    
    #deleting temp csv    
    os.remove(geo)

#formating the geolist
geolist = [s[:-4] for s in geolist]
geolist = [s[7:] for s in geolist]

#creating the CSV output
df = pd.DataFrame({'geopixel': geolist, 'auc': auclist})
df = df.sort_values(by='geopixel', ascending=True)
df.to_csv(output+'output-0.csv',index=False)

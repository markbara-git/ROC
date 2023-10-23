import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

input = "input\\input-we-10.csv"

data = pd.read_csv(input,usecols=['geopixel', 
                                  'cve_css_total_score',
                                  'osi_model_layer1_security_posture_assessment',
                                  'osi_model_layer2_security_posture_assessment',
                                  'osi_model_layer3_security_posture_assessment',
                                  'os_os_security_assessment'])

data['cve_css_total_score'] = data['cve_css_total_score'].values * 0.1
data = data.groupby('geopixel').mean().reset_index()
data['score'] = data['cve_css_total_score'].apply(lambda x: 1 if x >= 0.5 else 0)

ax = plt.subplot()

for name, group in data:
    y_true = group['score'].values
    y_score = group[['osi_model_layer1_security_posture_assessment',
                     'osi_model_layer2_security_posture_assessment',
                     'osi_model_layer3_security_posture_assessment',
                     'os_os_security_assessment']].values
    
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(y_score.Shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true, y_score[:, i])

        ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {name}')
    for i in range(y_score.shape[1]):
        ax.plot(fpr[i], tpr[i], label=f'{i} (AUC = {roc_auc[i]:.2f})')
    ax.legend(loc="lower right")
plt.show()

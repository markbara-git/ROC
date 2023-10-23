import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import LabelEncoder

input = "input\\input-we-10-slim.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(input)

# Group DataFrame by unique geopixel
groups = df.groupby('geopixel')

# Initialize figure and axes
fig, ax = plt.subplots()

# Iterate over groups and plot ROC curve for each column
for name, group in groups:
    # Extract required columns and convert to numpy arrays
    y_true = group['wynik'].values
    y_score = group[['cve_css_total_score', 'os_os_security_assessment', 'osi_model_layer1_security_posture_assessment', 'osi_model_layer2_security_posture_assessment', 'osi_model_layer3_security_posture_assessment']].values

    # Convert y_true to binary values
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)

    # Compute ROC curve and AUC score for each column
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_score.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true, y_score[:, i], multi_class='ovr')

    # Plot ROC curves for each column
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

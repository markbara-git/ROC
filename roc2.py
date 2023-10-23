import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

input = "input\\input-we-10.csv"
# Read the CSV file into a DataFrame
data = pd.read_csv(input, usecols=['geopixel','osi_model_layer1_security_posture_assessment','osi_model_layer2_security_posture_assessment','osi_model_layer3_security_posture_assessment',
           'cve_css_total_score',
           'os_os_security_assessment'])
#data.to_csv("input\\input-we-10-slim.csv",index=False)
data = data.fillna(0)

# Group data by unique geopixel values
unique_geopixels = data['geopixel'].unique()

# Set up the plot
plt.figure(figsize=(10, 6))

# Define a color cycle for plotting
colors = cycle(['darkorange', 'navy', 'red', 'green', 'purple'])

# Columns to use for ROC plot
"""columns = ['osi_model_layer1_security_posture_assessment',
           'osi_model_layer2_security_posture_assessment',
           'osi_model_layer3_security_posture_assessment',
           'osi_model_layer4_security_posture_assessment',
           'osi_model_layer5_security_posture_assessment',
           'osi_model_layer6_security_posture_assessment',
           'osi_model_layer7_security_posture_assessment',
           'cve_css_total_score',
           'os_os_security_assessment']"""
columns = ['osi_model_layer1_security_posture_assessment','osi_model_layer2_security_posture_assessment','osi_model_layer3_security_posture_assessment',
           'cve_css_total_score',
           'os_os_security_assessment']

for geopixel, color in zip(unique_geopixels, colors):
    geopixel_data = data[data['geopixel'] == geopixel]
    
    # Create empty lists to store results for each column
    fprs, tprs, aucs = [], [], []

    for col in columns:
        scores = geopixel_data[col]
        if col == 'cve_css_total_score':
            threshold = 5
        else:
            # Set a threshold to classify as positive or negative
            threshold = 0.5  # Adjust as needed        

        # Classify data based on the threshold
        labels = scores >= threshold

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(labels, geopixel_data['cve_css_total_score'])

        # Calculate AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)

        # Append results for the current column
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

        # Plot the ROC curve for the current column
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve ({col}) (area = {roc_auc:.2f})')

    # Optionally, you can calculate and display the average AUC for all columns
    avg_auc = sum(aucs) / len(aucs)
    print(f'Average AUC for {geopixel}: {avg_auc:.2f}')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic per Geopixel')
plt.legend(loc="lower right")
plt.show()

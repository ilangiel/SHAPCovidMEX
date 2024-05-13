import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
import shap
import pickle
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt


namedata = "Covid19MexInv2021-2022EDOS"
data = pd.read_csv(namedata+".csv", encoding='latin-1', low_memory=False)
print(data['class'].value_counts())

# Change the last column 'class' to 0 and 1
data['class'] = data['class'].apply(lambda x: 1 if x == 'Survived' else 0)

model = RandomForestClassifier(random_state=42)

skf = StratifiedKFold(n_splits=10)

X = data.drop(['class'], axis=1)
y = data['class']

metrics_results = {
    'train_accuracy': [],
    'test_accuracy': [],
    'train_f1_macro': [],
    'test_f1_macro': [],
    'train_geom_mean': [],
    'test_geom_mean': []
}

for train_idx, test_idx in skf.split(X, y):
    x_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    x_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]
    
    model.fit(x_train_fold, y_train_fold)
    y_train_pred = model.predict(x_train_fold)
    y_test_pred = model.predict(x_test_fold)
    
    # Collecting all metrics
    metrics_results['train_accuracy'].append(accuracy_score(y_train_fold, y_train_pred))
    metrics_results['test_accuracy'].append(accuracy_score(y_test_fold, y_test_pred))
    metrics_results['train_f1_macro'].append(f1_score(y_train_fold, y_train_pred, average='macro'))
    metrics_results['test_f1_macro'].append(f1_score(y_test_fold, y_test_pred, average='macro'))
    metrics_results['train_geom_mean'].append(geometric_mean_score(y_train_fold, y_train_pred))
    metrics_results['test_geom_mean'].append(geometric_mean_score(y_test_fold, y_test_pred))

# Print results for each metric
for key, value in metrics_results.items():
    print(f"{key}: {np.mean(value):.4f}")

# Save the model to disk
filename = 'model_TD_' + namedata + '.sav'
pickle.dump(model, open(filename, 'wb'))

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Save SHAP values
fileshapvalues = 'SHAP_TD_' + namedata + '.sav'
pickle.dump(shap_values, open(fileshapvalues, 'wb'))


#plot
plt.figure()
shap.summary_plot(shap_values[1], X)
plt.savefig('shap_summary_plotPoints2021-2022_0.png', dpi=300)
plt.close()



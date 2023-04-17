import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, train_test_split, cross_val_score
from pytorch_tabnet.tab_model import TabNetClassifier
from notebooks import *
import smote_variants as sv
import seaborn as sns


#loading
df_file_path = "./data_exp/dataset-filtered-8.csv"
df = pd.read_csv(df_file_path, low_memory=False)
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

# X_samp and y_samp contain the oversampled dataset
oversampler= sv.distance_SMOTE()
X_samp, y_samp= oversampler.sample(X, y_binary)
print(type(X_samp),type(y_samp))

X_train_full, X_test, y_train_full, y_test = train_test_split(X_samp, y_samp, test_size=0.3, random_state=42, stratify=y_samp)
print(X_train_full.shape)

default_params = {'gamma': 1.2, 'lambda_sparse': 0.0001, 'momentum': 0.3, 'optimizer_params': {'lr': 0.02}, 'verbose': 0}

tabnet = TabNetClassifier( **default_params)

def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# define dataset
X, y = X_train_full, y_train_full

scores = evaluate_model(tabnet, X, y)

print('>%s %.3f (%.3f)' % ('Tabnet', np.mean(scores), np.std(scores)))

#
tabnet.fit(X_train_full, y_train_full)
para_best = tabnet.get_params()
print(para_best)
feat_importances = tabnet.feature_importances_
indices = np.argsort(feat_importances)

#feature importance
plot_feature_importance(feature_names, feat_importances)

y_predicted = tabnet.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_predicted)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()
print('Test Validation - Accuracy:', accuracy_score(y_test, y_predicted),
      'Preciation:', precision_score(y_test, y_predicted) ,
      'Recall:', recall_score(y_test, y_predicted),
      'F1:', f1_score(y_test,y_predicted),
      'ROC_AUC:', roc_auc_score(y_test,y_predicted))
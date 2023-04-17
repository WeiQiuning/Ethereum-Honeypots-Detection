from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from notebooks import *

#loading
df_file_path = "./data_exp/dataset-filtered-balance.csv"
df = pd.read_csv(df_file_path, low_memory=False)
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
print(X_train_full.shape)

#model
params = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(1000,1500,100),
    'learning_rate': [0.1, 0.01, 0.05]
}
clf = XGBClassifier(
    objective= 'binary:logistic',
    seed=42
)
tuned_clf = GridSearchCV(estimator=clf,
                         param_grid=params,
                         scoring='f1',
                         cv=5,
                         verbose=2,
                         )
tuned_clf.fit(X_train_full,y_train_full)

print("Tuned Hyperparameters :", tuned_clf.best_params_)
print("Train F1 Score :",tuned_clf.best_score_)
best_y_pr = tuned_clf.predict(X_test)
print('Test F1 Score: ', f1_score(y_test, best_y_pr))

#feature importance
feat_importances = tuned_clf.best_estimator_.feature_importances_
indices = np.argsort(feat_importances)
plot_feature_importance(feature_names, feat_importances)

tuned_clf.best_estimator_
tuned_clf.best_score_

#evaluate
y_predicted = tuned_clf.predict(X_test)
print('Test Validation - Accuracy:', accuracy_score(y_test, y_predicted),
      'Preciation:', precision_score(y_test, y_predicted) ,
      'Recall:', recall_score(y_test, y_predicted),
      'F1:', f1_score(y_test,y_predicted),
      'ROC_AUC:', roc_auc_score(y_test,y_predicted))
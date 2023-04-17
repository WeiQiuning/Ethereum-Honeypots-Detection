import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from notebooks import *

#loading
df_file_path = "./data_exp/dataset-filtered-balance.csv"
df = pd.read_csv(df_file_path, low_memory=False)
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
print(X_train_full.shape)

params = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "num_leaves": [20, 50, 100, 500],
    "max_depth": [i for i in range(3, 8)],
    "max_bin": [200, 300],
    "min_gain_to_split": [i for i in range(0, 15, 5)],
    "bagging_fraction": [0.2, 0.5, 0.95],
    "bagging_freq": [1],
    "feature_fraction": [0.2, 0.5, 0.95]
}
clf = lgb.LGBMClassifier()

tuned_clf = GridSearchCV(estimator=clf,
                         param_grid=params,
                         scoring='f1',
                         cv=5,
                         verbose=0,
                         )
# tuned_clf.fit(X_train_full,y_train_full)
# print("Tuned Hyperparameters :", tuned_clf.best_params_)

lightgbm_params = {"bagging_fraction": 0.5,
                   "bagging_freq": 1,
                   "feature_fraction": 0.5,
                   "learning_rate": 0.2,
                   "max_bin": 200,
                   "max_depth": 3,
                   "min_gain_to_split": 0,
                   "num_leaves": 20}

lightgbm_model = lgb.LGBMClassifier(**lightgbm_params)
lightgbm_model.fit(X_train_full, y_train_full)

#feature importance
feat_importances = permutation_importance(lightgbm_model, X_test, y_test)['importances_mean']
indices = np.argsort(feat_importances)
plot_feature_importance(feature_names, feat_importances)


y_predicted = lightgbm_model.predict(X_test)
print('Test Validation - Accuracy:', accuracy_score(y_test, y_predicted),
      'Preciation:', precision_score(y_test, y_predicted) ,
      'Recall:', recall_score(y_test, y_predicted),
      'F1:', f1_score(y_test,y_predicted),
      'ROC_AUC:', roc_auc_score(y_test,y_predicted))
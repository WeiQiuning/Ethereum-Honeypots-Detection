import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV, train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from keras.constraints import max_norm as MaxNorm
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import random
random.seed(42)
from notebooks import *


#loading
df_file_path = "./data_exp/dataset-filtered-8.csv"
df = pd.read_csv(df_file_path, low_memory=False)
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)
scale_pos_weight = compute_scale_pos_weight(y_binary)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
print(X_train_full.shape)

#feature engineering
# Scaling
# only use training data to fit, to avoid data leakage
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)
print(np.isnan(X_train_full))

#Optimal Parameters for each model from hyperparameter tuning
tabnet_params ={'gamma': 1.2,
                'lambda_sparse': 0.0001,
                'momentum': 0.3,
                'n_steps': 3,
                'optimizer_params': {'lr': 0.02},
                'verbose': 0}

xgb_params = {'learning_rate': 0.05,
              'max_depth': 3,
              'n_estimators': 50,
              'scale_pos_weight': scale_pos_weight}

mlp_params = {'input_dim': X_train_full.shape[1],
              'H': 60,
              'activation': 'relu',
              'dropout_probability': 0.2,
              'num_epochs': 75,
              'num_layers': 5,
              'scale_pos_weight': scale_pos_weight}

svm_params = {'C': 1000,
              'gamma': 1}

rf_params = {'max_depth': 20,
               'min_samples_leaf': 5,
               'n_jobs': -1}

lightgbm_params = {"bagging_fraction": 0.5,
                   "bagging_freq": 1,
                   "feature_fraction": 0.5,
                   "learning_rate": 0.2,
                   "max_bin": 200,
                   "max_depth": 3,
                   "min_gain_to_split": 0,
                   "num_leaves": 20,
                   'scale_pos_weight': scale_pos_weight}


def compile_mlp(input_dim, H, num_epochs, num_layers, activation, dropout_probability):
    # Creating Sequential MLP
    model_n = Sequential()
    model_n.add(layers.Dense(H, input_shape=(input_dim, ), activation= activation))

    for _ in range(num_layers - 1):
        model_n.add(layers.Dense(H, activation= activation, kernel_constraint=MaxNorm(3)))
        model_n.add(layers.Dropout(dropout_probability))

    model_n.add(layers.Dense(1, activation='sigmoid'))
    # configure the model
    model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(from_logits=True)])
    return model_n


# get a list of models to evaluate
def get_models():
    models = dict()
    mlp = KerasClassifier(model=compile_mlp, **mlp_params)
    tabnet = TabNetClassifier(**tabnet_params)
    models['randomforest'] = RandomForestClassifier(**rf_params)
    models['svm'] = svm.SVC(**svm_params)
    models['mlp'] = mlp
    models['xgboost'] = XGBClassifier(**xgb_params)
    models['lightGBM'] = lgb.LGBMClassifier(**lightgbm_params)
    models['tabnet'] = tabnet
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)  # , error_score='raise')
    return scores

# define dataset
X, y = X_train_full, y_train_full
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
predictions = list()
for name, model in models.items():
    if name == 'mlp':
        names.append(name)
        model.fit(X, y)
        predictions.append(model.predict(X_test))
        print(name + ' done')
    else:
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        model.fit(X, y)
        predictions.append(model.predict(X_test))
        print(name + 'done')
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


results_df = pd.DataFrame()
results_df['Model'] = names
results_df['Optimal Parameters'] = [rf_params,
                                    svm_params,
                                    mlp_params,
                                    xgb_params,
                                    lightgbm_params,
                                    tabnet_params]

metrics_dict = {'Accuracy': accuracy_score,
                'Precision': precision_score,
                'Recall': recall_score,
                'F1': f1_score,
                'ROC-AUC': roc_auc_score}
for metric, func in metrics_dict.items():
    storage = []
    for prediction in predictions:
        storage.append(func(y_test, prediction))
    results_df[metric] = storage

results_df.sort_values(['Accuracy', 'ROC-AUC'], ascending = [False, False])
results_df.to_csv('./results/results_unbalance.csv')
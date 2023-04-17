import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import scikeras
from keras.models import Sequential
from keras import layers
from keras.constraints import MaxNorm
from keras.metrics import Recall, RecallAtPrecision
from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from notebooks import *

#loading
df_file_path = "./data_exp/dataset-filtered-balance.csv"
df = pd.read_csv(df_file_path, low_memory=False)
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
print(X_train_full.shape)

# metrics to evaluate is the highest recall we can get for a precision above 0.75
RecallAtPrecision_scorer = RecallAtPrecision(precision=0.75)


def compile_mlp(input_dim, H, num_epochs, num_layers, activation, dropout_probability):

    # Creating Sequential MLP
    model_n = Sequential()

    model_n.add(layers.Dense(H, input_shape=(input_dim, ), activation= activation))

    for _ in range(num_layers - 1):
        model_n.add(layers.Dense(H, activation= activation, kernel_constraint=MaxNorm(3)))
        model_n.add(layers.Dropout(dropout_probability))

    model_n.add(layers.Dense(1, activation='sigmoid'))

    # configure the model
    # use F1 score beause it balances between preciison and recall
    model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=[RecallAtPrecision_scorer])
    # model_n.compile(loss='binary_crossentropy', optimizer='adam', metrics=[Recall])
    return model_n

from scikeras.wrappers import KerasClassifier
# def compile_mlp(input_dim, H, num_epochs, num_layers, activation_function):
# number of hidden nodes
H = 10
# num of epochs
num_epochs = 50
# num_layers
num_layers = 3
# activation function
activation_function = 'relu'
#dropout probability
dropout_probability = 0.2
# input dim
input_dim = X_train_full.shape[1]

model = KerasClassifier(model=compile_mlp, input_dim=input_dim, H=H, num_epochs=num_epochs, num_layers=num_layers, activation=activation_function, dropout_probability=dropout_probability)

param_grid = {'activation':('relu', 'tanh', 'sigmoid'),
              'H': [50, 60, 70],
              'num_epochs': [75, 100],
              'num_layers': [8, 9, 10, 11, 12],
              'dropout_probability': [0.2, 0.3]}
seed = 42
tf.random.set_seed(42)

# # do grid search
# grid = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=2, scoring= 'f1')
# grid.fit(X_train_full, y_train_full)
# print(grid.best_params_)
# print(grid.best_score_)
#
# best_mlp = grid.best_estimator_
# print(best_mlp.score(X_test, y_test))

model_tuned = compile_mlp(X_train_full.shape[1], 60, 75, 5, 'relu', 0.2)
model_tuned.fit(X_train_full, y_train_full)

predicted_y = model_tuned.predict(X_test)
y_predicted = (predicted_y > 0.5).astype('int32')
print('Test Validation - Accuracy:', accuracy_score(y_test, y_predicted),
      'Preciation:', precision_score(y_test, y_predicted) ,
      'Recall:', recall_score(y_test, y_predicted),
      'F1:', f1_score(y_test,y_predicted),
      'ROC_AUC:', roc_auc_score(y_test,y_predicted))
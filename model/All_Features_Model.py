import pandas as pd
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras import layers,optimizers
from keras.constraints import MaxNorm
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from notebooks import *

#loading
df_file_path = "./data_exp/dataset-filtered-balance.csv"
df = pd.read_csv(df_file_path, low_memory=False)
print_dimensions(df)

#prepare data
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

#feature extraction
# #1.pca
# pca = PCA(n_components=50)
# X = pca.fit_transform(X)

# #2.mlp
# model = Sequential()
# model.add(layers.Dense(384, activation='relu'))
# model.add(layers.Dense(200, activation='relu'))
#
# # 训练
# model.compile(optimizer=optimizers.SGD(lr=0.5),
#              loss = 'sparse_categorical_crossentropy',
#              metrics=['accuracy'])
# model.fit(X_train_full, y_train_full, epochs=5,
#               batch_size=256,
#               validation_data=(X_test, y_test))
# X = model.predict(X)
# # print(output)
#
#
# # 3.AE
# input_size = 384
# hidden_size = 20
# output_size = 384
#
# x = Input(shape=(input_size,))
# h = Dense(hidden_size, activation='relu')(x)
# r = Dense(output_size, activation='sigmoid')(h)
#
# #自编码模型
# autoencoder = Model(inputs=x, outputs=r)
# autoencoder.compile(optimizer='adam', loss='mse')
#
# #输出模型结构
# print(autoencoder.summary())
#
# #训练模型
# epochs = 5
# batch_size = 128
#
# #Adam优化器和均方误差RMSE损失函数
# history = autoencoder.fit(X_train_full, X_train_full, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, X_test))
#
# #编码过程： 取出编码部分
# conv_encoder = Model(x, h)
#
# encoded_x = conv_encoder.predict(X_test)
# decoded_x = autoencoder.predict(X_test)
#
# print(encoded_x.shape,decoded_x.shape)

#classification
xgb_scale_pos_weight = compute_scale_pos_weight(y_binary)

def create_xgb_model():
    return XGBClassifier(n_jobs=10,
                         scale_pos_weight=xgb_scale_pos_weight,
                         n_estimators=25,
                         max_depth=3)

xgb_models = train_test_folds(X,
                              y_binary,
                              k_fold(X, n_splits=10),
                              create_xgb_model)

#feature importance
feature_importance = compute_average_feature_importance(X, xgb_models)
# display(create_feature_importance_table(feature_names, feature_importance, size=10))
plot_feature_importance(feature_names, feature_importance)
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from notebooks import *

#loading
honey_badger_labels = load_dictionary("./data_exp/honey_badger_labels.pickle")
df_file_path = "./data_exp/dataset-filtered-8.csv"
df = pd.read_csv(df_file_path, low_memory=False)

#prepare data
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

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

probas = np.zeros((10, y_binary.shape[0]))
for i, xgb_model in enumerate(xgb_models):
    probas[i, :] = xgb_model.predict_proba(X)[:, 1]

means = probas.mean(axis=0)
stds = probas.std(axis=0)
df["contract_label_name"] = df.contract_label_index.map(lambda index: honey_badger_labels["index_to_name"][index])

columns_out = [
    "contract_address",
    "contract_is_honeypot",
    "contract_label_index",
    "contract_label_name",
] + feature_names

df_out = df[columns_out]
df_out["predict_probability_mean"] = means
df_out["predict_probability_std"] = stds
df_out.sort_values(by="predict_probability_mean", ascending=False, inplace=True)
df_out.to_csv("./data_exp/probabilities.csv", index=False)

df_out_filtered = df_out[df_out["contract_is_honeypot"] == True]
df_out_filtered = df_out_filtered[[ "contract_address",
                                    "contract_is_honeypot",
                                    "contract_label_index",
                                    "contract_label_name"]]
df_out_filtered.to_csv("./data_exp/true_honeypot.csv", index=False)
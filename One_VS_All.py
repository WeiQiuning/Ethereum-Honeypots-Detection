import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use("ggplot")
from notebooks import *

#loading
honey_badger_labels = load_dictionary("./data_exp/honey_badger_labels.pickle")
df_file_path = "./data_exp/dataset-filtered-8.csv"
df = pd.read_csv(df_file_path, low_memory=False)
print_dimensions(df)

#prepare data
addresses, X, y_binary, y_multi, scaler, feature_names = extract_experiment_data(df)

#classification
xgb_scale_pos_weight = compute_scale_pos_weight(y_binary)

def create_xgb_model():
    return XGBClassifier(n_jobs=10,
                         scale_pos_weight=xgb_scale_pos_weight,
                         n_estimators=25,
                         max_depth=3)


num_labels = len(honey_badger_labels["index_to_name"])

for label_id, label_value in enumerate(honey_badger_labels["index_to_name"][1:], start=1):
    print("Label {:d}/{:d} = {}".format(label_id, num_labels, label_value))

    train_index = y_multi != label_id
    test_index = y_multi == label_id

    xgb_model = create_xgb_model()
    xgb_model.fit(X[train_index], y_binary[train_index])

    train_metrics = compute_metrics(y_binary[train_index], xgb_model.predict(X[train_index]))
    print_metrics("train", train_metrics)

    test_pred = xgb_model.predict(X[test_index])
    test_size = test_pred.shape[0]
    test_tp = test_pred.sum()
    test_fp = test_size - test_tp
    test_acc = test_tp / test_size
    print("test  Recall {:.03f} FN {:d} TP {:d}".format(test_acc, test_fp, test_tp))
    print()
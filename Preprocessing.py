import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from notebooks import *


def display_na_table(target):
    """
    Display a table with the count and the percentage of missing values per column.

    Arguments:
    target -- the pandas dataframe from which the frequencies are calculated
    """
    df_na = target.isna().sum()
    df_na = df_na[df_na > 0]
    df_na_t = pd.DataFrame([df_na, df_na / len(target) * 100]).transpose()
    df_na_t.columns = ["count", "percent"]
    display(df_na_t)


def na_row_matches_by_column(target, columns):
    """
    Take every pair of column and check if the rows with missing values match.

    Arguments:
    target -- the pandas dataframe from which the missing values are counted
    columns -- list of column names to match
    """
    non_matches = []
    for i, column_i in enumerate(columns):
        for j, column_j in enumerate(columns[i + 1:], start=1 + 1):
            if any(target[column_i].isna() != target[column_j].isna()):
                non_matches.append((column_i, column_j))

    if len(non_matches) > 0:
        print("Non matching columns found!")
        for column_i, column_j in non_matches:
            print(column_i, "and", column_j)
    else:
        print("All columns match.")


def add_ratio_column(target, numerator_column, denominator_column, ratio_column, fillna=False):
    """
    Adds a column resulting from the divition of two columns.

    Arguments:
    target -- the pandas dataframe to compute from and add to the new column
    numerator_column -- the column used as the numerator for the division
    denominator_column -- the column used as the denominator for the division
    ratio_column -- name of the new column to be added
    fillna -- True in case the missing values should be filled with zeros
    """
    target[ratio_column] = target[numerator_column] / target[denominator_column]

    if fillna:
        target[ratio_column] = target[ratio_column].fillna(0)


def categorical_counts(target_positive, target_negative, dictionary, column_prefix, limit=10, value_len_limit=30):
    """
    Display a table with the contract count per value of a categorical variable.

    Arguments:
    target_positive -- the pandas dataframe from which the positive counts are computed
    target_negative -- the pandas dataframe from which the negative counts are computed
    dictionary -- value definition of the categorical variable
    column_prefix -- column prefix of the categorical variable in the dataframes
    limit -- maximum amount of rows to show
    value_len_limit -- maximum amount of value characters to show
    """
    items = []
    for id_, value in enumerate(dictionary["index_to_name"]):
        column = column_prefix + str(id_)
        positive_count = target_positive[column].sum()
        items.append((id_, value, positive_count))

    items = sorted(items, key=lambda item: item[2], reverse=True)[:limit]

    ids, values, positive_counts = zip(*items)

    short_values = []
    for value in values:
        if len(value) > value_len_limit + 3:
            value = value[:(value_len_limit - 3)] + "..."
        short_values.append(value)

    total_counts = []
    negative_counts = []
    for id_, _, positive_count in items:
        column = column_prefix + str(id_)
        negative_count = target_negative[column].sum()
        negative_counts.append(negative_count)

        total_counts.append(positive_count + negative_count)

    df_counts = pd.DataFrame({
        "Value": short_values,
        "Honeypot": positive_counts,
        "Non-Honeypot": negative_counts,
        "Total": total_counts
    })

    display(HTML(df_counts.to_html(index=False)))

# loading all the dictionaries
honey_badger_labels = load_dictionary("honey_badger_labels.pickle")
contract_compiler_major_versions = load_dictionary("contract_compiler_major_versions.pickle")
contract_compiler_minor_versions = load_dictionary("contract_compiler_minor_versions.pickle")
contract_compiler_patch_versions = load_dictionary("contract_compiler_patch_versions.pickle")
contract_libraries = load_dictionary("contract_libraries.pickle")
fund_flow_cases = load_dictionary("fund_flow_cases.pickle")

# load dataset
df_file_path = "dataset.csv"
df = pd.read_csv(df_file_path, low_memory=False)

#Check that all contracts have creation transaction
has_creation = fund_flow_case_columns_accumulated_frequency(fund_flow_cases, df, creation=True) > 0
print(has_creation.value_counts())

df["contract_is_honeypot"] = (df.contract_label_index > 0)
print(df.contract_is_honeypot.value_counts())

# display distribution
honeypot_value_counts = df.contract_label_index[df.contract_is_honeypot].value_counts()
print(honeypot_value_counts)

honeypot_colors_by_frequency = plt.get_cmap("Set2").colors[:len(honeypot_value_counts)]

honeypot_color_by_label_index = {}
for (label_index, label_count), color in zip(honeypot_value_counts.iteritems(), honeypot_colors_by_frequency):
    honeypot_color_by_label_index[label_index] = color

# drop unused fund flow columns
non_fund_flow_columns = [column for column in df.columns
                         if not column.startswith("symbol_")]

print("Number of non fund flow case columns:", len(non_fund_flow_columns))
print("Number of fund flow case columns:", len(fund_flow_cases["index_to_name"]))

fund_flow_case_columns = []
for fund_flow_case_column in fund_flow_case_columns_with_fixed_values(fund_flow_cases):
    if df[fund_flow_case_column].sum() > 0:
        fund_flow_case_columns.append(fund_flow_case_column)
print("Number of used fund flow case columns:", len(fund_flow_case_columns))
print("Number of unused fund flow case columns:",
      len(fund_flow_cases["index_to_name"]) - len(fund_flow_case_columns))

df = df[non_fund_flow_columns + fund_flow_case_columns]
print("The dataset has {:d} rows and {:d} columns after removing unused fund flow case columns".format(*df.shape))

# statistics for numerical variables
df[non_fund_flow_columns].describe().transpose()

# transaction counts: validate that there are no contracts with zero normal transactions
has_one_normal_transaction = df.normal_transaction_count == 1
count_has_one_normal_transaction = len(df[has_one_normal_transaction])
print("Number of contracts with only one normal transaction: {:d} ({:.03f} %)".format(
    count_has_one_normal_transaction, count_has_one_normal_transaction * 100 / len(df)))

has_internal_transactions = df.internal_transaction_count > 0
has_no_internal_transactions = df.internal_transaction_count == 0
count_has_no_internal_transactions = len(df[has_no_internal_transactions])
print("Number of contracts with zero internal transactions: {:d} ({:.03f} %)".format(
    count_has_no_internal_transactions, count_has_no_internal_transactions * 100 / len(df)))

df[~df.contract_is_honeypot].normal_transaction_block_count.describe().apply("{:.2f}".format)
df[df.contract_is_honeypot].normal_transaction_block_count.describe().apply("{:.2f}".format)

block_span_per_label = df.groupby("contract_label_index").agg({"normal_transaction_first_block": "min",
                                                            "normal_transaction_last_block": "max"})
block_span_per_label["length"] = block_span_per_label["normal_transaction_last_block"] - block_span_per_label["normal_transaction_first_block"]
print(block_span_per_label)

honeypot_block_span_per_label = block_span_per_label[block_span_per_label.index > 0]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
fig.set_tight_layout(True)

plt.barh(range(len(honeypot_block_span_per_label)),
         honeypot_block_span_per_label["length"].values,
         left=honeypot_block_span_per_label["normal_transaction_first_block"].values,
         color=[honeypot_color_by_label_index[label_index] for label_index in honeypot_block_span_per_label.index],
         alpha=0.9)

plt.yticks(range(len(honeypot_block_span_per_label)),
           [honey_badger_labels["index_to_name"][label_index]
            for label_index in honeypot_block_span_per_label.index])

for label in ax.get_xticklabels():
    label.set_rotation(90)

plt.xlim(0, 6500000)

plt.title("Honeypot Technique Block Span")
plt.xlabel("Block")
plt.ylabel("Honeypot")
plt.show()

# missing value
source_code_na_columns = [
    "contract_compiler_runs",
    "contract_num_source_code_lines",
    "contract_compiler_major_version_index",
    "contract_compiler_minor_version_index",
    "contract_compiler_patch_index",
    "contract_library_index"
]
display_na_table(df)
display_na_table(df.loc[df.contract_has_source_code, source_code_na_columns])

# Value aggregated columns from normal transactions
columns_normal_value_aggregated = [
    "normal_transaction_value_mean",
    "normal_transaction_value_std"
]
na_row_matches_by_column(df, columns_normal_value_aggregated)
na_rows_normal_value_aggregated = df[columns_normal_value_aggregated[0]].isna()
all(fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                 df[na_rows_normal_value_aggregated],
                                                 error=True) > 0)
all(fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                 df[na_rows_normal_value_aggregated],
                                                 error=False) == 0)

# Delta aggregated columns from normal transactions
columns_normal_delta_aggregated = [
    "normal_transaction_time_delta_mean",
    "normal_transaction_time_delta_std",
    "normal_transaction_block_delta_mean",
    "normal_transaction_block_delta_std",
]
na_row_matches_by_column(df, columns_normal_delta_aggregated)
na_rows_normal_delta_aggregated = df[columns_normal_delta_aggregated[0]].isna()
all(has_one_normal_transaction == na_rows_normal_delta_aggregated)

# Aggregated columns from internal transactions
columns_internal_aggregated = [
    "internal_transaction_count_per_block_mean",
    "internal_transaction_count_per_block_std",
    "internal_transaction_gas_mean",
    "internal_transaction_gas_std",
    "internal_transaction_gas_used_mean",
    "internal_transaction_gas_used_std",
]
na_row_matches_by_column(df, columns_internal_aggregated)
na_rows_internal_aggregated = df[columns_internal_aggregated[0]].isna()
all(has_no_internal_transactions == na_rows_internal_aggregated)

internal_na_value_mean = df.internal_transaction_value_mean.isna() & has_internal_transactions
internal_na_value_std = df.internal_transaction_value_std.isna() & has_internal_transactions
all(internal_na_value_mean == internal_na_value_std)
all(fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                 df[internal_na_value_mean],
                                                 error=True) > 0)

# Filling missing values
df.loc[na_rows_normal_value_aggregated, columns_normal_value_aggregated] = \
    df.loc[na_rows_normal_value_aggregated, columns_normal_value_aggregated].fillna(0.0)
df.loc[na_rows_normal_delta_aggregated, columns_normal_delta_aggregated] = \
    df.loc[na_rows_normal_delta_aggregated, columns_normal_delta_aggregated].fillna(0.0)
df.loc[na_rows_internal_aggregated, columns_internal_aggregated] = \
    df.loc[na_rows_internal_aggregated, columns_internal_aggregated].fillna(0.0)
df.loc[:, "internal_transaction_value_mean"] = df.loc[:, "internal_transaction_value_mean"].fillna(0.0)
df.loc[:, "internal_transaction_value_std"] = df.loc[:, "internal_transaction_value_std"].fillna(0.0)

display_na_table(df[df.contract_has_source_code & df.contract_has_byte_code])

# normalize numeric
known_ignored_columns = [
    "contract_address",
    "contract_evaluation_positive",
    "contract_label_name",
    "contract_is_honeypot",
    "normal_transaction_first_block",
    "normal_transaction_last_block"
]

columns_to_normalize = set()

ignored_column_suffixes = ["_mean", "_std", "_index"]
ignored_column_infixes = ["_has_"]
for column in non_fund_flow_columns:
    ignore_column = False

    for known_ignored_column in known_ignored_columns:
        if column == known_ignored_column:
            ignore_column = True

    if not ignore_column:
        for ignored_column_suffix in ignored_column_suffixes:
            if column.endswith(ignored_column_suffix):
                ignore_column = True
                break

    if not ignore_column:
        for ignored_column_infix in ignored_column_infixes:
            if ignored_column_infix in column:
                ignore_column = True
                break

    if not ignore_column:
        columns_to_normalize.add(column)
        print(column)

add_ratio_column(df,
                 "normal_transaction_block_count",
                 "normal_transaction_count",
                 "normal_transaction_block_ratio",
                 fillna=False)

columns_to_normalize.remove("normal_transaction_block_count")

add_ratio_column(df,
                 "normal_transaction_before_creation_count",
                 "normal_transaction_count",
                 "normal_transaction_before_creation_ratio",
                 fillna=False)

columns_to_normalize.remove("normal_transaction_before_creation_count")

add_ratio_column(df,
                 "normal_transaction_from_other_count",
                 "normal_transaction_count",
                 "normal_transaction_from_other_ratio",
                 fillna=False)

columns_to_normalize.remove("normal_transaction_from_other_count")

add_ratio_column(df,
                 "normal_transaction_other_sender_count",
                 "normal_transaction_from_other_count",
                 "normal_transaction_other_sender_ratio",
                 fillna=True)  # when normal_transaction_from_other_count=0

columns_to_normalize.remove("normal_transaction_other_sender_count")

add_ratio_column(df,
                 "internal_transaction_block_count",
                 "internal_transaction_count",
                 "internal_transaction_block_ratio",
                 fillna=True)  # when internal_transaction_count=0

columns_to_normalize.remove("internal_transaction_block_count")

add_ratio_column(df,
                 "internal_transaction_creation_count",
                 "internal_transaction_count",
                 "internal_transaction_creation_ratio",
                 fillna=True)  # when internal_transaction_count=0

columns_to_normalize.remove("internal_transaction_creation_count")

add_ratio_column(df,
                 "internal_transaction_from_other_count",
                 "internal_transaction_count",
                 "internal_transaction_from_other_ratio",
                 fillna=True)  # when internal_transaction_count=0

columns_to_normalize.remove("internal_transaction_from_other_count")

add_ratio_column(df,
                 "internal_transaction_other_sender_count",
                 "internal_transaction_from_other_count",
                 "internal_transaction_other_sender_ratio",
                 fillna=True)  # when internal_transaction_from_other_count=0

columns_to_normalize.remove("internal_transaction_other_sender_count")

add_ratio_column(df,
                 "internal_transaction_to_other_count",
                 "internal_transaction_count",
                 "internal_transaction_to_other_ratio",
                 fillna=True)  # when internal_transaction_count=0

columns_to_normalize.remove("internal_transaction_to_other_count")

add_ratio_column(df,
                 "internal_transaction_other_receiver_count",
                 "internal_transaction_to_other_count",
                 "internal_transaction_other_receiver_ratio",
                 fillna=True)  # when internal_transaction_to_other_count=0

columns_to_normalize.remove("internal_transaction_other_receiver_count")

# One-hot-encoding the categorical variables
columns_to_one_hot_encode = [
    "contract_compiler_minor_version_index",
    "contract_compiler_patch_index",
    "contract_library_index"
]
columns_one_hot_encoded = []

for column in columns_to_one_hot_encode:
    dummies = pd.get_dummies(df[column], prefix=column.replace("_index", ""))
    columns_one_hot_encoded.extend(dummies.columns)
    df = pd.concat([df, dummies], axis=1)

column_rename = {}
for i, column in enumerate(columns_one_hot_encoded):
    new_column = column.replace(".0", "")
    columns_one_hot_encoded[i] = column
    column_rename[column] = new_column
df = df.rename(columns=column_rename)
del column_rename  # free some memory

# Categorical variables crossed with binary label
df_positive = df[df.contract_is_honeypot]
df_negative = df[~df.contract_is_honeypot]
pd.crosstab(~df.contract_compiler_minor_version_index.isna(),
            df.contract_is_honeypot,
            rownames=["contract_has_compiler_version_minor"])
categorical_counts(df_positive,
                   df_negative,
                   contract_compiler_minor_versions,
                   "contract_compiler_minor_version_")
pd.crosstab(~df.contract_compiler_patch_index.isna(),
            df.contract_is_honeypot,
            rownames=["contract_has_compiler_version_patch"])
categorical_counts(df_positive,
                   df_negative,
                   contract_compiler_patch_versions,
                   "contract_compiler_patch_",
                   limit=5)
categorical_counts(df_positive,
                   df_negative,
                   contract_libraries,
                   "contract_library_",
                   limit=5)

# Booleans crossed with the Binary Label
df["contract_has_library"] = ~df.contract_library_index.isna()
pd.crosstab(df.contract_has_library, df.contract_is_honeypot)
pd.crosstab(df.contract_has_byte_code, df.contract_is_honeypot)
pd.crosstab(df.contract_has_source_code, df.contract_is_honeypot)
df["has_internal_transactions"] = df["internal_transaction_count"] > 0
pd.crosstab(df.has_internal_transactions, df.contract_is_honeypot)

# Filtering
df_filtered = df[df.contract_has_byte_code & df.contract_has_source_code]
print_dimensions(df_filtered)
df_filtered.contract_is_honeypot.value_counts()

df_filtered = df_filtered.drop([
    # booleans that already filtered the dataset
    "contract_has_source_code",
    "contract_has_byte_code",

    # counts transformed into ratios
    "normal_transaction_block_count",
    "normal_transaction_before_creation_count",
    "normal_transaction_other_sender_count",
    "normal_transaction_from_other_count",
    "internal_transaction_block_count",
    "internal_transaction_creation_count",
    "internal_transaction_from_other_count",
    "internal_transaction_other_sender_count",
    "internal_transaction_to_other_count",
    "internal_transaction_other_receiver_count",

    # label encoded features replaced by one hot encoded features
    "contract_compiler_major_version_index",  # this was discarded for having no variance
    "contract_compiler_minor_version_index",
    "contract_compiler_patch_index",
    "contract_library_index",

    # not useful for machine learning, only useful for plots
    "normal_transaction_first_block",  # this was just for reference but not an acutal feature
    "normal_transaction_last_block",  # this was just for reference but not an acutal feature
], axis=1)

for column in sorted(df_filtered.columns):
    if not column.startswith("contract_compiler_minor_version_") \
        and not column.startswith("contract_compiler_patch_") \
        and not column.startswith("contract_library_") \
        and not column.startswith("symbol_"):
        print(column)

columns_ignore_variance = set([
    "contract_address",
    "contract_evaluation_positive",
    "contract_label_index",
    "contract_label_name",
])

columns_zero_variance = []
for column in df_filtered.columns:
    if column not in columns_ignore_variance:
        if df_filtered[column].var() < 1e-6:
            columns_zero_variance.append(column)
            print(column)

df_filtered = df_filtered.drop(columns_zero_variance, axis=1)


library_columns = []
for column in df_filtered.columns:
    if column.startswith("contract_library_"):
        library_columns.append(column)
        print(column)
df_filtered = df_filtered.drop(library_columns, axis=1)

df_filtered = df_filtered.drop([
    "internal_transaction_block_ratio",
    "internal_transaction_count_per_block_mean",
    "internal_transaction_count_per_block_std",
    "internal_transaction_creation_ratio",
    "internal_transaction_from_other_ratio",
    "internal_transaction_gas_mean",
    "internal_transaction_gas_std",
    "internal_transaction_gas_used_mean",
    "internal_transaction_gas_used_std",
    "internal_transaction_other_receiver_ratio",
    "internal_transaction_other_sender_ratio",
    "internal_transaction_to_other_ratio",
    "internal_transaction_value_mean",
    "internal_transaction_value_std",
], axis=1)

#saving
df_filtered.to_csv("../data_exp/dataset-filtered-8.csv", index=False)
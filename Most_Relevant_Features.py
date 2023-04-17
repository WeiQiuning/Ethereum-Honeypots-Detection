import pandas as pd
from notebooks import *


def distribution_table(target, column):
    """
    Creates a dataframe with the distribution of a column separated by honeypot and non-honeypots.

    Arguments:
    target -- the pandas dataframe to calculate the distribution
    column -- the column name to calculate the distribution
    """
    desc_positive = target[target.contract_is_honeypot][column].describe().apply("{:.2f}".format)
    desc_negative = target[~target.contract_is_honeypot][column].describe().apply("{:.2f}".format)
    return pd.DataFrame({"Yes": desc_positive, "No": desc_negative})

# loading
honey_badger_labels = load_dictionary("../data_exp/honey_badger_labels.pickle")
fund_flow_cases = load_dictionary("../data_exp/fund_flow_cases.pickle")
df_file_path = "../data_exp/dataset-filtered-8.csv"
df = pd.read_csv(df_file_path, low_memory=False)


# The number of source code lines seems to have a lower upper bound for honeypots
distribution_table(df, "contract_num_source_code_lines")
# Cross handcrafted rule with binary label
pd.crosstab((15 < df.contract_num_source_code_lines) & (df.contract_num_source_code_lines < 200),
            df.contract_is_honeypot,
            rownames=["15 < contract_num_source_code_lines < 200"])
# Extreme cases
df.loc[
    df.contract_num_source_code_lines > 10000,
    ["contract_address", "contract_num_source_code_lines"]
].sort_values("contract_num_source_code_lines", ascending=False)

# Number of normal transactions
distribution_table(df, "normal_transaction_count")
pd.crosstab(df.normal_transaction_count < 40,
            df.contract_is_honeypot,
            rownames=["normal_transaction_count < 40"])
df.loc[
    df.normal_transaction_count > 1000000,
    ["contract_address", "normal_transaction_count"]
].sort_values("normal_transaction_count", ascending=False)

# Contracts with deposits from others
weis_1_ether = 1000000000000000000
df["normal_transaction_value_mean_ether"] = df.normal_transaction_value_mean / weis_1_ether
distribution_table(df, "normal_transaction_value_mean_ether")

pd.crosstab(df.normal_transaction_value_mean_ether < 2,
            df.contract_is_honeypot,
            rownames=["normal_transaction_value_mean_ether < 2"])
df.loc[
    df.normal_transaction_value_mean_ether > 20000,
    ["contract_address", "normal_transaction_value_mean_ether"]
].sort_values("normal_transaction_value_mean_ether", ascending=False)

# Contracts with deposits from others
deposit_other_frequency = fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                                       df,
                                                                       sender="other",
                                                                       error=False,
                                                                       balance_sender="negative",
                                                                       balance_contract="positive")
pd.crosstab(df.contract_is_honeypot,
            deposit_other_frequency > 0,
            colnames=["deposit_other_frequency > 0"])

# Contracts with withdraws from others
withdraw_other_frequency = fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                                        df,
                                                                        sender="other",
                                                                        error=False,
                                                                        balance_sender="positive",
                                                                        balance_contract="negative")
pd.crosstab(df.contract_is_honeypot,
            withdraw_other_frequency > 0,
            colnames=["withdraw_other_frequency > 0"])

# Contracts with deposits from the creator
deposit_creator_frequency = fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                                         df,
                                                                         sender="creator",
                                                                         error=False,
                                                                         balance_creator="negative",
                                                                         balance_contract="positive")
pd.crosstab(df.contract_is_honeypot,
            deposit_creator_frequency > 0,
            colnames=["deposit_creator_frequency > 0"])

# Contracts with withdraws from the creator
withdraw_creator_frequency = fund_flow_case_columns_accumulated_frequency(fund_flow_cases,
                                                                          df,
                                                                          sender="creator",
                                                                          error=False,
                                                                          balance_creator="positive",
                                                                          balance_contract="negative")
pd.crosstab(df.contract_is_honeypot,
            withdraw_creator_frequency > 0,
            colnames=["withdraw_creator_frequency > 0"])

#######################################################################
#                TeCo Churn Problem
#######################################################################

## Data Story

"""
This dataset contains information about a fictional telecommunications company named TeCo, which provides home phone
 and internet services to 7043 customers in California.
It indicates which customers have left the service, stayed, or signed up for the service.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Features / Variables:
"""
CustomerId: Unique ID number of a Customer
Gender: Gender of a Customer
SeniorCitizen: Whether the customer is a senior citizen (1, 0)
Partner: Whether the customer has a partner (Yes, No)
Dependents: Whether the customer has dependents (Yes, No)
Tenure: The number of months the customer has stayed with the company
PhoneService: Whether the customer has phone service (Yes, No)
MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
InternetService: The customer's internet service provider (DSL, Fiber optic, No)
OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
TechSupport: Whether the customer has technical support (Yes, No, No internet service)
StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
Contract: The contract duration of the customer (Month-to-month, One year, Two year)
PaperlessBilling: Whether the customer has paperless billing (Yes, No)
PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
MonthlyCharges: The monthly amount charged from the customer
TotalCharges: The total amount charged from the customer
Churn: Whether the customer has churned (Yes or No)
"""


#######################################################
##########   STEP 1: STUDYING THE DATA     ############

df = pd.read_csv("datasets/TeCoCustomerChurn.csv")
df.info()

# Out[1] df.info()
# RangeIndex: 7043 entries, 0 to 7042
# Data columns (total 21 columns):
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   customerID        7043 non-null   object
#  1   gender            7043 non-null   object
#  2   SeniorCitizen     7043 non-null   int64
#  3   Partner           7043 non-null   object
#  4   Dependents        7043 non-null   object
#  5   tenure            7043 non-null   int64
#  6   PhoneService      7043 non-null   object
#  7   MultipleLines     7043 non-null   object
#  8   InternetService   7043 non-null   object
#  9   OnlineSecurity    7043 non-null   object
#  10  OnlineBackup      7043 non-null   object
#  11  DeviceProtection  7043 non-null   object
#  12  TechSupport       7043 non-null   object
#  13  StreamingTV       7043 non-null   object
#  14  StreamingMovies   7043 non-null   object
#  15  Contract          7043 non-null   object
#  16  PaperlessBilling  7043 non-null   object
#  17  PaymentMethod     7043 non-null   object
#  18  MonthlyCharges    7043 non-null   float64
#  19  TotalCharges      7043 non-null   object  //// OOppsss !!! Payment can not be "object"
#  20  Churn             7043 non-null   object


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)


### EXPLORATORY DATA ANALYSIS (EDA) :

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Out[2]
# Observations: 7043
# Variables: 21
# cat_cols: 17
# num_cols: 3
# cat_but_car: 1
# num_but_cat: 2


### CORRELATION


df[num_cols].corr()

# Out[3]:
#                 tenure  MonthlyCharges  TotalCharges
# tenure           1.000           0.248         0.826
# MonthlyCharges   0.248           1.000         0.651
# TotalCharges     0.826           0.651         1.000


# Correlation Matrix
df, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

"""
It is seen that "tenure" and "TotalCharges" correlated positively.
"""


# Analyzing the missing values
##############################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

# Out[4]               n_miss  ratio
# TotalCharges             11  0.160


# There are 11 missing values in the Total Charges variable. Let's examine the "Total Charges"
# and fill in the blanks accordingly.

df["TotalCharges"].describe()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


#######################################################
##########       STEP 2: BASE MODEL          ##########

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12)),
          ('RF', RandomForestClassifier(random_state=12)),
          ('XGB', XGBClassifier(random_state=12)),
          ("LightGBM", LGBMClassifier(random_state=12)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

# Out[5]
# ########## LR ##########
# Accuracy: 0.8051
# Auc: 0.843
# Recall: 0.5479
# Precision: 0.6599
# F1: 0.5985
# ########## KNN ##########
# Accuracy: 0.7619
# Auc: 0.7438
# Recall: 0.4446
# Precision: 0.5654
# F1: 0.4976
# ########## CART ##########
# Accuracy: 0.7263
# Auc: 0.6524
# Recall: 0.4933
# Precision: 0.4847
# F1: 0.4889
# ########## RF ##########
# Accuracy: 0.7894
# Auc: 0.8233
# Recall: 0.4815
# Precision: 0.6374
# F1: 0.5485
# ########## XGB ##########
# Accuracy: 0.7843
# Auc: 0.8227
# Recall: 0.5099
# Precision: 0.6132
# F1: 0.5565
# ########## LightGBM ##########
# Accuracy: 0.7933
# Auc: 0.8338
# Recall: 0.5201
# Precision: 0.6366
# F1: 0.572
# ########## CatBoost ##########
# Accuracy: 0.7988
# Auc: 0.84
# Recall: 0.511
# Precision: 0.6556
# F1: 0.5742


#######################################################
##########   STEP 3: FEATURE ENGINEERING     ##########

#   Analyzing the Outliers
##########################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Out[5]
# tenure False
# MonthlyCharges False
# TotalCharges False

## So, there is NO outlier.....

# CREATING THE NEW FEATURES
###########################

# 1: TENURE
df["tenure"].describe() # min:0 ve max:72 months

df.loc[(df["tenure"] > 0) & (df["tenure"] <= 12), "NEW_TENURE"] = "1-year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE"] = "2-year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE"] = "3-year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE"] = "4-year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE"] = "5-year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE"] = "6-year"

# 2: CONTRACT

# df.loc[(df["Contract"] =="One year") | (df["Contract"] == "Two year"), "NEW_CONTRACT_LENGTH"] = "yearly"
df["NEW_CONTRACT_LENGTH"] = df["Contract"].apply(lambda x: "yearly" if x in ["One year", "Two year"] else "monthly")

# 3: PAYMENT METHOD
df["NEW_PAYMENT_METHOD"] = df["PaymentMethod"].apply(lambda x: "elect" if x in ['Electronic check',
                                                                                'Bank transfer (automatic)',
                                                                                "Credit card (automatic)"] else "no_elect")
# 4: PROTECTION

df["NEW_BACKUP_PROTECTION_SUPPORT"] = df.apply(lambda x: 1 if (x["OnlineBackup"] =="Yes") or
                                                              (x["DeviceProtection"] == "Yes") or
                                                              (x["TechSupport"] == "Yes") else 0, axis=1)

# 5: CHARGES
df["NEW_AVG_CHARGES"] = df["TotalCharges"] / (df["tenure"] + 1)

# 6: INCREASE

df["NEW_INCREASE"] = df["NEW_AVG_CHARGES"] / df["MonthlyCharges"]

# 7: TOTAL SERVICE PURCHASED

df['NEW_TOTALSERVICE'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# 8: ANY STREAMING

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)


#######################            ENCODING      #############################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Out[6]
# Observations: 7043
# Variables: 29
# cat_cols: 23
# num_cols: 5
# cat_but_car: 1
# num_but_cat: 5

# Label Encoding
#############################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One Hot Encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TOTALSERVICES"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.info()
#############################################################################################################

##################################         STEP 4: FINAL MODEL         ######################################

y = df["Churn_1"]
X = df.drop(["Churn_1","customerID"], axis=1)


models = [('LR', LogisticRegression(random_state=12)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12)),
          ('RF', RandomForestClassifier(random_state=12)),
          ('XGB', XGBClassifier(random_state=12)),
          ("LightGBM", LGBMClassifier(random_state=12)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
    
# Out[7]
# ########## LR ##########
# Accuracy: 0.8042
# Auc: 0.8458
# Recall: 0.5275
# Precision: 0.6668
# F1: 0.588
# ########## KNN ##########
# Accuracy: 0.7613
# Auc: 0.7444
# Recall: 0.4419
# Precision: 0.5645
# F1: 0.4955
# ########## CART ##########
# Accuracy: 0.7251
# Auc: 0.6514
# Recall: 0.4896
# Precision: 0.4822
# F1: 0.4859
# ########## RF ##########
# Accuracy: 0.7889
# Auc: 0.826
# Recall: 0.489
# Precision: 0.6321
# F1: 0.5514
# ########## XGB ##########
# Accuracy: 0.7903
# Auc: 0.825
# Recall: 0.5142
# Precision: 0.6292
# F1: 0.5657
# ########## LightGBM ##########
# Accuracy: 0.7931
# Auc: 0.8355
# Recall: 0.5265
# Precision: 0.6326
# F1: 0.5746
# ########## CatBoost ##########
# Accuracy: 0.7965
# Auc: 0.8419
# Recall: 0.5147
# Precision: 0.6468
# F1: 0.5732

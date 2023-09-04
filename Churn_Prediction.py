#######################################################################
#                TeCo Churn Problem
#######################################################################

## Data Story

"""
This dataset contains information about a fictional telecommunications company named TeCo, which provides home phone
 and internet services to 7043 customers in New York.
It indicates which customers have left the service, stayed, or signed up for the service.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
# !pip install missingno

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

df = pd.read_csv("3_Feature_Engineering/HW_Feature_Engineering/Part_I/TelcoCustomerChurn-230423-212029.csv")
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

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
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

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

# Out[2] tenure False
# MonthlyCharges False
# TotalCharges False

# Analyzing the missing values
###################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

# Out[3]               n_miss  ratio
# TotalCharges             11  0.160


# TotalCharges değişkeninde 11 adet boşluk var. cinsiyet dağılına göre ortalamaya bakalım ve buna göre boş değerleri
# dolduralım

df.groupby("gender").agg({"TotalCharges": ["mean", "median"]})
df["TotalCharges"].describe()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# boş değerler ort ile dolduruldu


#########  aykırı değer incelemesi

# check_outlier(df, )

for col in num_cols:
    print(col, check_outlier(df, col))

# tenure False
# MonthlyCharges False
# TotalCharges False

# aykırı değer yoktur


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

    return

missing_vs_target(df, "TotalCharges", na_cols)
df.info()

#######################################################
####### Adım 2: Yeni değişkenler oluşturunuz  #########



def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TotalCharges", cat_cols)


# 1- genel inceleme için
for col in df.columns:
    print(df[col].describe())
    print("########################")
    print("########################", end="\n\n\n")

# 2- unique değerleri inceleme için
for col in df.columns:
    print({col: df[col].unique()})
    print("########################")
    print("########################", end="\n\n\n")


# üstteki 2 incelemeye göre yeni değişkenler üreteceğiz

# 1.YENİ: TENURE
df["tenure"].describe() # min:0 ve max:72 ay
df.head()
df.groupby(["SeniorCitizen", "PaymentMethod"]).agg({"TotalCharges": ["mean" ,"count"]})
df.groupby(["SeniorCitizen", "PaymentMethod"]).agg({"TotalCharges": ["mean" ,"count"], "tenure": ["mean" ,"count"]})
df.groupby(["SeniorCitizen", "PaymentMethod"]).agg({"MonthlyCharges": ["sum" , "mean", "count"]})
df.groupby(["SeniorCitizen", "PaperlessBilling", "PaymentMethod"]).agg({"TotalCharges": ["mean" , "sum", "count"]})

df.loc[(df["tenure"] > 0) & (df["tenure"] <= 12), "NEW_TENURE"] = "1-year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE"] = "2-year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE"] = "3-year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE"] = "4-year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE"] = "5-year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE"] = "6-year"

# 2. YENİ: CONTRACT

# df.loc[(df["Contract"] =="One year") | (df["Contract"] == "Two year"), "NEW_CONTRACT_LENGTH"] = "yearly"
df["NEW_CONTRACT_LENGTH"] = df["Contract"].apply(lambda x: "yearly" if x in ["One year", "Two year"] else "monthly")
df["NEW_PAYMENT_METHOD"] = df["PaymentMethod"].apply(lambda x: "elect" if x in ['Electronic check',
                                                                                'Bank transfer (automatic)',
                                                                                "Credit card (automatic)"] else "no_elect")


#######################################################################################################################

#######################      Adım 3: Encoding işlemlerini gerçekleştiriniz.       #####################################


# Label Encoding
#############################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# One Hot Encoding

OHE =[col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

one_hot_encoder(df, OHE, drop_first=True)

df.head()


# StandardScaler:
###################

for col in num_cols:
    ss = StandardScaler()
    df[col + "_standard_scaler"] = ss.fit_transform(df[[col]])
df.head()

# RobustScaler: Medyanı çıkar iqr'a böl.
###################

for col in num_cols:
    rs = RobustScaler()
    df[col + "_robust_scaler"] = rs.fit_transform(df[[col]])

df.head()

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

for col in num_cols:
    mms = MinMaxScaler()
    df[col + "_min_max_scaler"] = mms.fit_transform(df[[col]])
    df.describe().T

df.head()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# 8. Model
#############################################

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

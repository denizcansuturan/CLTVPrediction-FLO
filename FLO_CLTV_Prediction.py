##############################################################
# CLTV Prediction BG-NBD and Gamma-Gamma
##############################################################

###############################################################
# Business Problem
###############################################################

# FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make a long-term plan,
# it is necessary to estimate the potential value that existing customers will provide to the company in the future.

###############################################################
# Dataset Story
###############################################################

# Dataset includes information obtained from the past shopping behavior of OmniChannel
# (both online and offline shoppers) customers whose last purchases was in 2020–2021.

# master_id: Unique customer ID
# order_channel : Code for the platform that is used for the purchase (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : Code for the platform that is used for the last purchase
# first_order_date : Date of the first purchase
# last_order_date : Date of the last purchase
# last_order_date_online : Date of the last online purchase
# last_order_date_offline : Date of the last offline purchase
# order_num_total_ever_online : Total number of online purchases
# order_num_total_ever_offline : Total number of offline purchases
# customer_value_total_ever_offline : Total money spent on offline purchases
# customer_value_total_ever_online : Total money spent on online purchases
# interested_in_categories_12 : Category list that the customer purchased from in the last 12 months


###############################################################
# TASKS
###############################################################
# TASK 1: Data Understanding and Preparation

# 1. Read dataset flo_data_20K.csv. Make a copy of the dataframe.
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None) # to see all the columns
df_ = pd.read_csv("D:/MIUUL/CRM/CASE STUDY 2/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

# 2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.
# Note: When calculating cltv, frequency values must be integers.
# Therefore, round the lower and upper limits with round().


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    # quartile values are calculated
    interquantile_range = quartile3 - quartile1
    # difference of quartile values is calculated
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit
# this function determines thresholds for the variable.


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. If there are outliers in variables "order_num_total_ever_online","order_num_total_ever_offline",
# "customer_value_total_ever_offline","customer_value_total_ever_online, suppress them.


df.describe().T
# 75% and max values are highly different,
# outliers in all the columns are supressed

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
# outliers are replaced with the limits

df.describe().T
# check the new version

# 4.Customers are both online and offline shoppers.
# # Create new variables for each customer's total number of purchase and total spending

# a. Customer's total number of purchase:
df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# b.Customer's total spending:
df["total_spending"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 5. Check variable types. Convert date variables type into date type.

df.dtypes
df['first_order_date'] = pd.to_datetime(df['first_order_date'])
df['last_order_date'] = pd.to_datetime(df['last_order_date'])
df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])


# TASK 2: Preparing Lifetime Data Structure

# 1. Take 2 days after the date of the last purchase in the data set as the date of analysis.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

# 2. Create new cltv dataframe including customer_id, recency_cltv_weekly,
# T_weekly, frequency ve monetary_cltv_avg variables

# recency: Time between the first and the last purchase of a customer in weeks
# T: Age of the customer i.e. time after the first purchase until today in weeks
# frequency: Total number of transactions (frequency>1): (retention)
# monetary: Average monetary per order


df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"])/7
df["recency_cltv_weekly"] = df["recency_cltv_weekly"].astype('timedelta64[D]').astype(int)

df["T_weekly"] = (today_date - df["first_order_date"])/7
df["T_weekly"] = df["T_weekly"].astype('timedelta64[D]').astype(int)

df["frequency"] = df["total_order_number"]
df["monetary_cltv_avg"] = df["total_spending"] / df["frequency"]

cltv_df = df[["master_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]]

cltv_df["master_id"].nunique()
cltv_df.shape[0]
# to check if the customers info is repeated or not
# customers ids are unique no need to group by

# TASK 3: Setting up BG-NBD Model and Gamma-Gamma Models, Calculation of CLTV
# 1. Fit BG/NBD model.

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# a.Estimate expected purchases from customers in 3 months and add exp_sales_3_month to cltv dataframe.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# b. Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# 2. Fit Gamma-Gamma model.
# Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])


cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])


# a. Calculate 6 months of CLTV and add it to the dataframe with the name cltv.

# 2 models are merged
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # in months by definition of the function
                                   freq="W",  # frequency info (in weeks)
                                   discount_rate=0.01) # if there is a discount
cltv_df["cltv"] = cltv


# b. Observe the 20 people with the highest Cltv value.
cltv_df.sort_values(by="cltv", ascending=False).head(10)


# TASK 4: Segmentation according to CLTV

# 1. Divide all your customers into 4 groups (segments) according to 6-month CLTV
# and add the group names to the dataset.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])


# 2. Make 6-month action suggestions to the management for 2 groups that you will choose from among 4 groups.

cltv_6.groupby("cltv_segment").agg(
    {"count", "mean", "sum"})

# B and C segments' clv means are similar to each other
# They can be merged and treated as one segment.
# Total of expected sales in 6 months is highest in A segment
# these customers can be focused mainly.


# BONUS: Functionalization

def create_cltv_p(df, month=3):

    replace_with_thresholds(df, "order_num_total_ever_online")
    replace_with_thresholds(df, "order_num_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_offline")
    replace_with_thresholds(df, "customer_value_total_ever_online")

    df["total_order_number"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_spending"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

    df['first_order_date'] = pd.to_datetime(df['first_order_date'])
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])
    df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

    today_date = dt.datetime(2021, 6, 1)

    df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]) / 7
    df["recency_cltv_weekly"] = df["recency_cltv_weekly"].astype('timedelta64[D]').astype(int)

    df["T_weekly"] = (today_date - df["first_order_date"]) / 7
    df["T_weekly"] = df["T_weekly"].astype('timedelta64[D]').astype(int)

    df["frequency"] = df["total_order_number"]
    df["monetary_cltv_avg"] = df["total_spending"] / df["frequency"]

    cltv_df = df[["master_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=month,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df


cltv_6 = create_cltv_p(df, 6)

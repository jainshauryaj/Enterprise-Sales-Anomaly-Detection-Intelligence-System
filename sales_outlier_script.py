###################################################################################################################
# Author = Shaurya Jain
###################################################################################################################

# Import Library
import pencilbox as pb
import pandas as pd
import numpy as np
import time
import os
from datetime import date, datetime, timedelta
import json
import shutil
import sys
import subprocess
import psutil

#!pip install pymysql
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pymysql'])
import pymysql
#!pip install pandasql
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandasql'])
import pandasql as ps

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer

import boto3
import io
from tqdm import tqdm

#!pip install matplotlib
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
import matplotlib.pyplot as plt
import seaborn as sns


import requests
from requests.exceptions import HTTPError

from tqdm.notebook import tqdm

CON_REDSHIFT = pb.get_connection("[Warehouse] Redshift")
CON_TRINO = pb.get_connection("[Warehouse] Trino")

def read_sql_query(sql, con):
    max_tries = 3
    for attempt in range(max_tries):
        print(f"Read attempt: {attempt}...")
        try:
            start = time.time()
            df = pd.read_sql_query(sql, con)
            end = time.time()
            if (end - start) > 60:
                print("Time: ", (end - start) / 60, "min")
            else:
                print("Time: ", end - start, "s")
            return df
            break
        except BaseException as e:
            print(e)
            time.sleep(5)
            

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 300)
import warnings
warnings.filterwarnings('ignore')

import time
start_time = time.time()

outliers_fraction = 0.01
final_df = pd.DataFrame()

city_df = pd.read_csv('cities.csv')
city_df = city_df[city_df['city_name'] != 'Ahmedabad']
#print(city_df.head())

for city in city_df['city_name']:
    print(city)
    order_query = f"""
    with
    item_details as
        (select item_id, 
            (id.name || ' ' || id.variant_description) as item_name
                from lake_rpc.item_details id
        ),

    item_mapping as
        (select distinct ipr.product_id,
            case when ipr.item_id is null then ipom_0.item_id else ipr.item_id end as item_id,
            case when ipr.item_id is not null then COALESCE(ipom.multiplier,1) else COALESCE(ipom_0.multiplier,1) end as multiplier

                from lake_rpc.item_product_mapping ipr

                    left join
                        dwh.dim_item_product_offer_mapping ipom on ipom.product_id = ipr.product_id
                            and ipr.item_id = ipom.item_id
                    left join
                        dwh.dim_item_product_offer_mapping ipom_0 on ipom_0.product_id = ipr.product_id
        ),

    sales as
        (select 
            (oid.cart_checkout_ts_ist) as order_date,
            cl.name as city_name,
            rco.facility_id,
            oid.product_id,
            im.item_id,
            oid.cart_id,
            oid.dim_customer_key,
            ((unit_selling_price * 1.00)/im.multiplier) as item_selling_price,
            ((unit_mrp * 1.00)/im.multiplier) as item_selling_mrp,
            im.multiplier,
            oid.total_doorstep_return_quantity,
            ((oid.procured_quantity - oid.total_doorstep_return_quantity) * im.multiplier) as sales_quantity,
            (sales_quantity * item_selling_price) as sales_value,
            (sales_quantity * item_selling_mrp) as sales_value_mrp

                from dwh.fact_sales_order_item_details oid

                    join item_mapping im on im.product_id = oid.product_id

                    join lake_retail.console_outlet rco on rco.id = oid.outlet_id and business_type_id in (7)
                    join lake_retail.console_location cl on cl.id = rco.tax_location_id

                        where (oid.cart_checkout_ts_ist between (current_date - 90 || ' 00:00:00')::timestamp and (current_date || ' 23:59:59')::timestamp)
                            and oid.is_internal_order = false
                            and (oid.order_type not ilike '%%internal%%' or oid.order_type is null)
                            and oid.procured_quantity > 0
                            and oid.order_current_status = 'DELIVERED'
        ),

    discard_carts as 
        (
            SELECT
                DISTINCT facility_id,
                item_id,
                1 as remove_flag
            FROM
            (SELECT
                EXTRACT(MONTH FROM oid.cart_checkout_ts_ist) AS month,
                rco.facility_id,
                im.item_id,
                COUNT(DISTINCT oid.cart_id) as total_carts
            FROM
                dwh.fact_sales_order_item_details oid
            JOIN
                item_mapping im ON im.product_id = oid.product_id
            JOIN lake_retail.console_outlet rco ON rco.id = oid.outlet_id and business_type_id in (7)
            JOIN lake_retail.console_location cl ON cl.id = rco.tax_location_id
            WHERE
                (oid.cart_checkout_ts_ist between (current_date - 90 || ' 00:00:00')::timestamp and (current_date || ' 23:59:59')::timestamp)
                AND oid.is_internal_order = false
                AND (oid.order_type not ilike '%%internal%%' or oid.order_type is null)
                AND oid.procured_quantity > 0 
                AND oid.order_current_status = 'DELIVERED'
            GROUP BY 1,2,3)
            WHERE total_carts < 3
    ),

    cart_data AS
      (
        SELECT 
            DISTINCT cart_id,
            sum(sales_quantity) AS order_quantity
        FROM 
            sales
        GROUP BY 1
    ),

    final_sales as
        (select 
            date(order_date) as date_,
            city_name,
            facility_id,
            s.item_id, 
            item_name, 
            sum(sales_quantity)::int as sales_quantity,
            sum(sales_value) as sales_value,
            sum(sales_value_mrp) as sales_value_mrp,
            avg(cast(sales_quantity AS float)/cast(order_quantity AS float)) AS ipc

                from sales s

                    join
                        item_details id on id.item_id = s.item_id
                    JOIN
            cart_data c ON s.cart_id = c.cart_id

                        where sales_quantity > 0

                            group by 1,2,3,4,5
        )

            SELECT fs.*, dc.remove_flag
            FROM final_sales fs 
            LEFT JOIN discard_carts dc ON fs.facility_id = dc.facility_id AND fs.item_id = dc.item_id
            WHERE city_name IN ('{city}') and fs.item_id IN (10000490)
    """
    order_data = read_sql_query(order_query, CON_REDSHIFT)
    print(order_data.head())

    start_time_aq = time.time()

    error_df = order_data[order_data['remove_flag'] == 1]
    error_df['min'] = 1.00
    error_df['max'] = 1.00
    error_df['error_reason'] = 'Less Than 2 carts per month'
    error_df = error_df[['facility_id','item_id','min','max','error_reason']].drop_duplicates().reset_index(drop = True)
    error_df.head()

    order_df = order_data[order_data['remove_flag'] != 1]
    order_df = order_data.drop(columns=['remove_flag'])
    order_df['Sales'] = order_df['sales_quantity']
    order_df = order_df.dropna()
    order_df['date_'] = pd.to_datetime(order_df['date_'])

    print('Total Unique Stores', len(list(set(order_df['facility_id']))))
    store_count = 0
    item_count = 0
    list_facility_id = list(set(order_df['facility_id']))

    for i in tqdm(range(0, len(list_facility_id)), initial = 0, desc ="Stores Processing"):
        facility_id = list_facility_id[i]
        x_train = order_df[order_df['facility_id'] == facility_id].reset_index().drop(columns = {'index'})

        list_item_id = list(set(x_train['item_id']))
        print('Total Unique Items on this Store', len(list_item_id))
        for j in tqdm(range(0, len(list_item_id)), initial = 0, desc ="Items Processing"):
            item_id = list_item_id[j]
            box_x_train = x_train[x_train['item_id'] == item_id].reset_index().drop(columns = {'index'})

            # Boxcox Transformation
            try:
                box_x_train['Sales'] = stats.boxcox(box_x_train['Sales'])[0]
            except:
                min_fval = box_x_train['sales_quantity'].min()
                max_fval = box_x_train['sales_quantity'].max()
                error_df = error_df.append({'facility_id': int(facility_id) ,'item_id': int(item_id), 'min': round(min_fval, 2), 'max': round(max_fval, 2), 'error_reason': 'Breaking at Boxcox Transformation'}, ignore_index = True)
                pass

            # #Imputation
            # try:
            #     q1 = box_x_train['Sales'].quantile(0.25)
            #     q3 = box_x_train['Sales'].quantile(0.75)
            #     iqr = q3-q1
            #     Lower_tail = q1 - 1.5 * iqr
            #     Upper_tail = q3 + 1.5 * iqr
            #     med = np.median(box_x_train['Sales'])
            #     for i in box_x_train['Sales']:
            #         if i > Upper_tail or i < Lower_tail:
            #                 box_x_train['Sales'] = box_x_train['Sales'].replace(i, med)
            # except:
            #     min_fval = box_x_train['sales_quantity'].min()
            #     max_fval = box_x_train['sales_quantity'].max()
            #     error_df = error_df.append({'facility_id': int(facility_id) ,'item_id': int(item_id), 'min': round(min_fval, 2), 'max': round(max_fval, 2), 'error_reason': 'Breaking at Imputation'}, ignore_index = True)
            #     pass

            # Model
            box_x_train['hours'] = box_x_train['date_'].dt.hour

#             try:
#                 data = box_x_train[['Sales','hours']]
#                 min_max_scaler = preprocessing.StandardScaler()
#                 np_scaled = min_max_scaler.fit_transform(data)
#                 data = pd.DataFrame(np_scaled)

#             except:
#                 min_fval = box_x_train['sales_quantity'].min()
#                 max_fval = box_x_train['sales_quantity'].max()
#                 error_df = error_df.append({'facility_id': int(facility_id) ,'item_id': int(item_id), 'min': round(min_fval, 2), 'max': round(max_fval, 2), 'error_reason': 'Breaking at Scaling Process'}, ignore_index = True)
#                 pass

            try:
                data = box_x_train[['Sales','hours']]
                kmeans = KMeans(n_clusters = 3, init = 'k-means++',n_init = 20, random_state = 42, algorithm = 'auto', max_iter = 500).fit(data)
                scores = kmeans.score(data)
                box_x_train['cluster'] = kmeans.predict(data)
                pair = {}
                pair_list = []
                pair[0] = box_x_train[box_x_train['cluster'] == 0]['Sales'].mean()
                pair_list.append(box_x_train[box_x_train['cluster'] == 0]['Sales'].mean())
                pair[1] = box_x_train[box_x_train['cluster'] == 1]['Sales'].mean()
                pair_list.append(box_x_train[box_x_train['cluster'] == 1]['Sales'].mean())
                pair[2] = box_x_train[box_x_train['cluster'] == 2]['Sales'].mean()
                pair_list.append(box_x_train[box_x_train['cluster'] == 2]['Sales'].mean())

                minimum = min(pair.values())
                pair_list.remove(minimum)
                minimum = [key for key, value in pair.items() if value == minimum][0]
                maximum = max(pair.values())
                pair_list.remove(maximum)
                maximum = [key for key, value in pair.items() if value == maximum][0]
                no_outleir_key = pair_list[0]
                no_outleir_key = [key for key, value in pair.items() if value == no_outleir_key][0]

                min_val = box_x_train[box_x_train['cluster'] == minimum]['sales_quantity'].mean()
                max_val = box_x_train[box_x_train['cluster'] == maximum]['sales_quantity'].mean()
#                 fval = box_x_train[box_x_train['cluster'] == no_outleir_key]['sales_quantity'].mean()

#                 min_fval = (min_val + fval)/2
#                 max_fval = (max_val + fval)/2

                final_df = final_df.append({'facility_id': int(facility_id) ,'item_id': int(item_id), 'min': round(min_val, 2), 'max': round(max_val, 2)}, ignore_index = True)
                item_count += 1
                print('Items Processed', item_count)

            except:
                min_fval = box_x_train['sales_quantity'].min()
                max_fval = box_x_train['sales_quantity'].max()
                error_df = error_df.append({'facility_id': int(facility_id) ,'item_id': int(item_id), 'min': round(min_fval, 2), 'max': round(max_fval, 2), 'error_reason': 'Breaking at K Means '}, ignore_index = True)
                pass

        print('Processed Store ', facility_id, ' Items Processed', item_count)
        print('Processed Store ', facility_id, ' Items Rejected', (len(list(set(x_train['item_id']))) - item_count))
        store_count += 1
        print('Processed Store', store_count, 'out of', len(list(set(x_train['facility_id']))))
        # final_df.to_csv('middle_final_df.csv', index = False)
        # error_df.to_csv('middle_error_df.csv', index = False)

    print("Before Query", "--- %s seconds ---" % (time.time() - start_time))
    print("After Query", "--- %s seconds ---" % (time.time() - start_time_aq))
    print('Memory utilised - ', psutil.Process().memory_info().rss / (10 ** (9)))  # in gb 

    final_df['facility_id'] = final_df['facility_id'].astype('int')
    final_df['item_id'] = final_df['item_id'].astype('int')

    final_df["updated_at"] = pd.to_datetime(datetime.today() + timedelta(hours=5.5))

    final_df = final_df[
                            [
                                "facility_id",
                                "item_id",
                                "min",
                                "max",
                                "updated_at",
                            ]
                        ]

    column_dtypes = [
                        {
                            "name": "facility_id",
                            "type": "integer",
                            "description": "unique identifier for dark store",
                        },
                        {
                            "name": "item_id", 
                            "type": "integer", 
                            "description": "unique identifier for item"
                        },
                        {
                            "name": "min",
                            "type": "float",
                            "description": "sales outlier range minimum sales",
                        },
                        {
                            "name": "max",
                            "type": "float",
                            "description": "sales outlier range maxiumum sales",
                        },
                        {
                            "name": "updated_at",
                            "type": "datetime",
                            "description": "updated time stamp in IST",
                        },
                    ]

    final_df = final_df.drop_duplicates()
    final_df.to_csv('final_df.csv', index = False)
    #print(final_df)

    kwargs = {
                "schema_name": "metrics",
                "table_name": "sales_outlier_min_max_range_v2",
                "column_dtypes": column_dtypes,
                "primary_key": ["facility_id", "item_id"],
                "sortkey": ["facility_id", "item_id"],
                "incremental_key": "updated_at",
                "load_type": "upsert",  # , # append, rebuild, truncate or upsert
                "table_description": "Sales Outlier Minimum Maximum Range",  # Description of the table being sent to redshift
            }
    pb.to_redshift(final_df, **kwargs)

    print("final_base write complete")

#     error_df['facility_id'] = error_df['facility_id'].astype('int')
#     error_df['item_id'] = error_df['item_id'].astype('int')

#     error_df["updated_at"] = pd.to_datetime(datetime.today() + timedelta(hours=5.5))

#     error_df = error_df[
#                             [
#                                 "facility_id",
#                                 "item_id",
#                                 "min",
#                                 "max",
#                                 "error_reason",
#                                 "updated_at",
#                             ]
#                         ]

#     column_dtypes = [
#                         {
#                             "name": "facility_id",
#                             "type": "integer",
#                             "description": "unique identifier for dark store",
#                         },
#                         {
#                             "name": "item_id", 
#                             "type": "integer", 
#                             "description": "unique identifier for item"
#                         },
#                         {
#                             "name": "min",
#                             "type": "float",
#                             "description": "sales outlier range minimum sales",
#                         },
#                         {
#                             "name": "max",
#                             "type": "float",
#                             "description": "sales outlier range maxiumum sales",
#                         },
#                         {
#                             "name": "error_reason",
#                             "type": "varchar",
#                             "description": "reason of error",
#                         },
#                         {
#                             "name": "updated_at",
#                             "type": "datetime",
#                             "description": "updated time stamp in IST",
#                         },
#                     ]

#     error_df = error_df.drop_duplicates()
#     error_df.to_csv('error_df.csv', index = False)
#     #print(error_df)

#     kwargs = {
#                 "schema_name": "metrics",
#                 "table_name": "sales_outlier_min_max_range_error",
#                 "column_dtypes": column_dtypes,
#                 "primary_key": ["facility_id", "item_id"],
#                 "sortkey": ["facility_id", "item_id"],
#                 "incremental_key": "updated_at",
#                 "load_type": "upsert",  # , # append, rebuild, truncate or upsert
#                 "table_description": "Sales Outlier Minimum Maximum Range",  # Description of the table being sent to redshift
#             }
#     pb.to_redshift(error_df, **kwargs)

#     print("error_base write complete")
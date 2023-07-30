import warnings
warnings.filterwarnings("ignore")

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

#Imports
from pyspark.sql import functions as psfunctions
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import split, udf, desc, concat, col, lit
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.window import Window
from pyspark.sql.functions import substring


#Multiprocessing HashJoin

from collections import defaultdict
import pandas as pd
import copy
import time
import numpy as np
from IPython.display import display
from multiprocessing import Pool

#Every Thread will generate a part of the final df
def generate_partial_df(var):
    t1 = time.time()
    tmp_df_size, col_list, hash_map, df2, key2 = var
    tmp_df = pd.DataFrame(index=np.arange(tmp_df_size), columns = col_list)
    row_counter = 0
    for index, row in df2.iterrows():
        for h in hash_map[row[key2]]:
            c = copy.deepcopy(h)
            c.pop()
            c = c + list(row)
            tmp_df.loc[row_counter] = c
            row_counter += 1
    print("Thread x took ", time.time()-t1)
    return tmp_df

def parallel_hashJoin(df1, key1, df2, key2, n_threads):
    
    #Setup dict that stores all rows according to their key
    hash_map = defaultdict(list)
    df1 = df1.toPandas()
    df2 = df2.toPandas()
    for index, row in df1.iterrows():
        hash_map[row[key1]].append(list(row))
    
    #Defining column names for the returned/constructed df 
    #EXAMPLE: [subject, 1, 2, 3, ..., object]
    #Thus we can garantee multiple consecutive joines with joines on the (object, subject) pair
    col_list = df1.columns.tolist()
    col_list2 = df2.columns.tolist()
    col_list= col_list + col_list2
    col_len = len(col_list)
    col_list = [str(i) for i in range(0, col_len-1)]
    col_list[col_len-2] = "object"
    col_list[0] = "subject"

    #Calculate needed size for joined df
    df_size = 0
    t1 = time.time()
    for index, row in df2.iterrows():
        for h in hash_map[row[key2]]:
            df_size += 1
    
    #Calculate amount of rows each thread has to join
    thread_size = df_size // n_threads
    thread_start_join_index = list() #[0, 23, 43124, 433324, df.size]
    thread_start_join_index.append(-1)
    thread_start_return_index = list()
    thread_start_return_index.append(0)
    
    counter = 0
    counter2 = 0
    thread_index = 0
    for index, row in df2.iterrows():
        counter2 += 1
        for h in hash_map[row[key2]]:
            counter += 1
        if (counter // thread_size > thread_index):
            thread_start_join_index.append(index) #start[i] matches end[i+1] inclus
            thread_start_return_index.append(counter)
            thread_index += 1
    
    thread_start_join_index.pop()
    thread_start_join_index.append(counter2)
    
    #Splitting the input dataframe into parts
    list_to_be_generated = list()
    t1 = time.time()
    for i in range(n_threads):
        list_to_be_generated.append((thread_start_return_index[i+1]-thread_start_return_index[i], copy.deepcopy(col_list), copy.deepcopy(hash_map), copy.deepcopy(df2.iloc[thread_start_join_index[i]+1:thread_start_join_index[i+1]+1]), key2))

    #Splitting the task into different threads
    with Pool(n_threads) as p:
        l = (p.map(generate_partial_df, list_to_be_generated))
    tmp_df = pd.concat(l)
    #display(tmp_df)
    return spark.createDataFrame(tmp_df)

if __name__ == '__main__':

    #Initialise session
    spark = SparkSession.builder.appName("cite").getOrCreate()
    sc = spark.sparkContext
    sqlc = SQLContext(sc)
    sqlContext = HiveContext(sc)

    #load the ralations into df
    relation_df = spark.read.option("delimiter", "\t").csv("100k.txt")

    #Rename columns
    relation_df = relation_df.select(col("_c0").alias("subject"), col("_c1").alias("relation"), col("_c2").alias("object"))
    relation_df = relation_df.withColumn("object",psfunctions.regexp_replace('object', ' .', ''))
    #relation_df.show()

    #Count how many relations there are
    relation_index_df = relation_df.select("relation").distinct()
    relation_index_df = relation_index_df.withColumn("index", psfunctions.row_number().over(Window.orderBy("relation")))

    #Creating a list containing the different df
    relation_list = relation_index_df.select("relation").toPandas()
    relation_list = list(relation_list['relation'])
    relation_df_list = list(map(lambda rel : relation_df.filter(relation_df.relation == rel), relation_list))
    relation_dict = dict()
    for i,x in enumerate(relation_list):
        relation_dict[x] = i

    #Creating a dict for all subject and objects
    objects_list = relation_df.select("object").toPandas()
    objects_list = list(objects_list["object"])
    subjects_list = relation_df.select("subject").toPandas()
    subjects_list = list(subjects_list["subject"])
    so_list = objects_list + subjects_list
    so_set = set(so_list)

    #Index -> SO
    so_dict = dict()

    #SO -> Index
    so_dict_reverse = dict()

    for i, x in enumerate(so_set):
        so_dict_reverse[x] = i
        so_dict[i] = x
        
    #Map the values from the dict to the subjects and objects
    map_col = psfunctions.create_map([psfunctions.lit(x) for i in so_dict_reverse.items() for x in i])
    relation_df_list_numbers = list(map(lambda df : df.withColumn("subject", map_col[psfunctions.col('subject')].cast("int")).withColumn('object', map_col[psfunctions.col('object')].cast("int")), relation_df_list))

    #Dropping the relation column, as it is superfluous
    relation_df_list = list(map(lambda df : df.drop("relation"), relation_df_list))
    relation_df_list_numbers = list(map(lambda df : df.drop("relation"), relation_df_list_numbers))
    #relation_df_list[0].show()


    #Performing the requested SQL expression with multithreaded HASHJOIN from right to left on 4 threads
    #Takes approx 10 Mins on i7-7700hq for 100k.txt
    threads_number = 4

    print("Starting the first join.")

    t1 = time.time()
    likes_hasreview = parallel_hashJoin(relation_df_list[relation_dict["wsdbm:likes"]], "object", relation_df_list[relation_dict["rev:hasReview"]], "subject", threads_number)
    print("First Join took ", time.time()-t1, "seconds")
    #print("likes_hasreview", likes_hasreview.count())

    t1 = time.time()
    friends_likes_hasreview = parallel_hashJoin(relation_df_list[relation_dict["wsdbm:friendOf"]], "object", likes_hasreview, "subject", threads_number)
    print("Second Join took ", time.time()-t1, "seconds")
    #print("friends_likes_hasreview", friends_likes_hasreview.count())

    t1 = time.time()
    follows_friends_likes_hasreview_reversed = parallel_hashJoin(relation_df_list[relation_dict["wsdbm:follows"]], "object", friends_likes_hasreview, "subject", threads_number)
    print("Third Join took ", time.time()-t1, "seconds")
    #print("follows_friends_likes_hasreview_reversed", follows_friends_likes_hasreview_reversed.count())

    follows_friends_likes_hasreview_reversed.show()


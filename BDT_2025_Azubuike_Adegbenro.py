# Databricks notebook source
# MAGIC %md
# MAGIC ###  Team names: Adegbenro Adeyinka Samuel and Azubuike Olive 
# MAGIC ###  Academic year: 2025
# MAGIC ###  Course name: Big Data Tools
# MAGIC

# COMMAND ----------

#Path variable for reading in the data
# Train data paths 

# dbfs:/FileStore/products.csv
# dbfs:/FileStore/order_items.csv
# dbfs:/FileStore/orders.csv
# dbfs:/FileStore/order_payments.csv
# dbfs:/FileStore/order_reviews.csv

# Test data paths

# dbfs:/FileStore/tmp/test_products.csv
# dbfs:/FileStore/tmp/test_order_items.csv
# dbfs:/FileStore/tmp/test_orders.csv
# dbfs:/FileStore/tmp/test_order_payments.csv

# COMMAND ----------

# MAGIC %md
# MAGIC  _Loading the data, cleaning it and doing preprocessing steps_

# COMMAND ----------

# Loading products
train_products = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/products.csv")
train_products.show(3)
train_products.printSchema()

# Loading products
test_products = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/tmp/test_products.csv")
test_products.show(3)
test_products.printSchema()

# COMMAND ----------

# DBTITLE 1,Identifying numerical columns on the products table.
numerical_cols = [
    "product_name_lenght",          
    "product_description_lenght",  
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


train_products.select(numerical_cols).describe().show()

numerical_cols = [
    "product_name_lenght",          
    "product_description_lenght",  
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]


test_products.select(numerical_cols).describe().show()

# COMMAND ----------

#check for duplicate in columns that should have unique values in the products table

train_products.groupBy("product_id").count().filter("count > 1").show()
test_products.groupBy("product_id").count().filter("count > 1").show()

# COMMAND ----------

train_order_items = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/order_items.csv")
train_order_items.show(5)
train_order_items.printSchema()

test_order_items = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/tmp/test_order_items.csv")
test_order_items.show(5)
test_order_items.printSchema()

# COMMAND ----------

# Descriptive statistics for the numerical columns on the order_items table

numerical_cols = [
    "order_item_id",        
    "price",  
    "shipping_cost",
]

train_order_items.select(numerical_cols).describe().show()

# Descriptive statistics for the numerical columns on the order_items table

numerical_cols = [
    "order_item_id",        
    "price",  
    "shipping_cost",
]

test_order_items.select(numerical_cols).describe().show()

# COMMAND ----------

#Missing Values: Checking for null or missing values in each column of the Order_items table
   
for col in train_order_items.columns:
    missing_count = train_order_items.filter(train_order_items[col].isNull() | (train_order_items[col] == "NA")).count()
    null_count = train_order_items.filter(train_order_items[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")


for col in test_order_items.columns:
    missing_count = train_order_items.filter(train_order_items[col].isNull() | (train_order_items[col] == "NA")).count()
    null_count = train_order_items.filter(train_order_items[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

# COMMAND ----------

train_order_items_products = train_order_items.join(train_products, on='product_id', how='left')
test_order_items_products = test_order_items.join(test_products, on='product_id', how='left')

# COMMAND ----------

# DBTITLE 1,Performing aggregations on order_items of train and test data
from pyspark.sql.functions import col, when, count, max, sum as _sum, expr

from pyspark.sql.functions import trim, lower

from pyspark.sql.functions import expr

from pyspark.sql.functions import *

# Calculate surface_area and amount columns

train_order_items_products = train_order_items_products.withColumn(
    "surface_area",
    expr("2 * (product_length_cm * product_width_cm + product_length_cm * product_height_cm + product_width_cm * product_height_cm)")
).withColumn(
    "amount",
    col("price") + col("shipping_cost")
)

# Group by order_id and perform aggregations

train_order_items_products_agg = train_order_items_products.groupBy("order_id").agg(
    countDistinct("product_id").alias("distinct_product_id"),
    countDistinct("product_category_name").alias("product_category_name_count"),
    _sum("amount").alias("sum_amount"),
    count("order_item_id").alias("count_order_item"),
    mean("product_name_lenght").alias("avg_product_name_length"),
    mean("product_description_lenght").alias("avg_product_description_length"),
    _sum("product_weight_g").alias("sum_product_weight"),
    _sum("surface_area").alias("sum_surface_area"),
    expr("mode(product_category_name)").alias("mode_product_category_name")
)

# Show the result
print(train_order_items_products_agg.count())


# Calculate surface_area and amount columns
test_order_items_products = test_order_items_products.withColumn(
    "surface_area",
    expr("2 * (product_length_cm * product_width_cm + product_length_cm * product_height_cm + product_width_cm * product_height_cm)")
).withColumn(
    "amount",
    col("price") + col("shipping_cost")
)

# Group by order_id and perform aggregations

test_order_items_products_agg = test_order_items_products.groupBy("order_id").agg(
    countDistinct("product_id").alias("distinct_product_id"),
    countDistinct("product_category_name").alias("product_category_name_count"),
    _sum("amount").alias("sum_amount"),
    count("order_item_id").alias("count_order_item"),
    mean("product_name_lenght").alias("avg_product_name_length"),
    mean("product_description_lenght").alias("avg_product_description_length"),
    _sum("product_weight_g").alias("sum_product_weight"),
    _sum("surface_area").alias("sum_surface_area"),
    expr("mode(product_category_name)").alias("mode_product_category_name")
)

# Show the result
test_order_items_products_agg.count()

# COMMAND ----------

# Group by order_id and count the number of rows for each group

grouped = train_order_items_products_agg.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()


# Group by order_id and count the number of rows for each group

grouped = test_order_items_products_agg.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()

# COMMAND ----------

train_orders = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/orders.csv")
train_orders.show(5)
train_orders.printSchema()

test_orders = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/tmp/test_orders.csv")
test_orders.show(5)
test_orders.printSchema()

# COMMAND ----------

  #Missing Values: Checking for null or missing values in each column of the Orders table
   
for col in train_orders.columns:
    missing_count = train_orders.filter((train_orders[col] == "NA")).count()
    null_count = train_orders.filter(train_orders[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

for col in test_orders.columns:
    missing_count = test_orders.filter((test_orders[col] == "NA")).count()
    null_count = test_orders.filter(test_orders[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

# COMMAND ----------

#We chose to drop our missing values because a customer can only review a complete order, and not an incomplete one.

train_orders = train_orders.replace('NA', None).dropna()
print(train_orders.count())

test_orders = test_orders.replace('NA', None).dropna()
test_orders.count()

# COMMAND ----------

#checking for duplicates in columns that should have unique values in the orders table.

train_orders.groupBy("order_id").count().filter("count > 1").show()
test_orders.groupBy("order_id").count().filter("count > 1").show()

# COMMAND ----------

# Changed the data types from string to timestamp on the orders table
 
from pyspark.sql.functions import *

# List of columns in the `orders` table to convert to timestamp

timestamp_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

# Apply to_timestamp() to each column in the list

for col_name in timestamp_columns:
    train_orders = train_orders.withColumn(col_name, to_timestamp(col(col_name), "yyyy-MM-dd HH:mm:ss"))

# Verifying the schema after conversion

train_orders.printSchema()
train_orders.show(3)


# List of columns in the `orders` table to convert to timestamp

timestamp_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

# Apply to_timestamp() to each column in the list

for col_name in timestamp_columns:
    test_orders = test_orders.withColumn(col_name, to_timestamp(col(col_name), "yyyy-MM-dd HH:mm:ss"))

# Verifying the schema after conversion

test_orders.printSchema()
test_orders.show(3)

# COMMAND ----------

# Loading order_payments

train_order_payments = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/order_payments.csv")
train_order_payments.show(5)
train_order_payments.printSchema()


test_order_payments = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/tmp/test_order_payments.csv")
test_order_payments.show(5)
test_order_payments.printSchema()

# COMMAND ----------

# Descriptive statistics for the numerical columns on the order_payments table

numerical_cols = [
    "payment_sequential",        
    "payment_installments",  
    "payment_value",
]

train_order_payments.select(numerical_cols).describe().show()


numerical_cols = [
    "payment_sequential",        
    "payment_installments",  
    "payment_value",
]

test_order_payments.select(numerical_cols).describe().show()

# COMMAND ----------

#Missing Values: Checking for null or missing values in each column of the Order_payments table

for col in train_order_payments.columns:
    missing_count = train_order_payments.filter((train_order_payments[col] == "NA")).count()
    null_count = train_order_payments.filter(train_order_payments[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")


for col in test_order_payments.columns:
    missing_count = test_order_payments.filter((test_order_payments[col] == "NA")).count()
    null_count = test_order_payments.filter(test_order_payments[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

# COMMAND ----------

#collect only data that have payment_installments

train_order_payments = train_order_payments.where(train_order_payments['payment_installments'] > 0)

test_order_payments = test_order_payments.where(test_order_payments['payment_installments'] > 0)

# COMMAND ----------


from pyspark.sql.functions import *

# Clean payment_type column

order_payments_test = train_order_payments.withColumn('payment_type', lower(trim(col('payment_type'))))

# Get updated distinct payment types

payment_types = [row['payment_type'] for row in order_payments_test.select('payment_type').distinct().collect()]

# Create dummy variables for each payment type

for payment in payment_types:
    order_payments_test = order_payments_test.withColumn(f'payment_{payment}', when(col('payment_type') == payment, 1).otherwise(0))

# Group by order_id and perform aggregations

train_order_payments_agg = order_payments_test.groupBy("order_id").agg(
    count("payment_sequential").alias("payment_sequential_count"),
    max("payment_installments").alias("max_payment_installments"),
    _sum("payment_value").alias("sum_payment_value"),
    *[_sum(f'payment_{payment}').alias(f'{payment}') for payment in payment_types]
)

# Display the final aggregated DataFrame

train_order_payments_agg.show(3)


# Clean payment_type column

order_payments_test = test_order_payments.withColumn('payment_type', lower(trim(col('payment_type'))))

# Get updated distinct payment types

payment_types = [row['payment_type'] for row in order_payments_test.select('payment_type').distinct().collect()]

# Create dummy variables for each payment type

for payment in payment_types:
    order_payments_test = order_payments_test.withColumn(f'payment_{payment}', when(col('payment_type') == payment, 1).otherwise(0))

# Group by order_id and perform aggregations

test_order_payments_agg = order_payments_test.groupBy("order_id").agg(
    count("payment_sequential").alias("payment_sequential_count"),
    max("payment_installments").alias("max_payment_installments"),
    _sum("payment_value").alias("sum_payment_value"),
    *[_sum(f'payment_{payment}').alias(f'{payment}') for payment in payment_types]
)

# Display the final aggregated DataFrame

test_order_payments_agg.show(3)

# COMMAND ----------

from pyspark.sql.functions import when, col

# Add a new column 'payment_by_installment' based on the condition

train_order_payments_agg = train_order_payments_agg.withColumn(
    "payment_by_installment",
    when(col("max_payment_installments") > 1, 1).otherwise(0)
)

test_order_payments_agg = test_order_payments_agg.withColumn(
    "payment_by_installment",
    when(col("max_payment_installments") > 1, 1).otherwise(0)
)

# COMMAND ----------

# Group by order_id and count the number of rows for each group

grouped = train_order_payments_agg.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()

grouped = test_order_payments_agg.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()

# COMMAND ----------

# Loading order_reviews

train_order_reviews = spark \
    .read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/order_reviews.csv")
train_order_reviews.show(5)
train_order_reviews.printSchema()

# COMMAND ----------

#Descriptive statistics for the numerical columns on the order_reviews table

numerical_cols = [
    "review_score",        
    
]

train_order_reviews.select(numerical_cols).describe().show()

# COMMAND ----------

#Missing Values: Checking for null or missing values in each column of the Order_reviews table

for col in train_order_reviews.columns:
    missing_count = train_order_reviews.filter((train_order_reviews[col] == "NA")).count()
    null_count = train_order_reviews.filter(train_order_reviews[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

# COMMAND ----------

# Checking for duplicate in columns that should have unique values in the order_reviews table.

train_order_reviews.groupBy("review_id").count().filter("count > 1").show(3)
train_order_reviews.count()

# COMMAND ----------

# Deleting the duplicates in the review_id because review_id should be unique.

from pyspark.sql.functions import *

# Identify `review_id` values with duplicates

duplicate_review_ids = train_order_reviews.groupBy("review_id").count().filter(col("count") > 1).select("review_id")

#  Exclude rows with duplicate `review_id` values from the DataFrame

train_order_reviews = train_order_reviews.join(duplicate_review_ids, on="review_id", how="left_anti")

# Verify the result
train_order_reviews.count()

# COMMAND ----------

# Converting both columns to time stamp so that our columns will be uniform.
 
from pyspark.sql.functions import *

# List of columns in the `orders` table to convert to timestamp
timestamp_columns = [
    "review_creation_date",
    "review_answer_timestamp"
]

# Apply to_timestamp() to each column in the list

for col_name in timestamp_columns:
    train_order_reviews = train_order_reviews.withColumn(col_name, to_timestamp(col(col_name), "yyyy-MM-dd HH:mm:ss"))

# Verifying the schema after conversion

train_order_reviews.printSchema()
train_order_reviews.show(3)

# COMMAND ----------

from pyspark.sql.functions import row_number, col
from pyspark.sql.window import Window
window_spec = Window.partitionBy("order_id").orderBy(col("review_creation_date").desc())

# Add a row number column to each row within the window

order_reviews_with_row_num = train_order_reviews.withColumn("row_num", row_number().over(window_spec))

# Filter to keep only the rows with row_num = 1 (latest review_creation_date for each order_id)

train_order_reviews = order_reviews_with_row_num.filter(col("row_num") == 1).drop("row_num")

# Show the result

train_order_reviews.count()

# COMMAND ----------

# Group by order_id and count the number of rows for each group

grouped = train_order_reviews.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()

# COMMAND ----------

# Step 1: Join orders with order_items

train_orders_items_merged = train_orders.join(train_order_items_products_agg, on='order_id', how='left')

# Step 2: Join the above result with order_payments

train_orders_items_payments_merged = train_orders_items_merged.join(train_order_payments_agg, on='order_id', how='left')

# Step 3: Join the result with order_reviews

train_basetable = train_orders_items_payments_merged.join(train_order_reviews, on='order_id', how='left')

# Display the final merged DataFrame

print(train_basetable.count())


test_orders_items_merged = test_orders.join(test_order_items_products_agg, on='order_id', how='left')

# Step 2: Join the above result with order_payments

test_basetable = test_orders_items_merged.join(test_order_payments_agg, on='order_id', how='left')

# Display the final merged DataFrame

test_basetable.count()

# COMMAND ----------

# Group by order_id and count the number of rows for each group

grouped = train_basetable.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()


grouped = test_basetable.groupBy("order_id").agg(
    count("*").alias("count")
)

# Filter groups where the count is greater than 1 and show the first 3 rows

grouped.filter(col("count") > 1).limit(3).show()

# COMMAND ----------

#check entire table for duplicates

train_basetable.groupBy(train_basetable.columns).count().filter("count > 1").show()
test_basetable.groupBy(test_basetable.columns).count().filter("count > 1").show()

# COMMAND ----------

# DBTITLE 1,Cleaned Base table
#Missing Values: Checking for null or missing values in each column of the Order_payments table

for col in train_basetable.columns:
    missing_count = train_basetable.filter((train_basetable[col] == "NA")).count()
    null_count = train_basetable.filter(train_basetable[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")


# COMMAND ----------

# MAGIC %md
# MAGIC _Feauture Engineering_

# COMMAND ----------

#deleted 740 orders with missing product information
#deleted 1154 orders with missing review information except review_answer_timestamp
#deleted 3 orders with missing payment information
train_basetable = train_basetable.dropna(subset=["avg_product_name_length"])
train_basetable = train_basetable.dropna(subset=["review_score"])
train_basetable = train_basetable.dropna(subset=["payment_sequential_count"])

# COMMAND ----------

from pyspark.sql.functions import *
train_basetable.where(col('review_answer_timestamp').isNull()).select('order_id').show()

# COMMAND ----------

# Filter records where review_answer_timestamp is not null

filtered_df = train_basetable.filter(col("review_answer_timestamp").isNotNull())

# Compute the difference in days

diff_df = filtered_df.withColumn("date_diff", datediff(col("review_answer_timestamp"), col("review_creation_date")))

# Calculate the average and convert it to an integer
avg_diff = int(diff_df.select(avg("date_diff")).collect()[0][0])

# COMMAND ----------


from pyspark.sql.functions import col, when, date_add

# Replace null values in review_answer_timestamp with review_creation_date plus average difference between review_answer_timestamp and review_creation_date

train_basetable = train_basetable.withColumn(
    "review_answer_timestamp",
    when(
        col("review_answer_timestamp").isNull(),
        date_add(col("review_creation_date"), avg_diff)
    ).otherwise(col("review_answer_timestamp"))
)

# COMMAND ----------

train_basetable.count()

# COMMAND ----------


for col in test_basetable.columns:
    missing_count = test_basetable.filter((test_basetable[col] == "NA")).count()
    null_count = test_basetable.filter(test_basetable[col].isNull()).count()
    print(f"{col}: {missing_count} total missing values (including 'NA'), {null_count} NULL values only")

# COMMAND ----------

#deleted 78 orders with missing product information

test_basetable = test_basetable.dropna(subset=["avg_product_name_length"])

# COMMAND ----------

# DBTITLE 1,Renamed some feature columns
train_basetable = train_basetable.withColumnRenamed("distinct_product_id", "product_id_count").withColumnRenamed("count_order_item", "order_item_count").withColumnRenamed("sum_product_weight", "order_weight").withColumnRenamed("sum_surface_area", "order_surface_area").withColumnRenamed("count_payment_sequential", "payment_sequential_count")

test_basetable = test_basetable.withColumnRenamed("distinct_product_id", "product_id_count").withColumnRenamed("count_order_item", "order_item_count").withColumnRenamed("sum_product_weight", "order_weight").withColumnRenamed("sum_surface_area", "order_surface_area").withColumnRenamed("count_payment_sequential", "payment_sequential_count")

# COMMAND ----------

train_basetable.show(3)

# COMMAND ----------

# DBTITLE 1,Feature creation
from pyspark.sql.functions import *

train_basetable = train_basetable.withColumn("delivery_delay", datediff(col("order_delivered_customer_date"), col("order_estimated_delivery_date")))
train_basetable = train_basetable.withColumn("carrier_pickup_speed", datediff(col("order_delivered_carrier_date"), col("order_approved_at")))
train_basetable = train_basetable.withColumn("customer_wait_time", datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp")))
train_basetable = train_basetable.withColumn("delivery_performance", abs(col("delivery_delay")))
train_basetable = train_basetable.withColumn("is_late_delivery", when(col("delivery_delay") > 0, 1).otherwise(0))
train_basetable = train_basetable.withColumn("high_installments", when(col("max_payment_installments") > 3, 1).otherwise(0))
train_basetable = train_basetable.withColumn("heavy_order", when(col("order_weight") > 10, 1).otherwise(0))
train_basetable = train_basetable.withColumn("order_weekday", dayofweek(col("order_purchase_timestamp")))
train_basetable = train_basetable.withColumn("avg_product_weight", expr("order_weight / order_item_count"))
train_basetable = train_basetable.withColumn("avg_product_surface_area", expr("order_surface_area / order_item_count"))
train_basetable = train_basetable.withColumn("avg_product_price", expr("sum_payment_value / order_item_count"))


test_basetable = test_basetable.withColumn("delivery_delay", datediff(col("order_delivered_customer_date"), col("order_estimated_delivery_date")))
test_basetable = test_basetable.withColumn("carrier_pickup_speed", datediff(col("order_delivered_carrier_date"), col("order_approved_at")))
test_basetable = test_basetable.withColumn("customer_wait_time", datediff(col("order_delivered_customer_date"), col("order_purchase_timestamp")))
test_basetable = test_basetable.withColumn("delivery_performance", abs(col("delivery_delay")))
test_basetable = test_basetable.withColumn("is_late_delivery", when(col("delivery_delay") > 0, 1).otherwise(0))
test_basetable = test_basetable.withColumn("high_installments", when(col("max_payment_installments") > 3, 1).otherwise(0))
test_basetable = test_basetable.withColumn("heavy_order", when(col("order_weight") > 10, 1).otherwise(0))
test_basetable = test_basetable.withColumn("order_weekday", dayofweek(col("order_purchase_timestamp")))
test_basetable = test_basetable.withColumn("avg_product_weight", expr("order_weight / order_item_count"))
test_basetable = test_basetable.withColumn("avg_product_surface_area", expr("order_surface_area / order_item_count"))
test_basetable = test_basetable.withColumn("avg_product_price", expr("sum_payment_value / order_item_count"))

# COMMAND ----------

# DBTITLE 1,More features based on Time
from pyspark.sql.functions import *
#Purchase Hour (Peak vs. Off-Peak Orders)
train_basetable = train_basetable.withColumn("order_hour", hour(col("order_purchase_timestamp")))

# Order Weekday
train_basetable = train_basetable.withColumn("order_weekday", dayofweek(col("order_purchase_timestamp")))

# Is Weekend Order
train_basetable = train_basetable.withColumn("is_weekend", when(col("order_weekday").isin([1, 7]), 1).otherwise(0))

# Extracting month, quarter, and year
train_basetable = train_basetable.withColumn("order_month", month(col("order_purchase_timestamp")))
train_basetable = train_basetable.withColumn("order_quarter", quarter(col("order_purchase_timestamp")))
train_basetable = train_basetable.withColumn("order_year", year(col("order_purchase_timestamp")))

#Customers may behave differently when ordering at the end of the month (e.g., payday effects).
# Check if the order was placed in the last 5 days of the month
train_basetable = train_basetable.withColumn("is_end_of_month", 
                                 when(dayofmonth(col("order_purchase_timestamp")) >= dayofmonth(last_day(col("order_purchase_timestamp"))) - 5, 1).otherwise(0))
                                 
#To track early vs. late month trends:
train_basetable = train_basetable.withColumn("days_since_start_of_month", dayofmonth(col("order_purchase_timestamp")))

#Orders near holidays might have different characteristics (e.g., late deliveries during Christmas). You can manually define holiday periods and check if an order falls within them.
# Define holiday periods (Example: Christmas, Black Friday, New Year)
holiday_dates = [11, 12]  # November (Black Friday) & December (Christmas)
train_basetable = train_basetable.withColumn("is_holiday_season", when(col("order_month").isin(holiday_dates), 1).otherwise(0))

# COMMAND ----------

# DBTITLE 1,Extra Features
from pyspark.sql.functions import *
#Purchase Hour (Peak vs. Off-Peak Orders)
test_basetable = test_basetable.withColumn("order_hour", hour(col("order_purchase_timestamp")))

# Order Weekday
test_basetable = test_basetable.withColumn("order_weekday", dayofweek(col("order_purchase_timestamp")))

# Is Weekend Order
test_basetable = test_basetable.withColumn("is_weekend", when(col("order_weekday").isin([1, 7]), 1).otherwise(0))

# Extracting month, quarter, and year
test_basetable = test_basetable.withColumn("order_month", month(col("order_purchase_timestamp")))
test_basetable = test_basetable.withColumn("order_quarter", quarter(col("order_purchase_timestamp")))
test_basetable = test_basetable.withColumn("order_year", year(col("order_purchase_timestamp")))

#Customers may behave differently when ordering at the end of the month (e.g., payday effects).
# Check if the order was placed in the last 5 days of the month
test_basetable = test_basetable.withColumn("is_end_of_month", 
                                 when(dayofmonth(col("order_purchase_timestamp")) >= dayofmonth(last_day(col("order_purchase_timestamp"))) - 5, 1).otherwise(0))
                                 
#To track early vs. late month trends:
test_basetable = test_basetable.withColumn("days_since_start_of_month", dayofmonth(col("order_purchase_timestamp")))

#Orders near holidays might have different characteristics (e.g., late deliveries during Christmas). You can manually define holiday periods and check if an order falls within them.
# Define holiday periods (Example: Christmas, Black Friday, New Year)

holiday_dates = [11, 12]  # November (Black Friday) & December (Christmas)
test_basetable = test_basetable.withColumn("is_holiday_season", when(col("order_month").isin(holiday_dates), 1).otherwise(0))

# COMMAND ----------

train_basetable = train_basetable.withColumn("review_score_label", when(col("review_score") >= 4, 1).otherwise(0))

# COMMAND ----------

#remove canceled orders
train_basetable = train_basetable.where(col('order_status') != 'canceled')

#drop the order status column
train_basetable = train_basetable.drop("order_status")

# COMMAND ----------

#remove canceled orders
test_basetable = test_basetable.where(col('order_status') != 'canceled')

#drop the order status column
test_basetable = test_basetable.drop("order_status")

# COMMAND ----------

# compare sum_payment_value and sum_amount
# delete unmatching amounts and drop one of the columns
train_basetable = train_basetable.filter(
    col("sum_payment_value").cast("int") == col("sum_amount").cast("int")
)
train_basetable = train_basetable.drop("sum_payment_value")
train_basetable.count()

# COMMAND ----------

# compare sum_payment_value and sum_amount
# delete unmatching amounts and drop one of the columns

test_basetable = test_basetable.filter(
    col("sum_payment_value").cast("int") == col("sum_amount").cast("int")
)
test_basetable = test_basetable.drop("sum_payment_value")
test_basetable.count()

# COMMAND ----------

#drop columns that are not relevant to the order predictions

train_basetable_final = train_basetable.drop("review_id", "customer_id","mode_product_category_name","review_score","review_creation_date","review_answer_timestamp")
test_basetable_final = test_basetable.drop("review_id", "customer_id","mode_product_category_name","review_score","review_creation_date","review_answer_timestamp")

# COMMAND ----------

train_basetable_final.show(3)
test_basetable_final.show(3)

# COMMAND ----------

# DBTITLE 1,Feature Transformation
from pyspark.ml.feature import RFormula

train_basetable_final_pred = RFormula(formula="review_score_label ~ . - order_id").fit(train_basetable_final).transform(train_basetable_final)
print(str(train_basetable_final_pred.count()))

# COMMAND ----------

# Show features and label
train_basetable_final_pred.select("features", "label").show(5, truncate=False)

# COMMAND ----------

from pyspark.ml.feature import RFormula

# Add a dummy 'review_score_label' column (for testing purposes)
test_basetable_final = test_basetable_final.withColumn("review_score_label", lit(0))  # or lit(1) depending on your case

basetable_test = RFormula(formula="review_score_label ~ . - order_id").fit(test_basetable_final).transform(test_basetable_final)
print(str(basetable_test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC _Machine model training _

# COMMAND ----------

# Show features and label
basetable_test.select("features", "label").show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Random Data split into training and validation
basetable_train, basetable_val = train_basetable_final_pred.randomSplit([0.8, 0.2],seed=123)

print(train_basetable_final_pred.count(),basetable_train.count(),basetable_val.count())

# COMMAND ----------

# MAGIC %md
# MAGIC _Logistic Regresion Modelling_

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
logreg_model = LogisticRegression().fit(basetable_train)

# COMMAND ----------

logreg_pred = logreg_model.transform(basetable_val)
logreg_pred.show(5)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
logreg_auc =BinaryClassificationEvaluator().evaluate(logreg_pred)
print(f"Logic Regression AUC: {logreg_auc}")

accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(logreg_pred)
print(f"Logic Regression Accuracy: {accuracy}")

# COMMAND ----------

# DBTITLE 1,Hyper Parameter Tunning
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.classification import GBTClassifier

# # Define the GBTClassifier
# gbt = GBTClassifier(labelCol="label", featuresCol="features")

# # Define a parameter grid for hyperparameter tuning
# paramGrid = (ParamGridBuilder()
#              .addGrid(gbt.maxDepth, [5, 7])
#              .addGrid(gbt.maxIter, [40, 50])
#              .addGrid(gbt.stepSize, [0.05, 0.1])
#              .build())

# # Define an evaluator for Binary Classification
# evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# # Define the CrossValidator
# crossval = CrossValidator(estimator=gbt,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=evaluator,
#                           numFolds=2)

# # Train the model with cross-validation
# cv_model = crossval.fit(basetable_train)

# # Get the best model from cross-validation
# best_gbt_model = cv_model.bestModel

# # Display the best hyperparameters
# best_max_depth = best_gbt_model.getMaxDepth()
# best_max_iter = best_gbt_model.getMaxIter()
# best_step_size = best_gbt_model.getStepSize()

# print(f"Best Max Depth: {best_max_depth}")
# print(f"Best Max Iter: {best_max_iter}")
# print(f"Best Step Size: {best_step_size}")


# COMMAND ----------

# MAGIC %md
# MAGIC _Gradient boosting Modelling_

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train Gradient Boosting model
gbt = GBTClassifier(labelCol="label", featuresCol="features", seed =123, maxIter=50)
gbt_model = gbt.fit(basetable_train)

# Make predictions
gbt_pred = gbt_model.transform(basetable_val)

# Evaluate accuracy
gbt_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(gbt_pred)
print(f"Gradient Boosting Accuracy: {gbt_accuracy}")

AUC = BinaryClassificationEvaluator().evaluate(gbt_pred)
print(f"Gradient Boosting AUC: {AUC}")

# COMMAND ----------

# DBTITLE 1,Hyper parameter Tunning
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# # Define the Random Forest model
# rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# # Define a parameter grid for hyperparameter tuning
# paramGrid = (ParamGridBuilder()
#              .addGrid(rf.maxDepth, [5, 7, 10])
#              .addGrid(rf.numTrees, [100, 200, 300])
#              .addGrid(rf.maxBins, [32, 64])
#              .build())

# # Define an evaluator for Binary Classification
# evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# # Define the CrossValidator
# crossval = CrossValidator(estimator=rf,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=evaluator,
#                           numFolds=2)

# # Train the model with cross-validation
# cv_model = crossval.fit(basetable_train)

# # Get the best model from cross-validation
# best_rf_model = cv_model.bestModel

# # Display the best hyperparameters
# best_max_depth = best_rf_model.getMaxDepth()
# best_num_trees = best_rf_model.getNumTrees()
# best_max_bins = best_rf_model.getMaxBins()

# print(f"Best Max Depth: {best_max_depth}")
# print(f"Best Number of Trees: {best_num_trees}")
# print(f"Best Max Bins: {best_max_bins}")


# COMMAND ----------

# MAGIC %md
# MAGIC _Random Forest Modelling_

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train Random Forest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features",numTrees = 200, seed = 123)
rf_model = rf.fit(basetable_train)

# Make predictions
rf_pred = rf_model.transform(basetable_val)

# Evaluate accuracy
rf_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy}")

AUC = BinaryClassificationEvaluator().evaluate(rf_pred)
print(f"Random Forest AUC: {AUC}")

# COMMAND ----------

# MAGIC %md
# MAGIC _Final Prediction_

# COMMAND ----------

# Make predictions with holdout_data
gbt_pred_test = gbt_model.transform(basetable_test)

#Create an output table with user_id, prediction
out_pred = gbt_pred_test.select("order_id","prediction")

#check the prediction per cout
prediction_counts = out_pred.groupBy("prediction").count()
prediction_counts.show()

#output the prediction
out_pred.display()

# COMMAND ----------

# MAGIC %md
# MAGIC _Visualisation of key insights_

# COMMAND ----------

# Feature importances for gradient boosting model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    # Extract feature metadata
    attrs = dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]
    feature_list = [attr for attr_group in attrs.values() for attr in attr_group]
   
    # Ensure feature indices are within bounds
    valid_features = [f for f in feature_list if f["idx"] < len(featureImp)]
   
    # Build DataFrame and map importance scores
    feature_importance_df = pd.DataFrame(valid_features)
    feature_importance_df["score"] = feature_importance_df["idx"].apply(lambda x: featureImp[x])
   
    return feature_importance_df.sort_values("score", ascending=False)
 
top_features = ExtractFeatureImp(gbt_model.featureImportances, basetable_train, "features")

# Assuming 'top_features' is your pandas DataFrame containing the top features and scores
top_features = top_features.sort_values('score', ascending=False)

# Plotting the top features
plt.figure(figsize=(10, 6))
sns.barplot(x='score', y='name', data=top_features.head(20), palette="viridis")
plt.title("Top 20 Feature Importances for Gradient Boosting Model")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.show()

# COMMAND ----------

# DBTITLE 1,**Roc Curve Evaluation**
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get the probability and label columns
roc_data = gbt_pred.select("probability", "label")

# Convert the probability vector to an array
roc_data = roc_data.withColumn("probability_array", vector_to_array(col("probability")))

# Collect the data into Python (this can be slow for large datasets, consider working with a subset of data if necessary)
roc_data_collected = roc_data.select("label", "probability_array").rdd.collect()

# Extract true labels and predicted probabilities
labels, probs = zip(*[(row['label'], row['probability_array'][1]) for row in roc_data_collected])  # Getting the probability for the positive class

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# COMMAND ----------

# DBTITLE 1,**Confusion Matrix Evaluation **
#Confusion Matrix
 
def confusion_matrix(pred_df):
    """
    Computes the confusion matrix for a Spark DataFrame with label & prediction columns.
    Parameters:
        pred_df (DataFrame): DataFrame with "label" and "prediction" columns.
    Returns:
        Pandas DataFrame: Confusion matrix.
    """
    cm = (pred_df
          .groupBy("label")
          .pivot("prediction")
          .count()
          .fillna(0)  # Replace missing values with 0
          .orderBy("label"))
 
    return cm.toPandas()
 
# Compute confusion matrices for validation and test sets
print("Confusion Matrix - Validation Set")
cm_val = confusion_matrix(gbt_pred)
print(cm_val)

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns

# Aggregating data: Calculate the sum of is_late_delivery for each review_score_label
agg_data_late_delivery = train_basetable_final.groupBy("review_score_label") \
    .agg(F.sum("is_late_delivery").alias("sum_late_delivery"))

# Collect the aggregated data into a Pandas DataFrame
agg_data_late_delivery_pd = agg_data_late_delivery.toPandas()

# Plot the bar plot: sum_late_delivery on y-axis, review_score_label on x-axis
plt.figure(figsize=(10, 6))
sns.barplot(x='review_score_label', y='sum_late_delivery', data=agg_data_late_delivery_pd, palette="viridis")
plt.title("Review Score vs Sum of Late Deliveries")
plt.xlabel("Review Score")
plt.ylabel("Sum of Late Deliveries")
plt.show()


# COMMAND ----------

from pyspark.sql import functions as F

# Aggregating data: Calculate the average payment installment for each review_score_label
agg_data_review_score = train_basetable_final.groupBy("review_score_label") \
    .agg(F.sum("order_item_count").alias("sum_order_item_count"))

# Collect the aggregated data into a Pandas DataFrame
agg_data_review_score_pd = agg_data_review_score.toPandas()

# Plot the bar plot with switched axes: avg_payment_installment on x-axis, review_score_label on y-axis
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the bar plot with swapped axes: review_score_label on x-axis, avg_payment_installment on y-axis
plt.figure(figsize=(10, 6))
sns.barplot(x='review_score_label', y='sum_order_item_count', data=agg_data_review_score_pd, palette="viridis")
plt.title("Review Score vs Order Items")
plt.xlabel("Review Score")
plt.ylabel("Order Items")
plt.show()


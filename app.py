import sys
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, avg, stddev, lag, datediff, round, expr, sqrt, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_path'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read from Glue Data Catalog
dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database="raw_data_db",
    table_name="raw_data",
    format="csv"
)

# Convert to Spark DataFrame
df = dynamic_frame.toDF()
# Your existing transformation logic remains the same
window_spec = Window.partitionBy("ticker").orderBy("Date")
df_with_returns = df.withColumn("prev_close", lag("close", 1).over(window_spec)) \
    .withColumn("daily_return", (col("close") - col("prev_close")) / col("prev_close"))

# Calculate average daily return for each date
avg_daily_returns = df_with_returns.groupBy("Date") \
    .agg(round(avg("daily_return") * lit(100), 2).alias("avg_daily_return_pct")) \
    .orderBy("Date")

# Calculate average trading worth
trading_worth = df.withColumn("trading_worth", col("close") * col("volume")) \
    .groupBy("ticker") \
    .agg(avg("trading_worth").alias("avg_trading_worth")) \
    .orderBy(col("avg_trading_worth").desc())

# Calculate volatility
volatility = df_with_returns.groupBy("ticker") \
    .agg(round(stddev("daily_return") * sqrt(lit(252)) * lit(100), 2).alias("annualized_volatility_pct")) \
    .orderBy(col("annualized_volatility_pct").desc())

# Calculate 30-day returns
window_30_days = Window.partitionBy("ticker").orderBy("Date")
df_30_day_returns = df.withColumn("price_30_days_ago",
    lag("close", 30).over(window_30_days)) \
    .withColumn("return_30_day",
        round(((col("close") - col("price_30_days_ago")) / col("price_30_days_ago")) * lit(100), 2)) \
    .select("Date", "ticker", "return_30_day") \
    .where(col("return_30_day").isNotNull()) \
    .orderBy(col("return_30_day").desc()) \
    .limit(3)


output_base = args['output_path']
daily_returns_dyf = DynamicFrame.fromDF(avg_daily_returns, glueContext, "daily_returns")
trading_worth_dyf = DynamicFrame.fromDF(trading_worth.limit(1), glueContext, "trading_worth")
volatility_dyf = DynamicFrame.fromDF(volatility.limit(1), glueContext, "volatility")
top_returns_dyf = DynamicFrame.fromDF(df_30_day_returns, glueContext, "top_returns")


outputs = [
    {"frame": daily_returns_dyf, "name": "daily_returns"},
    {"frame": trading_worth_dyf, "name": "trading_worth"},
    {"frame": volatility_dyf, "name": "volatility"},
    {"frame": top_returns_dyf, "name": "top_returns"}
]

for output in outputs:
    # Create sink
    s3output = glueContext.getSink(
        path=f"{output_base}/{output['name']}/",
        connection_type="s3",
        updateBehavior="UPDATE_IN_DATABASE",
        partitionKeys=[],
        compression="snappy",
        enableUpdateCatalog=True,
        transformation_ctx="s3output"
    )
    
    # Set catalog info
    s3output.setCatalogInfo(
        catalogDatabase="processed_data_db",
        catalogTableName=output['name']
    )
    
    # Set format
    s3output.setFormat("glueparquet", compression="snappy")
    
    # Write frame
    s3output.writeFrame(output['frame'])



# Commit the job
job.commit()

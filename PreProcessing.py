# Databricks notebook source
import os
from functools import reduce
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# COMMAND ----------

# Initialize Spark session
spark = (
    SparkSession.builder
    .appName("Data Processing")
    .getOrCreate()
)

# Define the range of years for the dataset
years = [2018 + i for i in range(5)]

# Read each year's Parquet file, add a "year" column, and store it in a dictionary
year_wise_data = {
    year: spark.read.parquet(f"dbfs:/FileStore/tables/Project/Combined_Flights_{year}.parquet")
            .withColumn("year", F.lit(year))
    for year in years
}

# Combine all DataFrames into a single DataFrame using unionByName
combined_df = reduce(DataFrame.unionByName, year_wise_data.values())

# Display the first few rows of the combined DataFrame to verify
display(combined_df)

# COMMAND ----------

sampled_25_df = combined_df.sample(fraction=0.25, seed=42)
sampled_25_df.count()

# Write the sampled DataFrame to a Parquet file
output_path = "dbfs:/FileStore/tables/Project/sample_25_data.parquet"
sampled_25_df.coalesce(1).write.mode("overwrite").parquet(output_path)


# COMMAND ----------

sampled_df = spark.read.parquet("dbfs:/FileStore/tables/Project/sample_25_data.parquet/part-00000-tid-8227792046037547548-a3152df5-8825-475d-9def-73cc117ea65e-41-1-c000.snappy.parquet")
display(sampled_df)

# COMMAND ----------

sampled_df = sampled_df.drop(*["Diverted", "year", "Quarter", "Marketing_Airline_Network", "Marketing_Airline_Network", "DOT_ID_Marketing_Airline", "IATA_Code_Marketing_Airline", "Flight_Number_Marketing_Airline", "Operating_Airline", "IATA_Code_Operating_Airline", "Tail_Number", "Flight_Number_Operating_Airline", "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID", "OriginStateFips", "OriginStateName", "OriginWac", "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "DestStateFips", "DestStateName", "DestWac", "DepDel15", "DepTimeBlk", "WheelsOff", "WheelsOn", "ArrDel15", "ArrTimeBlk", "DistanceGroup", "__index_level_0__"])
display(sampled_df)

# COMMAND ----------

print(combined_df.count())
print(sampled_df.count())

# COMMAND ----------

total_count = sampled_df.count()
# columns = ["col1", "col2", "col3"]
# Find columns with more than 50% null values
null_columns = []
for col_name in sampled_df.columns:
    null_count = sampled_df.filter(F.col(col_name).isNull()).count()
    null_percentage = (null_count / total_count) * 100

    if null_percentage != 0:
        null_columns.append((col_name, null_percentage))

# Print columns with more than 50% null values
if null_columns:
    for column, percentage in null_columns:
        print(f"Column '{column}' has {percentage:.2f}% null values.")
else:
    print("No columns have more than 50% null values.")

# COMMAND ----------

dot_id = sampled_df.groupby( F.col("DOT_ID_Operating_Airline"), F.col("Airline")).count()
display(dot_id)


# COMMAND ----------

div_count = sampled_df.groupby(F.col("DivAirportLandings")).count()
display(div_count)

# COMMAND ----------

sampled_df = sampled_df.withColumn("DepTime",
                   F.format_string("%02d:%02d",
                                   (F.col("DepTime") / 100).cast("int"),  # Extract hours
                                   (F.col("DepTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

sampled_df = sampled_df.withColumn("CRSDepTime",
                   F.format_string("%02d:%02d",
                                   (F.col("CRSDepTime") / 100).cast("int"),  # Extract hours
                                   (F.col("CRSDepTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

sampled_df = sampled_df.withColumn("ArrTime",
                   F.format_string("%02d:%02d",
                                   (F.col("ArrTime") / 100).cast("int"),  # Extract hours
                                   (F.col("ArrTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

sampled_df = sampled_df.withColumn("CRSArrTime",
                   F.format_string("%02d:%02d",
                                   (F.col("CRSArrTime") / 100).cast("int"),  # Extract hours
                                   (F.col("CRSArrTime") % 100).cast("int")   # Extract minutes
                                  )
                  )

# Show the result
sampled_df = sampled_df.withColumn("flight_hour", F.substring(F.col("DepTime"), 1, 2).cast("int"))

# Classify the time into Early Morning, Morning, Afternoon, or Night
sampled_df = sampled_df.withColumn("time_of_day",
                   F.when((F.col("flight_hour") >= 0) & (F.col("flight_hour") < 6), "Early Morning")
                    .when((F.col("flight_hour") >= 6) & (F.col("flight_hour") < 12), "Morning")
                    .when((F.col("flight_hour") >= 12) & (F.col("flight_hour") < 16), "Afternoon")
                    .when((F.col("flight_hour") >= 16) & (F.col("flight_hour") < 20), "Evening")
                    .when((F.col("flight_hour") >= 20) & (F.col("flight_hour") < 24), "Night")
                    .otherwise("Unknown"))

# Drop the temporary flight_hour column if not needed
sampled_df = sampled_df.drop("flight_hour")
display(sampled_df)

# COMMAND ----------

day_time = sampled_df.select(F.col("time_of_day")).distinct()
display(day_time)

# COMMAND ----------

time_delay_avg = sampled_df.groupBy("time_of_day").avg("DepDelayMinutes").orderBy("time_of_day")

pd_df = time_delay_avg.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(pd_df["time_of_day"], pd_df["avg(DepDelayMinutes)"])
plt.xlabel("Time of Day")
plt.ylabel("Average Departure Delay (Minutes)")
plt.title("Average Departure Delay by Time of Day")
plt.show()


# COMMAND ----------

day_time_group = sampled_df.groupBy("DayOfWeek", "time_of_day").count().orderBy("DayOfWeek", "time_of_day")

pd_df2 = day_time_group.toPandas()

plt.figure(figsize=(14, 8))
for day in sorted(pd_df2["DayOfWeek"].unique()):
    subset = pd_df2[pd_df2["DayOfWeek"] == day]
    plt.plot(subset["time_of_day"], subset["count"], marker='o', label=f"Day {day}")

plt.xlabel("Time of Day")
plt.ylabel("Avg Count of Departures")
plt.title("Count of Departures by Time of Day for Each Day of the Week")
plt.legend(title="Day of Week")
plt.grid(True)
plt.show()

# COMMAND ----------

pd_df3 = sampled_df.select("DepDelay", "ArrDelay", "Distance", "DayOfWeek", "DepDelayMinutes").toPandas()

# Plotting DepDelayMinutes vs ArrDelayMinutes
plt.figure(figsize=(10, 6))
plt.scatter(pd_df3["DepDelay"], pd_df3["ArrDelay"], alpha=0.5)
plt.xlabel("Departure Delay (Minutes)")
plt.ylabel("Arrival Delay (Minutes)")
plt.title("Departure Delay vs Arrival Delay")
plt.grid(True)
plt.show()

# COMMAND ----------

pd_df3[['DepDelay','ArrDelay']].corr()

# COMMAND ----------

# Plotting DepDelayMinutes vs ArrDelayMinutes
plt.figure(figsize=(10, 6))
plt.scatter(pd_df3["Distance"], pd_df3["DepDelay"], alpha=0.5)
plt.xlabel("Distance (in miles)")
plt.ylabel("Departure Delay (in minutes)")
plt.title("Distance vs Departure Delay Group")
plt.grid(True)
plt.show()

# COMMAND ----------



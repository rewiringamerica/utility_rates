# Databricks notebook source
# MAGIC %md ### Time Test
# MAGIC Test rates calculations with a large number buildings (100K). On Rates Cluster, Driver = 64 GB, 8 cores:
# MAGIC * Building test data : 15s
# MAGIC * Construct load profile numpy matrix: 19s
# MAGIC * Calculating bills: 1s
# MAGIC
# MAGIC 100K ResStock buildings = 25M households, which is well over the population of the largest state (CA = 13.5M hh), and the largest utility (PG&E = 5.2M hh). This illustrates that this method is sufficient for our purposes, which will generally only be for one utility at a time. 
# MAGIC

# COMMAND ----------

N_BUILDINGS = 100000

# COMMAND ----------

import numpy as np
import pandas as pd
import sys

sys.path.append("../src")

from rates.rate_structure import FlatRateSchedule
from rates.modeled_rate import ModeledRate

# COMMAND ----------

def generate_large_load_profile(num_buildings=5000):
    """
    Generate toy hourly load profiles for num_buildings buildings. 

    Note that the true number of days in the month is not respected, this just assumes 31 days per month
    since datetime operations are expensive, and the downstream code 0 pads to 31 days per month anyway. 
    
    """
    
    hours_per_year = 372 * 24
    total_rows = num_buildings * hours_per_year
    
    # Pre-allocate arrays
    building_ids = np.repeat(np.arange(0, num_buildings * 10, 10), hours_per_year).astype('int32')
    electricity_kwh = np.full(total_rows, 1.5, dtype='float32')
    
    # Generate time columns once and repeat for all buildings
    hours = np.tile(np.arange(hours_per_year), num_buildings)
    month = ((hours // 24) // 31 + 1).astype('int8')
    day = ((hours // 24) % 31 + 1).astype('int8')
    hour = (hours % 24).astype('int8')
    
    df = pd.DataFrame({
        'building_id': building_ids,
        'electricity_kwh': electricity_kwh,
        'month': month,
        'day': day,
        'hour': hour
    })
    
    return df


# COMMAND ----------

# create flat rate schedule -- note that computation is identical for any rate schedule,
# a flat schedule is merely exploded out into a full rate matrix by [month, hour]
rate_schedule = FlatRateSchedule(
    volumetric_rate=.15,
    monthly_fixed_charge=5,
)

# COMMAND ----------

df_large = generate_large_load_profile(num_buildings=N_BUILDINGS)

# COMMAND ----------

rate = ModeledRate(load_profile_df=df_large, rate_schedule=rate_schedule)

# COMMAND ----------

x = rate.calculate_monthly_bills()
print(x.shape)
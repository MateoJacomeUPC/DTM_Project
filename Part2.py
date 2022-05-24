import pandas as pd
import geopandas as gpd
import psycopg2 as psy
import matplotlib.pyplot as plt
import seaborn as sns


############
###### Load Files
############

### ais
ais = pd.read_csv("data_part2/ais.csv")
ais.columns = ais.columns.str.lower()
ais["speed"] = ais["speed"].str.replace(",", ".").astype("float")
ais["date"] = pd.to_datetime(ais["date"])
ais = gpd.GeoDataFrame(ais, geometry=gpd.points_from_xy(ais.longitude, ais.latitude)
                       ).drop(["longitude", "latitude"], axis=1)

### fishing
fishing = pd.read_csv("data_part2/fishing.csv")
fishing.columns = fishing.columns.str.lower()
fishing["date"] = pd.to_datetime(fishing["day"])
fishing = gpd.GeoDataFrame(fishing, geometry=gpd.points_from_xy(fishing.longitude, fishing.latitude)
                           ).drop(["day", "longitude", "latitude"], axis=1)

### temperature
temperature = pd.read_csv("data_part2/temperature_012020-062020_res05.csv")
temperature.columns = temperature.columns.str.lower()
temperature["time"] = pd.to_datetime(temperature["time"])
temperature["temperature"] = temperature["temperature"] - 273.15
temperature.loc[temperature["partoftheday"] == "day", "time"] = \
    temperature.loc[temperature["partoftheday"] == "day", "time"] + pd.Timedelta(hours=12)
temperature = gpd.GeoDataFrame(temperature, geometry=gpd.points_from_xy(temperature.longitude, temperature.latitude)
                               ).drop(["partoftheday", "longitude", "latitude"], axis=1)

### chlorophyll
chlorophyll = pd.read_csv("data_part2/chlorophylla_012020-062020_res05.csv")
chlorophyll.columns = chlorophyll.columns.str.lower()
chlorophyll["time"] = pd.to_datetime(chlorophyll["time"])
chlorophyll = gpd.GeoDataFrame(chlorophyll, geometry=gpd.points_from_xy(chlorophyll.longitude, chlorophyll.latitude)
                               ).drop(["longitude", "latitude"], axis=1)


############
###### Data cleaning
############

###### 1.A - Calculate % of NAs
### Calculate and report the % of missing values for temperature and chlorophyll data sets.
temp_NAs = temperature["temperature"].isna().sum()
temp_NAs_pct = round(temp_NAs / temperature.shape[0] * 100, 2)
chlr_NAs = chlorophyll["chlor_a"].isna().sum()
chlr_NAs_pct = round(chlr_NAs / chlorophyll.shape[0] * 100, 2)

NA_df = pd.DataFrame({"Variable": ["Temperature", "Chlorophyll_a"], "% of NAs": [temp_NAs_pct, chlr_NAs_pct]})
print("NAs per variable, including land points")
print(NA_df, "\n")

### Land points are always NA. Get those and remove them!
# get number of NAs per raster point
temp_NAs_per_point = temperature.drop('geometry', 1).isna().groupby(temperature.geometry,
                                                                    sort=False).sum().reset_index()
# get total number of timepoints and compile all points with that exact number of NAs (if always NA it's land)
temp_unique_times = len(temperature["time"].unique())
chlr_unique_times = len(chlorophyll["time"].unique())
land_geom = temp_NAs_per_point[temp_NAs_per_point["temperature"] == temp_unique_times]["geometry"]

# create a true/false land mask for the 1200 point raster, repeat it once per time point
land_mask = temperature["geometry"][:1200].intersects(land_geom)
land_repeated_mask_temp = pd.concat([land_mask for i in range(temp_unique_times)]).reset_index(drop=True)
land_repeated_mask_chlr = pd.concat([land_mask for i in range(temp_unique_times)]).reset_index(drop=True)

# mark temperature and chlorophyll rows as land or not
temperature.loc[land_repeated_mask_temp == True, "land"] = True
temperature.loc[land_repeated_mask_temp == False, "land"] = False
chlorophyll.loc[land_repeated_mask_chlr == True, "land"] = True
chlorophyll.loc[land_repeated_mask_chlr == False, "land"] = False

# keep clean copy for later (just in case)
temperature_copy = temperature.copy()
chlorophyll_copy = chlorophyll.copy()

# plot land to check if it looks good. seems so.
chlorophyll[:1200].plot(c=chlorophyll["land"])
plt.show()

# re-calculate NAs only over the sea
sea_temp_NAs = temperature[temperature["land"] == False]["temperature"].isna().sum()
sea_temp_NAs_pct = round(sea_temp_NAs / temperature.shape[0] * 100, 2)
sea_chlr_NAs = chlorophyll[chlorophyll["land"] == False]["chlor_a"].isna().sum()
sea_chlr_NAs_pct = round(sea_chlr_NAs / chlorophyll.shape[0] * 100, 2)

sea_NA_df = pd.DataFrame({"Variable": ["Temperature", "Chlorophyll_a"], "% of NAs": [sea_temp_NAs_pct, sea_chlr_NAs_pct]})
print("NAs per variable, excluding land points")
print(sea_NA_df, "\n")


###### 1.B. Imputing missing values
# keep track of which values were NAs for future reference
temperature["wasNA"] = temperature["temperature"].isna().astype(int)
chlorophyll["wasNA"] = chlorophyll["chlor_a"].isna().astype(int)

# get the rows belonging to points in the sea (we'll impute only those),
# then get the list of sea points for a timepoint's raster (so, get all the raster's spatial points that belong to sea).
sea_temperature = temperature.loc[land_repeated_mask_temp == False]
n_sea_points = len(land_mask[land_mask == False])
sea_points = [(index, row["geometry"]) for index, row in sea_temperature[:n_sea_points].iterrows()]

### Temperature
# iterate through the sea points and make one timeseries per raster point with the temperature
for index, row in sea_points:
    point_indices = [index + 1200 * i for i in range(temp_unique_times)]

    timeseries = sea_temperature.loc[point_indices, ["time","temperature"]].set_index("time")
    imp_timeseries = timeseries["temperature"].interpolate(method='index')
    imp_timeseries.set_axis(point_indices, inplace=True)
    imp_timeseries.fillna(method="bfill", inplace=True)
    sea_temperature.loc[point_indices, "temperature"] = imp_timeseries

temperature.loc[land_repeated_mask_temp == False] = sea_temperature

### Save all the temperature rasters. Create two versions of each timepoint: one with a visual cue for imputed points,
### and one without it. Commented out because slow. I'll attach some zip files / GIF animations with the results.
### Disclaimer: these raster images and the others below were generated before integrating the boat data, so the raster
### pixels are off by (0.25 , 0.25). They are corrected later for the integration of AIS/fishing and temp/chlr data.

# for i in range(temp_unique_times):
#     date = str(temperature.loc[i*1200, "time"]).replace(":","")
#     print("temperature ", date)
#
#     df = temperature[1200*i:1200*i+1200]
#
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(111)
#     img = ax.scatter(x=df.geometry.x, y=df.geometry.y, c=df["temperature"], cmap='plasma', marker=',', s=80, vmin=0,
#                      vmax=27)
#     df = df[df["land"] == True].copy()
#     ax.scatter(x=df.geometry.x, y=df.geometry.y, c="beige", marker=',', s=80)
#     plt.colorbar(img)
#     plt.title(date)
#     plt.savefig("timeseries/temperature/temperature " + date + ".jpg")
#
#     temperature[temperature["wasNA"] == 1 ].loc[1200*i:1200*i+1200].plot(c="r", ax=ax, markersize=8)
#     ax.scatter(x=df.geometry.x, y=df.geometry.y, c="beige", marker=',', s=80)
#
#     plt.savefig("timeseries/temperature_redImputation/temperature_redImp " + date + ".jpg")
#     plt.close()


### chlorophill
# get the rows belonging to points in the sea, then get the list of sea points of a single timepoint raster.
sea_chlorophyll = chlorophyll.loc[land_repeated_mask_chlr == False]

# iterate through the sea points and make one timeseries per raster point with the chlorophyll
for index, row in sea_points:
    point_indices = [index + 1200 * i for i in range(chlr_unique_times)]

    timeseries = sea_chlorophyll.loc[point_indices, ["time","chlor_a"]].set_index("time")
    imp_timeseries = timeseries["chlor_a"].interpolate(method='index')
    imp_timeseries.set_axis(point_indices, inplace=True)
    imp_timeseries.fillna(method="bfill", inplace=True)
    sea_chlorophyll.loc[point_indices, "chlor_a"] = imp_timeseries

chlorophyll.loc[land_repeated_mask_chlr == False] = sea_chlorophyll

### Save all the chlorophyll rasters. Create two versions of each timepoint: one with a visual cue for imputed points,
### and one without it. Commented out because slow. I'll attach some zip files / GIF animations with the results.

# for i in range(chlr_unique_times):
#     date = str(chlorophyll.loc[i*1200, "time"]).replace(":","")
#     print("chlorophyll ", date)
#
#     df = chlorophyll[1200*i:1200*i+1200]
#
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(111)
#     img = ax.scatter(x=df.geometry.x, y=df.geometry.y, c=df["chlor_a"], cmap='BuGn', marker=',', s=80, vmin=0,
#                      vmax=18)
#     df = df[df["land"] == True].copy()
#     ax.scatter(x=df.geometry.x, y=df.geometry.y, c="goldenrod", marker=',', s=80)
#     plt.colorbar(img)
#     plt.title(date)
#     plt.savefig("timeseries/chlorophyll/chlorophyll " + date + ".jpg")
#
#     chlorophyll[chlorophyll["wasNA"] == 1 ].loc[1200*i:1200*i+1200].plot(c="r", ax=ax, markersize=8)
#     ax.scatter(x=df.geometry.x, y=df.geometry.y, c="goldenrod", marker=',', s=80)
#
#     plt.savefig("timeseries/chlorophyll_redImputation/chlorophyll_redImp " + date + ".jpg")
#     plt.close()


###### 1.C. Reduce the frequency of AIS
# create a list with the 4 daily timepoints that will be kept. use ".hour" to only get the hours.
filter_times = pd.to_datetime(['00:03:00','06:03:00','18:03:00' ,'12:03:00']).hour
ais[ais["date"].dt.hour.isin(filter_times)]

# iterate through all the boats in ais to create a new dataframe with the resampled times.
boat_names = ais["boatname"].unique()
boat_dfs = []
for boat in boat_names:
    df = ais.loc[ais["boatname"] == boat]       # create partial dataframe with a filter by boatname
    df["old_index"] = df.index                  # keep old index
    df.set_index("date", inplace=True)          # reset index to date (required for .resample())
    mask = df.index.duplicated()                # create a mask for all duplicate timepoints.
    df = df[~mask].resample("6H").bfill()       # resample the report time to 6 hours, get the closest point going backwards
    df["date"] = df.index                       # undo the index change
    df.set_index("old_index", inplace=True, )   # undo the index change
    boat_dfs.append(df)                         # append partial dataframe to list

# reconstruct the resampled ais dataframe from list of partial dataframes
ais_resampled = pd.concat(boat_dfs).sort_index()
ais_resampled.index.name = None
ais_resampled["month"] = ais_resampled["date"].dt.to_period("M").astype("str")

### Save a raster of the boat trajectories over the chlorophyll rasters. Only for the first 25 days.
### Commented out because slow. I'll attach some zip files / GIF animations with the results.

# boat_colors = {"Korbin":"blue", "Rey":"red","Rodney":"orange","Armani":"pink","Mason":"white"}
# for i in range(chlr_unique_times):
#     date = str(chlorophyll.loc[i*1200, "time"]).replace(":","")
#     timestamp = chlorophyll.loc[i*1200, "time"]
#     print("barquitos ", date)
#
#     df = chlorophyll[1200*i:1200*i+1200]
#
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(111)
#     img = ax.scatter(x=df.geometry.x, y=df.geometry.y, c=df["chlor_a"], cmap='BuGn', marker=',', s=80, vmin=0,
#                      vmax=18)
#     df = df[df["land"] == True].copy()
#     ax.scatter(x=df.geometry.x, y=df.geometry.y, c="goldenrod", marker=',', s=80)
#     plt.colorbar(img)
#     plt.title(date)
#
#     for i in range(4):
#         barquitos = ais_resampled.loc[ais_resampled["date"] == timestamp - pd.Timedelta(hours=6*i)]
#         ax.scatter(x=barquitos.geometry.x, y=barquitos.geometry.y, c=barquitos["boatname"].map(boat_colors), s=120,
#                lw = 1, edgecolor="black", alpha=1/(i*2+1))
#         if i == 0:
#             for i, txt in enumerate(barquitos["boatname"]):
#                 ax.annotate(txt, (barquitos[barquitos["boatname"] == txt].geometry.x,
#                                   barquitos[barquitos["boatname"] == txt].geometry.y))
#
#     plt.savefig("timeseries/barquitos/barquitos " + date + ".jpg")
#     plt.close()
#
#     if date == '2020-01-25 000000':
#         break

#############
###### Data unification
#############

# transform temperature and chlorophyll dataframes by shifting points 0.25 degrees to the top right,
# then adding a square (cap_style=3) buffer of 0.25 "radius".
grid_temperature = temperature.copy()
grid_temperature.geometry = temperature.translate(xoff=0.25, yoff=0.25).buffer(0.25, cap_style=3)

grid_chlorophyll = chlorophyll.copy()
grid_chlorophyll.geometry = chlorophyll.translate(xoff=0.25, yoff=0.25).buffer(0.25, cap_style=3)

#############
###### Data integration
#############

# 3.A. Joining fishing data with temperature and chlorophyll data
# first, iterate by date in the fishing dataset.
fish_joins_dict = {}
for date in fishing["date"].unique():
    fish_df = fishing[fishing["date"] == date]                      # get a partial fishing dataset for the given date
    temp_df = grid_temperature[grid_temperature["time"] == date]    # get a partial temp dataset
    chlr_df = grid_chlorophyll[grid_chlorophyll["time"] == date]    # get a partial chlr dataset
    temp_df.columns = ['temp_' + i if i != "geometry" else "geometry" for i in temp_df.columns] # rename columns for traceability
    chlr_df.columns = ['chlr_' + i if i != "geometry" else "geometry" for i in chlr_df.columns] # rename columns
    first_join = fish_df.sjoin(temp_df, rsuffix = "temp")                   # join fish data with temp data
    fish_joins_dict[date] = first_join.sjoin(chlr_df, rsuffix = "chlr")     # join chlr data too, this time into the dictionary to later concat the partial DFs.
    fish_joins_dict[date].drop(['temp_time','chlr_time','temp_land', 'chlr_land'], inplace=True, axis=1) # drop redundant columns
    fish_joins_dict[date]["temp_temperature"] = fish_joins_dict[date]["temp_temperature"] - 273.15 # integrate temperature scales

# concat partial dictionaries
fishing_plus = pd.concat(fish_joins_dict.values())

# fishing_plus has more rows than fishing because boats can be in the border between two squares in the grid.
# I'll get the first one appearing in the df for simplicity. It'd be better to group by and get the average of the two fields.
fishing_plus = fishing_plus[~fishing_plus.index.duplicated(keep='first')]
fishing_plus["boat_trip"] = fishing_plus["boatname"] + "_" + fishing_plus["trip"]
fishing_plus["month"] = fishing_plus["date"].dt.to_period("M").astype("str")

### Loading tables to MobilityDB
conn = psy.connect("host=dtim.essi.upc.edu port=5432 dbname=dbmateojacome user=mateo.jacome password=***REMOVED***")
cur = conn.cursor()

# Load AIS_resampled to MobilityDB. Execution commented out because already created
q_ais_resampled_creation = '''create table ais_resampled(boatname varchar, boatid int, date date,
                                                                time timestamptz, speed varchar, course varchar,
                                                                geom geography(POINT,4326), month varchar, 
                                                                     primary key(boatname, geom));'''

# cur.execute(q_ais_resampled_creation)

q_fishing_plus_creation = '''create table fishing_plus(boatname varchar, boatid varchar, trip varchar, kg int, 
                                                        duration int, lines int, temperature float, date date,
                                                        geometry geography(POINT,4326), index_temp int,
                                                         temp_temperature float, temp_wasNA int, index_chlr int, 
                                                         chlr_chlor_a float, chlr_wasNA int, boat_trip varchar, 
                                                         month varchar, primary key(boatname, date));'''
# cur.execute(q_fishing_plus_creation)
# conn.commit()

### import the rows into the DB, encoding the geom data during the transfer and then decoding it
import sqlalchemy as sal
# Function to generate WKB hex
def wkb_hexer(line):               # modified from Stackexchange
    return line.wkb_hex
# Convert `'geom'` column in GeoDataFrame `gdf` to hex
    # Note that following this step, the GeoDataFrame is just a regular DataFrame
    # because it does not have a geometry column anymore. Also note that
    # it is assumed the `'geom'` column is correctly datatyped.
ais_resampled['geometry'] = ais_resampled['geometry'].apply(wkb_hexer)
fishing_plus['geometry'] = fishing_plus['geometry'].apply(wkb_hexer)
# Create SQL connection engine
engine = sal.create_engine('postgresql://mateo.jacome:***REMOVED***@dtim.essi.upc.edu:5432/dbmateojacome')
# Connect to database using a context manager
with engine.connect() as conn, conn.begin():
    # Note use of regular Pandas `to_sql()` method.
    ais_resampled.to_sql("ais_resampled", con=conn, schema="public",
               if_exists='replace', index=False)
    fishing_plus.to_sql("fishing_plus", con=conn, schema="public",
                         if_exists='replace', index=False)

    # Convert the `'geom'` column back to Geometry datatype, from text
    sql1 = """ALTER TABLE public.ais_resampled
               ALTER COLUMN geometry TYPE Geometry(POINT, 4326)
                 USING ST_SetSRID(geometry::Geometry, 4326)"""
    conn.execute(sql1)

    sql2 = """ALTER TABLE public.fishing_plus
               ALTER COLUMN geometry TYPE Geometry(POINT, 4326)
                 USING ST_SetSRID(geometry::Geometry, 4326)"""
    conn.execute(sql2)

# transform the geometries into time-geometries. Execution commented out because tables already created
with engine.connect() as conn, conn.begin():
    q_ais_resampled_tgeom_creation = '''create table ais_resampled_tgeom(boatname varchar, boatid int, tloc tgeompoint,	
                                                                         month varchar, speed float, course varchar, 
                                                                         primary key(boatname, tloc));'''
    # conn.execute(q_ais_resampled_tgeom_creation)
    # Load rows from the temporal table into the persistent one, modifying spatiotemporal data to tgeompoint
    q_ais_resampled_tgeom_insertion = '''
    insert into ais_resampled_tgeom(boatname, boatid, tloc, speed, course, month)
    select boatname, boatid, tgeompoint_inst(geometry, date),
        cast(speed as float) as speed, course, month
    from ais_resampled;'''
    # conn.execute(q_ais_resampled_tgeom_insertion)

############
###### Querying
############

### 4.A. Compare the trajectories of fishing and ais data sets (before any preprocessing) and comment on their difference.

# get the frequency of the AIS and fishing tables to compare them.
reporting_rates_df = pd.DataFrame()
for boat in boat_names:
    ais_df = ais[ais["boatname"] == boat]
    fishing_df = fishing[fishing["boatname"] == boat]

    total_ais_days = max(ais_df["date"].dt.date.unique()) - min(ais_df["date"].dt.date.unique())
    total_fishing_days = max(fishing_df["date"].dt.date.unique()) - min(fishing_df["date"].dt.date.unique())

    ais_rate = len(ais_df["date"].unique()) / total_ais_days.days
    fishing_rate = len(fishing_df["date"].unique()) / total_fishing_days.days

    reporting_rates_df.loc[boat, "ais_daily_reports (avg)"] = ais_rate
    reporting_rates_df.loc[boat, "fishing_daily_reports (avg)"] = fishing_rate
    reporting_rates_df.loc[boat, "total_ais_dates"] = int(total_ais_days.days)
    reporting_rates_df.loc[boat, "total_fishing_dates"] = int(total_fishing_days.days)

reporting_rates_df["total_ais_dates"] = reporting_rates_df["total_ais_dates"].astype(int)
reporting_rates_df["total_fishing_dates"] = reporting_rates_df["total_fishing_dates"].astype(int)

print("Comparison of AIS and fishing reporting frequencies:")
print(reporting_rates_df, "\n")
# Fishing data is reported at maximum at a daily rate, although boats don't report anything in days where they
# don't go out fishing. The average reporting rates are thus around 0.6-0.8 reports per day depending on the boat.
# AIS data is reported several times per hour, although there can be gaps in the reports. The average
# reporting rates are between 90 and 105 reports per day approximately, which is approx. a report every 15 minutes.


###### 4.B.1. What is the distance travelled by each vessel, per month.
### send the query to MobilityDB, since we can only query trajectory distances there
with engine.connect() as conn, conn.begin():
    q_monthly_kms = """
    SELECT boatname, month, length(tgeompoint_seq(array_agg(tloc))) as monthly_kms
    FROM (SELECT * FROM ais_resampled_tgeom ORDER BY getTimestamp(tloc)) as ais_resampled_tgeom
    GROUP BY boatname, month
    ORDER BY boatname, month"""
    monthly_distances = pd.read_sql(q_monthly_kms, conn)
    print("Monthly distances traveled by each boat, in degrees")
    print(monthly_distances, "\n")

###### 4.B.2. What is the quantity of fish (in kg) caught by each vessel, per month.
with engine.connect() as conn, conn.begin():
    q_monthly_kgs = """
    SELECT boatname, to_char(date, 'YYYY-MM') as month , sum(kg) as monthly_kgs
    FROM fishing_plus
    GROUP BY boatname, to_char(date, 'YYYY-MM')
    ORDER BY boatname, to_char(date, 'YYYY-MM')"""
    monthly_catches = pd.read_sql(q_monthly_kgs, conn)
    print("Monthly catches by each boat, in kg")
    print(monthly_catches)

###### 4.B.3. Find Correlation between the quantity of fish caught and the temperature/chlorophyll.
# Tip: You can use DataFrame functionality to calculate correlations.
corr = fishing_plus[["kg","duration","lines","temperature","temp_temperature","chlr_chlor_a"]].corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap="vlag", vmin=-1, vmax=1)
plt.title("Fishing Correlation plot")
plt.show()
# none of the variables shows correlation with the target variable (catches)


###### Misc
# create gifs from all the timeseries visualizations
import glob
from PIL import Image

for timeseries in ["temperature", "chlorophyll", "barquitos", "chlorophyll_redimputation", "temperature_redImputation"]:
    # filepaths
    fp_in = "timeseries/{timeseries}/*.jpg".format(timeseries=timeseries)
    fp_out = "gifs/{timeseries}.gif".format(timeseries=timeseries)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
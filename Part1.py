import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import csv
import numpy as np

####
#### This is a deprecated script since the visualizations produced in it were improved in Part2.py
####

# conn = psy.connect("host=dtim.essi.upc.edu port=5432 dbname=DBmateo.jacome user=mateo.jacome password=DMT2022!")
# cur = conn.cursor()

# Create table
# cur.execute("""CREATE TABLE temporal_chlorophyll(
#     time timestamp,
#     latitude float,
#     longitude float,
#     chlor_a float)""")
# conn.commit()


# def execute_many(conn, df, table):
#     """
#     Using cursor.executemany() to insert the dataframe
#     """
#     # Create a list of tupples from the dataframe values
#     tuples = [tuple(x) for x in df.to_numpy()]
#     # Comma-separated dataframe columns
#     cols = ','.join(list(df.columns))
#     # SQL quert to execute
#     query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s)" % (table, cols)
#     cursor = conn.cursor()
#     try:
#         cursor.executemany(query, tuples)
#         conn.commit()
#     except (Exception, psy.DatabaseError) as error:
#         print("Error: %s" % error)
#         conn.rollback()
#         cursor.close()
#         return 1
#     print("execute_many() done")
#     cursor.close()
#

### Load chlorophyll to DB:
# csv file has 31.449.601: header +  182 days x 480 lon x 360 lat.  One day has 172.800 rows
# raster limits: lat [-49.97916666666666, -35.02083333333333], lon [-50.02083333333334, -69.97916666666667']
# boats are bound: -55 to -65 long,  -39.5, -49.5  lat

# df_iter = pd.read_csv('data_part1/chlorophyll.csv', chunksize = 172800, sep=',') # file has 31.449.601 lines: 182 days, 480x360
# sql_chlorophyll = 'INSERT INTO temporal_chlorophyll(time, latitude, longitude, chlor_a) values (%s,%s,%s,%s)'
# for i, df in enumerate(df_iter):
#     # plt.scatter(x=df['longitude'], y=df['latitude'], c=df['chlor_a'], cmap='Oranges')
#     # plt.show()
#
#     array = np.reshape(df['chlor_a'].to_numpy(), newshape=(360, 480))
#
#     plt.imshow(array, cmap = 'Greens')
#     plt.show()


    # tuples = []
    # for index, row in df.iterrows():
    #     tuples.append(row if row[3] != '' else row[:-1] + [None])
    # cur.executemany(sql_chlorophyll, tuples)
    #
    # try:
    #     conn.commit()
    #     print('Segment {}/315 was succesfully loaded'.format(i))
    # except:
    #     print('Something went wrong while trying to load segment {}'.format(i))
    #     sys.exit(1)


### temperature  43680001
# csv file has 43.680.001: header +  182 days x 400 lon x 300 lat x 2 measurements.  One measurement has 120.000 rows
# raster limits: lat [-49.975, -35.025], lon [-50.025, -69.975']
# boats are bound: -55 to -65 long,  -39.5, -49.5  lat
with open('data_part1/temperature.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row.
    for i, line in enumerate(reader):
        if i == 800*300-1:
            print(line)
        if i == 800*300:
            print(line)
            break


boats = pd.read_csv('data_part1/fishing.csv', sep=',', )
ais = pd.read_csv('data_part1/ais.csv', sep=',')
ais['Date'] = pd.to_datetime(ais['Date'])
ais = ais[ais['Date'].dt.hour == 12]


boat_tracks = ais[['BoatName', 'Date','Latitude', 'Longitude']]
# boat_tracks['Date'] = pd.to_datetime(boat_tracks['Date'])
boat_colors = {'Mason':'yellowgreen', 'Rey':'darkolivegreen', 'Korbin':'maroon', 'Armani':'navy',
               'Rodney':'darkslategray'}

boat_colors = {'Mason':(255, 0, 0), 'Rey':(255, 0, 0), 'Korbin':(255, 0, 0), 'Armani':(255, 0, 0),
               'Rodney':(255, 0, 0)}

lon_lims_temp, lat_lims_temp = [-69.975,-50.025], [-49.975, -35.025]
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
argentina = world[world['name']=='Argentina']


df_temperature_iter = pd.read_csv('data_part1/temperature.csv', chunksize = 120000, sep=',')
sql_temperature = 'INSERT INTO temporal_chlorophyll(time, latitude, longitude, chlor_a) values (%s,%s,%s,%s)'
for i, df in enumerate(df_temperature_iter):
    # plt.scatter(x=df['longitude'], y=df['latitude'], c=df['chlor_a'], cmap='Oranges')
    # plt.show()
    print(df['temperature'].isna().sum() * 100 / len(df['temperature']))

    array = np.reshape(df['temperature'].to_numpy(), newshape=(300, 400))

    # plt.imshow(array, cmap = 'plasma', vmin=273, vmax=298)
    # plt.show()

    date = pd.to_datetime(df['time'].unique()[:9])[0]
    temp_boats = boat_tracks[boat_tracks['Date'] <= date]
    print(date)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    argentina.plot(ax=ax, color='beige')
    ax.set_facecolor('slateblue')

    ax.set_xlim(lon_lims_temp)
    ax.set_ylim(lat_lims_temp)
    img = ax.scatter(x=df['longitude'], y=df['latitude'], c=df['temperature'], cmap='plasma', marker=',',lw=0, s=1)
    temp_boats.groupby('BoatName').plot(x='Longitude', y='Latitude', ax=ax)
    # ax.plot(boat_tracks[boat_tracks['Day'] <= date]['Longitude'], boat_tracks[boat_tracks['Day'] <= date]['Latitude'],
    #         c = boat_tracks[boat_tracks['Day'] <= date]['BoatName'].map(boat_colors))
    ax.set_title("Day Temperature on 2020-01-01")
    fig.colorbar(img, ax=ax, orientation='horizontal')
    fig.tight_layout()
    plt.show()
    # fig.savefig('plot.png', dpi=fig.dpi)


######BOATS











#######/boats


# IMPLEMENT STUFF IN A CHUNKWISE WAY SO THAT EVERY X ROWS WE COMMIT KEEPING TRACK OF THE LAST COMMITED ROW.
# AFTER THAT, FOLLOW https://gis.stackexchange.com/questions/145007/creating-geometry-from-lat-lon-in-table-using-postgis
# SO THAT WE CREATE THE POINTS FROM INSIDE THE DBMS

#
with open('data_part1/chlorophyll.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row.
    for i, line in enumerate(reader):
        if i == 480*360-1:
            print(line)
        if i == 480*360:
            print(line)
            break
#
#     i = 0
#     buffer_command = ''
#     for row in reader:
#         buffer_command +=
#         ####TO DO: fix this so that it adds the string with the formatted values for each row
#         #### and then each X rows it runs the command.
#             "INSERT INTO temporal_chlorophyll VALUES (%s, %s, %s, %s)\n",
#             row if row[3] != '' else row[:-1] + [None]
#         )
#         i += 1
#         if i % 5000 == 0:
#             print(i)
#             cur.execute()
#     conn.commit()




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt
from pandas.io.json import json_normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#####################


def load_data():

	# Loads the data from the csv files and returns them as one large dataframe.

	# First import the csv files

	q1 = pd.read_csv('2017-q1_trip_history_data.csv')

	q2 = pd.read_csv('2017-q2_trip_history_data.csv')

	# There seemed to be some difficulty importing the data for q3 as the columns start 
	# station number and end station number contained mixed types.

	q3 = pd.read_csv('2017-q3_trip_history_data.csv', dtype = {'Start station number' : str, 'End station number': str})

	q4 = pd.read_csv('2017-q4_trip_history_data.csv')

	# Create a list of quarters entitles 'frames':

	frames = [q1, q2, q3, q4]

	# Concatenate rows to create data frame for entire year:

	raw_df = pd.concat(frames).reset_index(drop=True)

	# Pickle raw data for easy access later

	raw_df.to_pickle('raw_data.pkl')
	
	return raw_df 

####################

def clean_and_tidy():

	bike_data = pd.read_pickle('raw_data.pkl')

	# Cleans and tidies the dataframe bike_data, returning trips between stations in the DC area
	# as well as an extra column - 'Duration (mins)'. Also
	# converts member/casual column into boolean integers - 0 for casual 1 for member.

	# Firstly, delete entries that include the warehouse, since although they say on the site 
	# that the data doesn't contain the entries for the warehouse this clearly isn't true.

	bike_data = bike_data[~((bike_data['Start station'].str.contains('Warehouse')) | (bike_data['End station'].str.contains('Warehouse')))]
	
	# Replace some street names that are actual stations but aren't found by Google V3 geocoder:
	
	streets_to_change = {'Virginia Square Metro / ' : '', 
						 '/DOL': '', 
						 'Shirlington Transit Center / ' : '', 
						 'Silver Spring Metro/':'', 
						 'Columbus Ave & Gramercy Blvd': 'Capital bikeshare, Columbus Avenue, Derwood, MD 20855', 
						 ' / Oxon Run Trail': '', 
						 'Columbus Circle / Union Station': 'Columbus Circle Northeast', 
						 'Anacostia Metro':'2431 Shannon Pl SE', 
						 'Clarendon Blvd & Pierce St': np.nan, 
						 'Rhode Island Ave Metro':'2300 Washington Pl NE',
						 'Congress Heights Metro':'1321 Alabama Ave SE', 
						 'Rockville Metro East': np.nan, 
						 'Rockville Metro West': np.nan, 
						 '1st & H St NW': '801-840 1st NW', 
						 '10th & G St NW': '1020 G St NW', 
						 'Van Ness Metro / UDC': '4200 Connecticut Ave NW',
						 'McKinley St & Connecticut Ave NW': '3807 McKinley St NW', 
						 'Massachusetts Ave & Dupont Circle NW':'Capital Bikeshare Station,  Massachusetts Ave NW',
						 '25th St & Pennsylvania Ave NW': '2503 Pennsylvania Ave NW',
						 '20th & O St NW / Dupont South' : '1401 20th St NW',
						 "Independence Ave & L'Enfant Plaza SW/DOE":"300-320 L'Enfant Plaza SW"}

	replacements = {'Start station' : streets_to_change, 'End station' : streets_to_change}

	bike_data = bike_data.replace(replacements, regex=True)

	# Get geocoded objects - I defined a function 'get_geocoded_objects' that finds all the 
	# geocoded locations using GoogleV3 and pickled the resulting dictionary with the name 
	# 'Dict_of_geocoded_locations'. It is commented out to avoid excessively querying
	# GoogleMaps and can instead be loaded using the pickled dictionary (using the function 
	# 'load_obj()' defined toward the end of this file).

	# locations = get_geocoded_objects(bike_data)

	locations_dict = load_obj('Dict_of_geocoded_locations')

	bike_data['Start station'] = bike_data['Start station'].apply(lambda x: address_return(x, locations_dict))
	bike_data['End station'] = bike_data['End station'].apply(lambda x: address_return(x,locations_dict))

	# Keep only those rides that start and end in Washington DC. The question asked specifically for Washington DC data, and the files hold
	# data for 5 different jurisdictions:  Washington, DC.; Arlington, VA; Alexandria, VA; Montgomery, MD, and Fairfax County, VA.

	bike_data = bike_data[ (bike_data['Start station'].str.contains('Washington, DC')) & (bike_data['End station'].str.contains('Washington, DC')) ]

	# Encode 'Member type' as a boolean integer (1 for member, 0 for casual)

	mem = {'Member': int(1), 'Casual': int(0)}

	bike_data['Member type'] = bike_data['Member type'].map(mem)

	# Ensure that 'Start station number' and 'End station number' are all numeric, those that 
	# cannot be converted to a digit are coerced into being NaNs.

	bike_data['Start station number'] = pd.to_numeric(bike_data['Start station number'], errors = 'coerce')
	bike_data['End station number'] = pd.to_numeric(bike_data['End station number'], errors = 'coerce')

	# Ensure that 'Start date' and 'End date' entries are datetime objects, if an entry cannot 
	# be converted to a datetime it is coerced into being a NaN.

	bike_data['Start date'] = pd.to_datetime(bike_data['Start date'], errors = 'coerce')
	bike_data['End date'] = pd.to_datetime(bike_data['End date'], errors = 'coerce')

	# Remove column 'Bike number' which I do not think is particularly useful.

	bike_data = bike_data.drop(['Bike number'], axis = 1)

	# Add column for duration of ride in minutes.

	bike_data['Duration (mins)'] = bike_data['Duration (ms)']/60000

	# Remove any rows that contain troublesome NaNs

	bike_data = bike_data.dropna()

	# Pickle data for easy access later

	bike_data.reset_index(drop=True).to_pickle('clean_tidy_bike.pkl')

	return bike_data.reset_index(drop=True)

#####################

def ride_time_box():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Generates boxplot of ride times across the whole year, omitting outliers, and saves 
	# as a png.

	bike_data.boxplot('Duration (mins)', showfliers=False, showmeans=True)

	# Plot options

	plt.ylabel('Time in Minutes')
	plt.xlabel('')
	plt.title('Box plot of ride time over the whole of 2017')
	
	# Save plot

	plt.savefig('RideTimeBoxPlot.png')

	# Show plot

	plt.show()
	
	return

#####################

def round_trip_pie_chart():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Produces pie chart of round trips vs. non-round trips and saves in current directory as png

	# Set label names

	labels = 'Round trips', 'Not round trips'

	# Get number of round trips

	num_round_trips = len(bike_data[bike_data['Start station']==bike_data['End station']])
	sizes = [num_round_trips, len(bike_data)-num_round_trips]
	
	# Set colours

	colors = ['yellowgreen', 'lightskyblue']
	explode = (0.1, 0)
	
	# Plot pie chart

	plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
	plt.axis('equal')

	# Save plot

	plt.savefig('RoundTrips.png')

	# Show plot

	plt.show()

	return

###################

def members_casual_ride_time():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Produces two boxplots of ride time over the whole year for members and non-members
	# and saves as png

	# Plot data

	bike_data.boxplot(column=['Duration (mins)'], by='Member type', showmeans=True, showfliers=False)
	
	# Plot options

	plt.xticks([0,1,2],['', 'Casual', 'Member'])
	plt.xlabel('')
	plt.ylabel('Time (mins)')

	# Save plot

	plt.savefig('MemberCasualRideTimeBoxplot.png')

	# Show plot

	plt.show()

	return

##################

def usage_rates_daily():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Create two data frames, one for members, one for casual users, and group by day and count the number of trips per day.

	casual = bike_data.groupby(bike_data[bike_data['Member type'] == 0]['Start date'].dt.month).count()
	members = bike_data.groupby(bike_data[bike_data['Member type'] == 1]['Start date'].dt.month).count()

	# Amalgamate them into another dataframe

	full_data = pd.DataFrame(index=casual.index, data = {'Casual' : casual['Duration (ms)'], 'Registered' : members['Duration (ms)']})
	
	# Calculate the proportion of casual users that used scheme each day

	full_data['Cas_prop'] = full_data['Casual']/(full_data['Casual'].sum())

	# Calculate the proportion of members that used scheme each day

	full_data['Reg_prop'] = full_data['Registered']/(full_data['Registered'].sum())

	# Plot data

	ax = full_data[['Cas_prop','Reg_prop']].plot(kind = 'bar')

	lines, labels = ax.get_legend_handles_labels()

	labels = ['Casual users', 'Registered users']

	plt.legend(lines, labels)

	plt.ylabel('Proportion of total rides')

	plt.xlabel('Month')

	plt.xticks(range(0,12),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

	plt.title('Daily usage over the year for members/casual users during 2017')

	plt.tight_layout()

	plt.savefig('AnnualUsage.png')

	plt.show()

	return

#################

def usage_rates_weekly():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Count trips per day for members/casual users

	casual = bike_data.groupby(bike_data[bike_data['Member type'] == 0]['Start date'].dt.dayofweek).count()
	members = bike_data.groupby(bike_data[bike_data['Member type'] == 1]['Start date'].dt.dayofweek).count()

	# Put into Dataframe

	full_data = pd.DataFrame(index=casual.index, data = {'Casual' : casual['Duration (ms)'], 'Registered' : members['Duration (ms)']})

	# Create new columns that show proportion of trips per day for casual users/members

	full_data['Proportion_Casual'] = full_data['Casual']/(full_data['Casual'].sum())
	full_data['Proportion_Registered'] = full_data['Registered']/(full_data['Registered'].sum())

	# Plot data

	ax = full_data[['Proportion_Casual','Proportion_Registered']].plot(kind='bar')

	lines, labels = ax.get_legend_handles_labels()

	labels = ['Casual users', 'Registered users']

	plt.legend(lines, labels)

	plt.ylabel('Proportion of total rides')

	plt.xticks([0,1,2,3,4,5,6], ['Mon', 'Tues', 'Weds', 'Thurs', 'Fri', 'Sat', 'Sun'])

	plt.xlabel('Day')

	plt.title('Daily usage during the week for members/casual users')

	plt.tight_layout()

	plt.savefig('DailyUsage.png')

	plt.show()

	return

##################

def usage_rate_weekday():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Keep weekday data

	bike_data = bike_data[bike_data['Start date'].dt.dayofweek <5]

	# Count number of trips started in each hour for groups

	casual = bike_data.groupby(bike_data[bike_data['Member type'] == 0]['Start date'].dt.hour).count()
	members = bike_data.groupby(bike_data[bike_data['Member type'] == 1]['Start date'].dt.hour).count()

	# Put into one large data frame

	full_data = pd.DataFrame(index=casual.index, data = {'Casual' : casual['Duration (ms)'], 'Registered' : members['Duration (ms)']})

	# Calculate proportion

	full_data['Proportion_Casual'] = full_data['Casual']/(full_data['Casual'].sum())
	full_data['Proportion_Registered'] = full_data['Registered']/(full_data['Registered'].sum())

	# Plot data

	ax = full_data[['Proportion_Casual','Proportion_Registered']].plot(kind='bar')

	lines, labels = ax.get_legend_handles_labels()

	labels = ['Casual users', 'Registered users']

	plt.legend(lines, labels)

	plt.ylabel('Proportion of total rides')

	plt.xlabel('Hour (in 24h format)')

	plt.title('Hourly usage during weekdays for members/casual users')

	plt.tight_layout()

	plt.savefig('WeekdayHourlyUsage.png')

	plt.show()

	return

###############

def usage_rate_weekend():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# SAME AS ABOVE BUT FOR THE WEEKEND

	bike_data = bike_data[bike_data['Start date'].dt.dayofweek >=5]

	casual = bike_data.groupby(bike_data[bike_data['Member type'] == 0]['Start date'].dt.hour).count()
	members = bike_data.groupby(bike_data[bike_data['Member type'] == 1]['Start date'].dt.hour).count()

	full_data = pd.DataFrame(index=casual.index, data = {'Casual' : casual['Duration (ms)'], 'Registered' : members['Duration (ms)']})
	full_data['Proportion_Casual'] = full_data['Casual']/(full_data['Casual'].sum())
	full_data['Proportion_Registered'] = full_data['Registered']/(full_data['Registered'].sum())

	ax = full_data[['Proportion_Casual','Proportion_Registered']].plot(kind='bar')

	lines, labels = ax.get_legend_handles_labels()

	labels = ['Casual users', 'Registered users']

	plt.legend(lines, labels)

	plt.ylabel('Proportion of total rides')

	plt.xlabel('Hour')

	plt.title('Hourly usage during weekends for members/casual users')

	plt.tight_layout()

	plt.savefig('WeekendHourlyUsage.png')

	plt.show()

	return

###############

def ride_time_monthly():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Produces boxplots of ride times per month and saves it as a png

	# Create column with entries that are the month in which each ride took place

	bike_data['Month'] = bike_data['Start date'].apply(lambda x: x.month)

	# Creates plot

	bike_data.boxplot(column=['Duration (mins)'], by = 'Month', showmeans=True, showfliers=False)
	
	# Plot options

	plt.ylabel('Time (mins)')

	
	# Saves figure as png

	plt.savefig('MonthlyRideTimes.png')

	# Show plot

	plt.show()

	return

################

def ride_time_monthly_members_casual():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Generates boxplots of ride times per month for members/non-members and saves plot
	# as a png file

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Creates column which corresponds to month of each ride

	bike_data['Month'] = bike_data['Start date'].apply(lambda x: x.month)

	# Generates plot

	(bike_data.replace({'Member type': {0: 'Casual',1:'Member'}})
			.groupby('Member type')
			.boxplot(column=['Duration (mins)'], by = 'Month', showmeans=True, showfliers=False))
	
	# Plot options

	plt.ylabel('Time in minutes')
	
	# Saves plot as png

	plt.savefig('MonthlyCasMemRideTimes.png')

	# Show plot

	plt.show()
	
	return

################

def time_limits():

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Generates pie chart of proportion of rides that exceeded time limit.

	# Set labels for pie portions

	labels = 'Rides exceding time limit', 'Rides within time limit'
	
	# Counts number of rides that exceeded 30 mins

	num_excedences = len(bike_data[bike_data['Duration (ms)']> 30*60*1000])

	# Sets sizes of groups for pie portions

	sizes = [num_excedences, len(bike_data)-num_excedences]
	
	# Pie chart settings

	colors = ['yellowgreen', 'lightskyblue']
	explode = (0.1, 0)

	# Generate pie chart

	plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
	plt.axis('equal')
	
	# Save figure to current directory

	plt.savefig('TimeLimitExcedences.png')

	# Show plot

	plt.show()

	return

#################

def get_lat_long():

	# De-pickle

	bike_data = pd.read_pickle("clean_tidy_bike.pkl")

	# Load a pickled dictionary called 'Dict_of_geocoded_locations.pkl', obtained using the function
	# 'get_geocoded_locations()' below, which makes use of the GoogleV3 geocoder

	location_dict = load_obj('Dict_of_geocoded_locations')

	# Create dictionary with google addresses as keys and latitudes as values

	lat_dict = {v.address : v.latitude for k,v in location_dict.items() if v is not None}

	# Create dictionary with google addresses as keys and longitudes as values

	long_dict = {v.address : v.longitude for k,v in location_dict.items() if v is not None}

	# Create new columns for starting and ending latitudes and longitudes

	bike_data['Start_lat'] = bike_data['Start station'].map(lat_dict)
	bike_data['Start_long'] = bike_data['Start station'].map(long_dict)
	bike_data['End_lat'] = bike_data['End station'].map(lat_dict)
	bike_data['End_long'] = bike_data['End station'].map(long_dict)

	# Apply Haversine formula to calculate distance between points using the latitude and longitudes.

	bike_data['S_LAT_rad'], bike_data['S_LON_rad'], bike_data['E_LAT_rad'], bike_data['E_LON_rad'] = np.radians(bike_data['Start_lat']), np.radians(bike_data['Start_long']), np.radians(bike_data['End_lat']), np.radians(bike_data['End_long'])
	bike_data['diff_LON'] = bike_data['E_LON_rad'] - bike_data['S_LON_rad']
	bike_data['diff_LAT'] = bike_data['E_LAT_rad'] - bike_data['S_LAT_rad']
	bike_data['Distance in km'] = 6367 * 2 * np.arcsin(np.sqrt(np.sin(bike_data['diff_LAT']/2)**2 + np.cos(bike_data['S_LAT_rad']) * np.cos(bike_data['E_LAT_rad']) * np.sin(bike_data['diff_LON']/2)**2))

	# Drop unnecessary columns

	bike_data = bike_data.drop(['S_LAT_rad', 'S_LON_rad', 'E_LAT_rad', 'E_LON_rad', 'diff_LON', 'diff_LAT'], axis=1)

	# Pickle the data

	bike_data.to_pickle('bike_data_with_distances.pkl')

	return bike_data

################# TEST FROM HERE:##########

def ride_dist_box():

	bike_data = pd.read_pickle('bike_data_with_distances.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Generates boxplot of ride times across the whole year, omitting outliers, and saves 
	# as a png.

	bike_data.boxplot('Distance in km', showfliers = False, showmeans=True)

	# Plot options

	plt.ylabel('kilometres')
	plt.xlabel('')
	plt.title('Distance of rides over 2017')
	
	
	# Save plot

	plt.savefig('RideDistBoxPlot.png')

	# Show plot

	plt.show()
	
	return

#################

def members_casual_ride_dist():

	bike_data = pd.read_pickle('bike_data_with_distances.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Produces two boxplots of ride time over the whole year for members and non-members
	# and saves as png

	# Plot data

	bike_data.boxplot(column=['Distance in km'], by='Member type', showfliers = False, showmeans=True)
	
	# Plot options

	plt.xticks([0,1,2],['', 'Casual', 'Member'])
	plt.xlabel('')
	plt.ylabel('Distance (km)')
	

	# Save plot

	plt.savefig('MemberCasualRideDistBoxplot.png')

	# Show plot

	plt.show()

	return

####################

def ride_dist_monthly():

	bike_data = pd.read_pickle('bike_data_with_distances.pkl')

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Produces boxplots of ride times per month and saves it as a png

	# Create column with entries that are the month in which each ride took place

	bike_data['Month'] = bike_data['Start date'].apply(lambda x: x.month)

	# Creates plot

	bike_data.boxplot(column=['Distance in km'], by = 'Month', showmeans=True, showfliers=False)
	
	# Plot options

	plt.ylabel('Distance (km)')
	
	
	# Saves figure as png

	plt.savefig('MonthlyRideDistances.png')

	# Show plot

	plt.show()

	return

################

def ride_dist_monthly_members_casual():

	bike_data = pd.read_pickle('bike_data_with_distances.pkl')

	# Generates boxplots of ride times per month for members/non-members and saves plot
	# as a png file

	# Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()

	# Creates column which corresponds to month of each ride

	bike_data['Month'] = bike_data['Start date'].apply(lambda x: x.month)

	# Generates plot

	(bike_data.replace({'Member type': {0: 'Casual',1:'Member'}})
			.groupby('Member type')
			.boxplot(column=['Distance in km'], by = 'Month', showmeans=True, showfliers=False))
	
	# Plot options

	plt.ylabel('Distance in km')
	
	
	# Saves plot as png

	plt.savefig('MonthlyCasMemRideDist.png')

	# Show plot

	plt.show()
	
	return

################

def ttesting():
	# Import ttest_ind function

	from scipy.stats import ttest_ind

	# Unpickle data

	data = pd.read_pickle('bike_data_with_distances.pkl')

	# Extract data for registered/casual users

	mem_data = data[data['Member type'] == 1]['Distance in km']
	cas_data = data[data['Member type'] == 0]['Distance in km']

	# Run t-test

	res = ttest_ind(mem_data,cas_data)

	if res[1]<0.01:
		ans = ''
	else:
		ans = 'not '

	print('There is '+ans+'a statistical difference between ride distances of casual and registered users.')
	
	return

################

def get_top_2000():

	bike_data = pd.read_pickle('bike_data_with_distances.pkl')

	# Gets top 2000 routes for Washington DC over 2017.

	for i in range(2):

		mem_bike_data = bike_data[bike_data['Member type'] == int(i)]

		# Group data by start and end station

		grouped = mem_bike_data.groupby(['Start station', 'End station'])

		# Create data frame of top 2000 trips and their number

		grouped_sizes = pd.DataFrame(data=grouped.size().nlargest(2000), columns=['Number of trips'])

		grouped_sizes.to_pickle('top_2000_'+str(i)+'.pkl')

	return grouped_sizes

######################

def add_google_dist_and_dur():

	# Grab bikeshare dataframe:

	bike_data = pd.read_pickle('clean_tidy_bike.pkl')

	# Grab google raw JSON dataframe:

	GoogleDist_df = pd.read_pickle('2000RoutesRawJSON.pkl')

	# Merge the two dataframes:

	merged = pd.merge(bike_data,GoogleDist_df, how='inner', on=['Start station', 'End station'])

	# Create benchmark column

	merged['Benchmarked time (seconds)'] = merged['Duration (ms)']/1000 - merged['GoogleDuration']

	# Create average speed column

	merged['Average speed m/s'] = 1000*merged['GoogleDistance']/(merged['Duration (ms)'])

	# Pickle it

	merged.to_pickle('top2000_with_speed_and_benchmark.pkl')

	return merged

######################

def show_popular_streets():

	import gmplot

	# Unpickle top 2000 routes data frame from get_top_2000_routes_duration_distance() function above.

	routes = pd.read_pickle('2000RoutesRawJSON.pkl')

	for i in range(2):

		b_df = pd.read_pickle('top_2000_'+str(i)+'.pkl')

		b_df.reset_index(inplace=True)

		routes2 = pd.merge(b_df,routes,how='inner',on=['Start station', 'End station'])

		# Initiate empty list

		street_lat_lng= list()

		# Loop over routes in 'Dirs_JSON' column of dataframe

		for route_js in routes2['Dirs_JSON']:

			# Turn route into dataframe

			route_df = json_normalize(route_js[0]['legs'][0]['steps'])

			# Extract list of end location latitudes and longitudes for each part of the trip

			coords = list(zip(route_df['end_location.lat'],route_df['end_location.lng']))

			# Add these new coordinates to list

			street_lat_lng = street_lat_lng + coords

		# Turn list of coordinates into dataframe

		street_df = pd.DataFrame({'Street_lat_lng': street_lat_lng})

		# Get unique coordinates

		most_popular_streets = street_df['Street_lat_lng'].unique()

		# Get list of latitudes for heat map

		heat_lats = [x[0] for x in most_popular_streets]

		# Get list of longitudes for heat map

		heat_lngs = [x[1] for x in most_popular_streets]

		###### NEED TO CHECK HOW THIS WORKS AND IMPORT CORRECT LIBRARY - cmd: pip install gmplot


		gmap = gmplot.GoogleMapPlotter(38.911681, -77.020079, 14)

		gmap.heatmap(heat_lats,heat_lngs)

		gmap.draw("MostPopularStreetsHeatMap-"+str(i)+".html")

	return

##########

def bike_ml_df():

	bike_data = pd.read_pickle('top2000_with_speed_and_benchmark.pkl')

	# Remove columns that are not features:

	bike_data = bike_data[['Start date', 'Start station number', 'Average speed m/s','Benchmarked time (seconds)', 'GoogleDistance','Member type','End station number']]

	# Add more features that may differentiate casual users from registered ones

	bike_data['Day'] = bike_data['Start date'].dt.dayofweek.astype(int)
	bike_data['Hour'] = bike_data['Start date'].dt.hour.astype(int)
	bike_data['Month'] = bike_data['Start date'].dt.month.astype(int)
	bike_data.drop('Start date', axis=1, inplace=True)

	bike_data.to_pickle('bike_ML.pkl')

	return bike_data

################

def knn_test(k_range):

	data = pd.read_pickle('bike_ML.pkl')

	X = data.drop('Member type', axis=1)

	y = data['Member type']

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	train_scores = list()
	test_scores = list()

	for k in range(1,k_range):
		knn = KNeighborsClassifier(n_neighbors = k)
		knn.fit(X_train, y_train)
		train_scores.append(knn.score(X_train,y_train))
		test_scores.append(knn.score(X_test, y_test))

    # Forget previous plot

	plt.clf()
	plt.cla()
	plt.close()
	plt.xlabel("Number of Neighbours")
	plt.ylabel("Accuracy")
	plt.title("Accuracy of knn")
	plt.plot(range(1,k_range),test_scores,label = 'Test set Accuracy')
	plt.plot(range(1,k_range),train_scores,label = 'Train set Accuracy')
	plt.legend()
	plt.savefig('kNNAccuracyPlot.png')
	plt.show()
	
	return



###########GEOCODING FUNCTIONS##########

def get_geocoded_objects(bike_data):

	# Returns a dictionary with keys that are street names of bike stations in Metro DC area
	# and values that are the corresponding GoogleV3 geocode locations.

	from geopy import geocoders
	from geopy.geocoders import GoogleV3
	gmaps = GoogleV3(api_key= MY_KEY)

	# Get list of all stations in data

	all_stations = pd.concat([bike_data['Start station'], bike_data['End station']]).unique()

	# Initialise empty lists
	
	lat_long_dict = dict()

	# Loop over station street names

	for station in all_stations:

		# Grab street name

		street_name = station + ', Washington, District of Columbia'

		# Add street name to loc_address

		try:

			# Use googlemaps v3 to geocode the street name

			location = gmaps.geocode(street_name, timeout=15)

			# If the above does not throw exception (due to connectivity) then append the
			# coordinates to loc_coord

			lat_long_dict[station] = location
		
		except:

			# If exception is thrown then return np.nan for that particular street.

			lat_long_dict[street_name] = np.nan

	lat_long_dict = {k:v for k,v in lat_long_dict.items() if v is not None}

	# PICKLE DICT

	save_obj(lat_long_dict, 'Dict_of_geocoded_locations')

	return lat_long_dict

########################

def get_top_2000_routes_duration_distance():

	import googlemaps
	gmaps = googlemaps.Client(key = MY_KEY)

	# Unpickle top 2000 routes dataframe:

	top2000Cas = pd.read_pickle('top_2000_0.pkl')
	top2000Mem = pd.read_pickle('top_2000_1.pkl')
	top2000 = pd.DataFrame(index = top2000Cas.index.union(top2000Mem.index)).reset_index()

	# Load gecoded locations dictionary from before

	loc_dict = load_obj('Dict_of_geocoded_locations')

	# Create new dictionary using a comprehension

	lat_long_pairs = {v.address : (v.latitude,v.longitude) for k,v in loc_dict.items() if v is not None}

	# Create new column in dataframe with entries given by the raw JSON data returned by Google Directions API

	top2000['Dirs_JSON'] = top2000.apply(lambda x: gmaps.directions(lat_long_pairs[x['Start station']],
																	lat_long_pairs[x['End station']],
																	mode='bicycling',
																	units='metric'), axis=1)

	# Extract duration and distance data from JSON and turn it into a dataframe, then put info into 
	# 'GoogleDuration' and 'GoogleDistance' columns respectively

	top2000['GoogleDistance'] = top2000['Dirs_JSON'].apply(lambda x: json_normalize(x[0]['legs'][0]['steps'])['distance.value'].sum())
	top2000['GoogleDuration'] = top2000['Dirs_JSON'].apply(lambda x: json_normalize(x[0]['legs'][0]['steps'])['duration.value'].sum())

	# Pickle the dataframe for access later

	top2000.to_pickle('2000RoutesRawJSON.pkl')


	return top2000

#################


### PICKLING #####

def save_obj(obj, name ):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

##########

def address_return(street_name, loc_dict):
	try:
		ans = loc_dict[street_name].address
	except:
		ans = np.nan
	return ans	





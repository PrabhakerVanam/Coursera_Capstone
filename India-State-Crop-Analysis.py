#!/usr/bin/env python
# coding: utf-8

# In[222]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from geopy.geocoders import Nominatim

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib.pyplot as plt


## Dataset Source
## https://data.gov.in/resources/district-wise-season-wise-crop-production-statistics-1997

data = pd.read_excel("D:\\GitHub\\Projects\\Coursera_Capstone\\India_crop_data.xlsx", sheet_name='apy' )

data.head()


# In[224]:


# All Indiain State Names
''' ['Andaman and Nicobar Islands', 'Andhra Pradesh',
       'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
       'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat',
       'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
       'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ',
       'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'] '''


# In[226]:


# Select the State for Analysis
state_name='Andhra Pradesh'
data=data[data['State_Name']==state_name ]
data.shape

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

# Kaarif & Rabi
season1 = 'Kharif     '
season2 = 'Rabi       '

# Kharif
data_kharif=data[data['Season']==season1]
data_kharif.shape
data_kharif = data_kharif.groupby(['District_Name','Crop' ]).sum()
data_kharif.sort_values('Production', ascending=False)
data_kharif.reset_index(inplace=True)

#Rabi
data_rabi=data[data['Season']==season2]
data_rabi.shape
data_rabi = data_rabi.groupby(['District_Name','Crop' ]).sum()
data_rabi.sort_values('Production', ascending=False)
data_rabi.reset_index(inplace=True)

# Whole year
data_wy = data[data['Season']== 'Whole Year ']
data_wy = data_wy.groupby(['District_Name','Crop' ]).sum()
data_wy.sort_values('Production', ascending=False)
data_wy.reset_index(inplace=True)



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height.""" '"%.2f" %height '
    for rect in rects:
        height = rect.get_height()
        print(height)
        ax.annotate('{}'.format("%.2f" %height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def normalize(df):
    """Method to normalize dataframe """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Top 5 Crops (Whole Year)
top_crop = data_wy.groupby(['Crop' ]).sum().sort_values('Production', ascending=False)
top_crop.reset_index(inplace=True)
top5_crop=top_crop.head(5)
top5_crop['Production'] = np.log10(top5_crop['Production'])

crop_labels = top5_crop['Crop'].tolist()
crop_means = top5_crop['Production'].tolist()


x = np.arange(len(crop_labels))  # the label locations
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(10,8))
rects = ax.bar(x - width+1.5/2, crop_means, width, label='Production')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('In Tonnes (1000K per Units)')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Crops in Andhra Pradesh (Whole Year)')
ax.set_xticks(x)
ax.set_xticklabels(crop_labels, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_crop_wy.png')
plt.show()

# Top 5 Crops (Season = Karif)
top_crop = data_kharif.groupby(['Crop' ]).sum().sort_values('Production', ascending=False)
top_crop.reset_index(inplace=True)
top5_crop=top_crop.head(5)
top5_crop['Production'] = np.log10(top5_crop['Production'])

crop_labels = top5_crop['Crop'].tolist()
crop_means = top5_crop['Production'].tolist()


x = np.arange(len(crop_labels))  # the label locations
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(10,8))
rects = ax.bar(x - width+1.5/2, crop_means, width, label='Production')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('In Tonnes (1000K per Units)')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Crops in Andhra Pradesh (Kharif)')
ax.set_xticks(x)
ax.set_xticklabels(crop_labels, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_crop-kharif.png')
plt.show()

# Top 5 Crops (Season = Rabi)
top_crop = data_rabi.groupby(['Crop' ]).sum().sort_values('Production', ascending=False)
top_crop.reset_index(inplace=True)
top5_crop=top_crop.head(5)
top5_crop['Production'] = np.log10(top5_crop['Production'])

crop_labels = top5_crop['Crop'].tolist()
crop_means = top5_crop['Production'].tolist()


x = np.arange(len(crop_labels))  # the label locations
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(10,8))
rects = ax.bar(x - width+1.5/2, crop_means, width, label='Production')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('In Tonnes (1000K per Units)')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Crops in Andhra Pradesh (Rabi)')
ax.set_xticks(x)
ax.set_xticklabels(crop_labels, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_crop-rabi.png')
plt.show()

#plt.savefig('ap_top5_crop.png')


# Top 5 districts for Coconut
#only_coconut = data_wy['Crop']=='Coconut '
#only_coconut.head()
data_wy_coconut= data_wy[data_wy['Crop']=='Coconut ']
data_wy_coconut.set_index('District_Name', inplace=True)
data_wy_coconut = normalize(data_wy_coconut[['Production']])
data_wy_coconut = data_wy_coconut.sort_values('Production', ascending=False)
data_wy_coconut_top5 = data_wy_coconut.head(5)
#data_wy_coconut_top5['Production'] = np.log10(data_wy_coconut_top5['Production'])

crop_labels1 = data_wy_coconut_top5.index
crop_means1 = data_wy_coconut_top5['Production'].tolist()

x1 = np.arange(len(crop_labels1))  # the label locations
fig, ax= plt.subplots(figsize=(10,8))
rects1 = ax.bar(x - width+1.5/2, crop_means1, width, label='Production')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Production')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Coconut Production Districts in Andhra Pradesh')
ax.set_xticks(x)
ax.set_xticklabels(crop_labels1, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects1)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_coconut_production.png')
plt.show()


# Top 5 districts for Sugarcane
#only_sugarcane = data_wy['Crop']=='Sugarcane'
#only_sugarcane.head()
data_wy_sugarcane= data_wy[data_wy['Crop']=='Sugarcane']
data_wy_sugarcane.set_index('District_Name', inplace=True)
data_wy_sugarcane = normalize(data_wy_sugarcane[['Production']])
data_wy_sugarcane = data_wy_sugarcane.sort_values('Production', ascending=False)
data_wy_sugarcane_top5 = data_wy_sugarcane.head(5)
#data_wy_sugarcane_top5['Production'] = np.log10(data_wy_sugarcane_top5['Production'])

crop_labels2 = data_wy_sugarcane_top5.index
crop_means2 = data_wy_sugarcane_top5['Production'].tolist()

x2 = np.arange(len(crop_labels2))  # the label locations
fig, ax= plt.subplots(figsize=(10,8))
rects2 = ax.bar(x2 - width+1.5/2, crop_means2, width, label='Production')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Production')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Sugarcane Production Districts in Andhra Pradesh')
ax.set_xticks(x2)
ax.set_xticklabels(crop_labels2, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects2)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_sgarcane_production.png')
plt.show()




# Top 5 districts for Rice
#only_rice = data_wy['Crop']=='Rice'
#only_rice.head()
data_wy_rice= data_wy[data_wy['Crop']=='Rice']
data_wy_rice.set_index('District_Name', inplace=True)
data_wy_rice = normalize(data_wy_rice[['Production']])
data_wy_rice = data_wy_rice.sort_values('Production', ascending=False)
data_wy_rice_top5 = data_wy_rice.head()
#data_wy_rice_top5['Production'] = np.log10(data_wy_rice_top5['Production'])

crop_labels3 = data_wy_rice_top5.index
crop_means3 = data_wy_rice_top5['Production'].tolist()

x3 = np.arange(len(crop_labels3))  # the label locations
fig, ax= plt.subplots(figsize=(10,8))
rects3= ax.bar(x3 - width+1.5/2, crop_means3, width, label='Production')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Production')
ax.set_xlabel('Crop')            
ax.set_title('Top 5 Rice Production Districts in Andhra Pradesh')
ax.set_xticks(x3)
ax.set_xticklabels(crop_labels3, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects3)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_rice_production.png')
plt.show()


# Top 5 districts for Banana
data_wy_banana= data_wy[data_wy['Crop']=='Banana']
data_wy_banana.set_index('District_Name', inplace=True)
data_wy_banana = normalize(data_wy_banana[['Production']])
data_wy_banana = data_wy_banana.sort_values('Production', ascending=False)
data_wy_banana_top5 = data_wy_banana.head()
#data_wy_maize_top5['Production'] = np.log10(data_wy_maize_top5['Production'])

crop_labels4 = data_wy_banana_top5.index
crop_means4 = data_wy_banana_top5['Production'].tolist()

x4 = np.arange(len(crop_labels4))  # the label locations
fig, ax =plt.subplots(figsize=(10,8))
rects4= ax.bar(x4 - width+1.5/2, crop_means4, width, label='Production')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Production')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Banana Production Districts in Andhra Pradesh')
ax.set_xticks(x4)
ax.set_xticklabels(crop_labels4, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects4)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_banana_production.png')
plt.show()



# Top 5 districts for Tobacco

data_wy_tobacco= data_wy[data_wy['Crop']=='Tobacco']
data_wy_tobacco.set_index('District_Name', inplace=True)
data_wy_tobacco = normalize(data_wy_tobacco[['Production']])
data_wy_tobacco = data_wy_tobacco.sort_values('Production', ascending=False)
data_wy_tobacco_top5 = data_wy_tobacco.head()
#data_wy_tobacco_top5['Production'] = np.log10(data_wy_tobacco_top5['Production'])

crop_labels5 = data_wy_tobacco_top5.index
crop_means5 = data_wy_tobacco_top5['Production'].tolist()

x5 = np.arange(len(crop_labels4))  # the label locations
fig, ax =plt.subplots(figsize=(10,8))
rects5= ax.bar(x5 - width+1.5/2, crop_means5, width, label='Production')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Production')
ax.set_xlabel('Crop')
ax.set_title('Top 5 Tobacco Production Districts in Andhra Pradesh')
ax.set_xticks(x4)
ax.set_xticklabels(crop_labels5, rotation=45)
ax.set_yticklabels('')
ax.legend()

autolabel(rects5)
fig.tight_layout()
fig.savefig('D:\\GitHub\\Projects\\Coursera_Capstone\\ap_top5_tobacco_production.png')
plt.show()


# In[228]:


geolocator = Nominatim(user_agent="my-application1")
location = geolocator.geocode(state_name)
latitude = location.latitude
longitude = location.longitude

print("State :{}, Latitude: {}, Longitude: {}".format(state_name, latitude, longitude))


# In[229]:


# one hot encoding
data_onehot = pd.get_dummies(data['Crop'], prefix="", prefix_sep="")
data_onehot.head()
data_onehot.columns


# In[230]:


# Correct the District name for VISHAKAPATNAM for Andhra Pradesh state
data_onehot['District_Name'] = data['District_Name'].replace('VISAKHAPATANAM',"VISAKHAPATNAM")
#data_onehot['District_Name'].replace({'RANGAREDDI':'RANGAREDDY'}, inplace=True)
data_onehot.shape


# In[233]:


#Next, let's group rows by District and by taking the mean of the frequency of occurrence of each category

data_onehot_grouped= data_onehot.groupby('District_Name').mean().reset_index()

data_onehot_grouped.head()


# In[232]:


# Get all districts fod a given state
districts = data_onehot_grouped.District_Name.unique()
districts


# In[198]:


# Get the langitude and latitude for each district
district_names =[]
district_lats=[]
district_lngs= []
#geolocator = Nominatim(user_agent="my-application1")
for district in districts:
    #print('' + district)    
    location = geolocator.geocode(district)
    latitude = location.latitude
    longitude = location.longitude
    #print('The geograpical coordinate of {} are {}, {}.'.format(district, latitude, longitude))
    district_lats.append(latitude)
    district_lngs.append(longitude)
    district_names.append(district)

district_lats
district_lngs
district_names

for dist, lat, lon in zip(district_names, district_lats, district_lngs):
    print ('{} - {}, {}'.format(dist, lat, lon))


# In[199]:


dist_lat_lng = pd.DataFrame([district_names,district_lats,district_lngs]).T.reset_index(drop=True)
dist_lat_lng.columns = ['District_Name','latitude','longitude']
dist_lat_lng.head()

dist_data_onehot_grouped = data_onehot_grouped.join(dist_lat_lng.set_index('District_Name'), on='District_Name' )
dist_data_onehot_grouped.fillna(0, inplace=True)


# In[200]:


dist_data_onehot_grouped.shape


# In[201]:



#dist_data_onehot_grouped.columns
dist_crop_columns = ['District_Name','Arecanut', 'Arhar/Tur', 'Bajra', 'Banana',
       'Beans & Mutter(Vegetable)', 'Bhindi', 'Bottle Gourd', 'Brinjal',
       'Cabbage', 'Cashewnut', 'Castor seed', 'Citrus Fruit', 'Coconut ',
       'Coriander', 'Cotton(lint)', 'Cowpea(Lobia)', 'Cucumber',
       'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Grapes',
       'Groundnut', 'Horse-gram', 'Jowar', 'Korra', 'Lemon', 'Linseed',
       'Maize', 'Mango', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Niger seed',
       'Onion', 'Orange', 'Other  Rabi pulses', 'Other Fresh Fruits',
       'Other Kharif pulses', 'Other Vegetables', 'Papaya',
       'Peas  (vegetable)', 'Pome Fruit', 'Pome Granet', 'Potato', 'Ragi',
       'Rapeseed &Mustard', 'Rice', 'Safflower', 'Samai', 'Sannhamp', 'Sapota',
       'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower',
       'Sweet potato', 'Tapioca', 'Tobacco', 'Tomato', 'Turmeric', 'Urad',
       'Varagu', 'Wheat', 'other fibres', 'other misc. pulses',
       'other oilseeds']


# In[202]:


num_top_crops = 5
dist_crop_data = dist_data_onehot_grouped.loc[:,dist_crop_columns]

for district in dist_crop_data['District_Name']:
    print("----"+district+"----")
    temp = dist_crop_data[dist_crop_data['District_Name'] == district].T.reset_index()
    temp.columns = ['crop','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_crops))
    print('\n')


# In[203]:


#Let's put that into a pandas dataframe
def return_most_common_crops(row, num_top_crops):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_crops]


# In[234]:


num_top_crops = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top Crops
columns = ['District_Name']
for ind in np.arange(num_top_crops):
    try:
        columns.append('{}{} Most Common Crop'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Crop'.format(ind+1))

# create a new dataframe
districts_crops_sorted = pd.DataFrame(columns=columns)
districts_crops_sorted['District_Name'] = dist_crop_data['District_Name']

for ind in np.arange(dist_crop_data.shape[0]):
    districts_crops_sorted.iloc[ind, 1:] = return_most_common_crops(dist_crop_data.iloc[ind, :], num_top_crops)

districts_crops_sorted.head()


# In[210]:


#Cluster Districts
#Run k-means to cluster the neighborhood into 4 clusters

# set number of clusters
kclusters = 4

dist_crop_data_clustering = dist_crop_data.drop('District_Name', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dist_crop_data_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[211]:


# add clustering labels
districts_crops_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

district_merged = dist_data_onehot_grouped

# merge district_merged with toronto_data to add latitude/longitude for each neighborhood
district_merged = district_merged.join(districts_crops_sorted.set_index('District_Name'), on='District_Name')
district_merged['Cluster Labels'] =  district_merged['Cluster Labels'].fillna(0.0).astype(int)

district_merged.head() # check the last columns!
district_merged.columns


# In[217]:



# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(district_merged['latitude'], district_merged['longitude'], district_merged['District_Name'], district_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
#map_path = "d:\\Temp\\"+state_name+"-map.html"
map_path = "d:\\Temp\\"+state_name+"-map.html"
print(map_path)
map_clusters.save(map_path, close_file=True)



#import selenium.webdriver
#driver = selenium.webdriver.PhantomJS('D:\\Software\\phantomjs-2.1.1-windows\\phantomjs-2.1.1-windows\\bin\\phantomjs.exe')
#driver.set_window_size(4000, 3000)
#driver.get(map_path)
#driver.save_screenshot('ap_map.png')

# In[235]:


##Examine Clusters

#Cluster 1

district_merged.loc[district_merged['Cluster Labels'] == 0, district_merged.columns[[1] + list(range(5, district_merged.shape[1]))]]


# In[219]:



#Cluster 2

district_merged.loc[district_merged['Cluster Labels'] == 1, district_merged.columns[[1] + list(range(5, district_merged.shape[1]))]]


# In[220]:


#Cluster 3

district_merged.loc[district_merged['Cluster Labels'] == 2, district_merged.columns[[1] + list(range(5, district_merged.shape[1]))]]


# In[221]:


#Cluster 4

district_merged.loc[district_merged['Cluster Labels'] == 3, district_merged.columns[[1] + list(range(5, district_merged.shape[1]))]]


# In[ ]:





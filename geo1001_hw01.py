#-- GEO1001.2020--hw01
#-- Pratyush Kumar
#-- 5359252

#%% Imports
import os
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
#%% # Functions
def megadf(path='./hw01'):
    """takes the path of existing sensor datasets and
        returns a dataframe with all datasets combined

    Args:
        path ([str in linux format pss: use /]): [the path to the datasets in excel]
    """
    ls = []
    for file in os.listdir(path):
        if(file.endswith(".xls") and not file.startswith("._")):
            #print(file) 
            ls.append(path + '/' + file)
    ## make a list of dataframes
    templist=[]
    for _ in ls:
        # read each df
        tempDF = pd.read_excel(_ , header=[0], skiprows=[0,1,2,4])
        #add a row to each df for the sensor name
        sensor_name = _.split('/')[-1][7:8]
        tempDF['sensor'] = sensor_name
        templist.append( tempDF )
    df = pd.DataFrame()
    df = pd.concat(templist, ignore_index=True)
    return df
#############


df = megadf()
#df.to_json('joinedDF.json') #save as json for future usage

# %%
# to continue with the assignment
"""
df=pd.read_json('./joinedDF.json')
df['FORMATTED DATE-TIME']=df['FORMATTED DATE-TIME'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))
#df['FORMATTED DATE-TIME'] = pd.to_datetime(df['FORMATTED DATE-TIME'] , unit='ns')#.astype('datetime64[ns]')
"""
# %% ##part 1 from A1
"""
##Aim:  Compute mean statistics
Create 1 plot that contains histograms for the 5 sensors Temperature values. 
Compare histograms with 5 and 50 bins, 
why is the number of bins important?

Create 1 plot where frequency poligons for the 5 sensors Temperature values 
overlap in different colors with a legend.

Generate 3 plots that include the 5 sensors boxplot for: 
Wind Speed, Wind Direction and Temperature.
"""
# %%
#from part a1
# ## mean stats: mean, variance and standard deviation
collist=  ['FORMATTED DATE-TIME', 'Direction ‚ True', 'Wind Speed',
       'Crosswind Speed', 'Headwind Speed', 'Temperature', 'Globe Temperature',
       'Wind Chill', 'Relative Humidity', 'Heat Stress Index', 'Dew Point',
       'Psychro Wet Bulb Temperature', 'Station Pressure',
       'Barometric Pressure', 'Altitude', 'Density Altitude',
       'NA Wet Bulb Temperature', 'WBGT', 'TWL', 'Direction ‚ Mag', 'sensor']


# %%
print('mean')
print(df.groupby('sensor').mean())
print('var')
print(df.groupby('sensor').var())
print('std')
print(df.groupby('sensor').std())
#%%
"""sns.set_style("white")
plt.figure(figsize=(12,10))
plt.xlabel('Ration short/total', fontsize=18)
plt.title ('CO3 In vitro transcription, Na+', fontsize=22)

ax = sns.distplot(df.Temperature, hist = True, bins=5, kde=False)

#plt.savefig("hist.svg", format="svg")
plt.show()"""
# %%
# plot
f, axes = plt.subplots(1, 5, figsize=(20,6), sharex=True)
sns.distplot( df['Temperature'].where(df.sensor=='A'), color="skyblue", ax=axes[0])
sns.distplot( df['Temperature'].where(df.sensor=='B'), color="olive", ax=axes[1])
sns.distplot( df['Temperature'].where(df.sensor=='C'), color="gold", ax=axes[2])
sns.distplot( df['Temperature'].where(df.sensor=='D'), color="teal", ax=axes[3])
sns.distplot( df['Temperature'].where(df.sensor=='E'), color="magenta", ax=axes[4])

plt.show()
# %%
plt.figure()
grid = sns.FacetGrid(df, col="sensor", hue = 'sensor', palette="Set1",margin_titles=True)
grid.map(sns.distplot , "Temperature", bins= 10 );
plt.show();
# %%
plt.figure()
grid = sns.FacetGrid(df, col="sensor", hue = 'sensor', palette="Set1",margin_titles=True)
grid.map(sns.distplot , "Temperature", bins= 5, color='blue' );
grid.map(sns.distplot , "Temperature", bins= 50 );
plt.show();
#%%
# 1 plot with all freq as diff legends
plt.figure( figsize=(20,10) )
sns.distplot( df['Temperature'].where(df.sensor=='A'), color="skyblue" , hist=False, kde=True , label='A')
sns.distplot( df['Temperature'].where(df.sensor=='B'), color="olive", hist=False, kde=True, label='B')
sns.distplot( df['Temperature'].where(df.sensor=='C'), color="gold", hist=False, kde=True, label='C')
sns.distplot( df['Temperature'].where(df.sensor=='D'), color="teal", hist=False, kde=True, label='D')
sns.distplot( df['Temperature'].where(df.sensor=='E'), color="magenta", hist=False, kde=True, label='E')
plt.legend()
# %%
# Generate 3 plots that include the 5 sensors boxplot for: 
# Wind Speed, Wind Direction and Temperature.\
#wind speed
""" # this peice of shit doesnt work for now but will once you figure out 
# how to be able to update the facetgrid with more rows of your own
# each row needs to have a facet of wind speed, temperature etc

f, axes = plt.subplots(1, 5, figsize=(20,6), sharex=True)
sns.distplot( df['Temperature'].where(df.sensor=='A'), color="skyblue", ax=axes[0])
sns.distplot( df['Temperature'].where(df.sensor=='B'), color="olive", ax=axes[1])
sns.distplot( df['Temperature'].where(df.sensor=='C'), color="gold", ax=axes[2])"""
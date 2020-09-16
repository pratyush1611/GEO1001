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
def megadf(path='./GEO1001_maths/Assignment/asg1_21sep/hw01'):
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
df=pd.read_json('./GEO1001_maths/Assignment/asg1_21sep/joinedDF.json')
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

g = sns.FacetGrid(df, col='sensor')
g.map(sns.distplot(bin=5), "Temperature")
plt.show()

# %%
#les see how it goes on in github
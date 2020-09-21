#-- GEO1001.2020--hw01
#-- Pratyush Kumar
#-- 5359252

#%% Imports
import os
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem, t , ttest_ind
from scipy import mean
#%% # Functions

def megadf(path='./hw01'):
    """#takes the path of existing sensor datasets and
        #returns a dataframe with all datasets combined

    #Args:
        #path ([str in linux format pss: use /]): [the path to the datasets in excel]
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
#####
#run first time to create a conglomerate of the csv files into a single json file
#df = megadf()
#df.to_json('joinedDF.json') #save as json for future usage
# %%
# to load dataset from json on github

#df=pd.read_json('./joinedDF.json') # if json is present locally
df= pd.read_json('https://raw.githubusercontent.com/pratyush1611/geo1001/main/joinedDF.json')
df['FORMATTED DATE-TIME']=df['FORMATTED DATE-TIME'].apply(lambda d: datetime.datetime.fromtimestamp(int(d)/1000).strftime('%Y-%m-%d %H:%M:%S'))

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


# %% discriptive stats
print('mean')
print(df.groupby('sensor').mean())
print('var')
print(df.groupby('sensor').var())
print('std')
print(df.groupby('sensor').std())

# %% 1 plot with histograms

plt.figure()
grid = sns.FacetGrid(df, col="sensor", hue = 'sensor', palette="coolwarm",margin_titles=True)
grid.map(sns.distplot , "Temperature" , bins= 15 );
plt.show()


# %% histograms at 5 and 50 bins

grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot , "Temperature", bins= 5 , kde=False );

grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True , hue='sensor')
grid = grid.map(sns.distplot , "Temperature", bins= 50 , kde=False );

plt.show()

#%%
# 1 plot with all freq as diff legends
"""
plt.figure( figsize=(20,10) )
sns.distplot( df['Temperature'].where(df.sensor=='A'), color="skyblue" , hist=False, kde=True , label='A')
sns.distplot( df['Temperature'].where(df.sensor=='B'), color="olive", hist=False, kde=True, label='B')
sns.distplot( df['Temperature'].where(df.sensor=='C'), color="gold", hist=False, kde=True, label='C')
sns.distplot( df['Temperature'].where(df.sensor=='D'), color="teal", hist=False, kde=True, label='D')
sns.distplot( df['Temperature'].where(df.sensor=='E'), color="magenta", hist=False, kde=True, label='E')
plt.legend()
"""

for i in df.sensor.unique():
    sns.distplot(df['Temperature'].where(df.sensor==str(i)) ,hist=False, kde=True, label=i)


# %%
# Generate 3 plots that include the 5 sensors boxplot for: 
# Wind Speed, Wind Direction and Temperature.\
# 'Direction ‚ True', 'Wind Speed'

f, axes = plt.subplots(3,  figsize=(10,20), sharex=False)

sns.boxplot( x='sensor', y='Direction ‚ True' , data = df , palette='Set3', ax=axes[0] ,orient='v', width = 0.4 )
sns.boxplot( x='sensor', y='Wind Speed' , data = df , palette='Set3',ax=axes[1],orient='v', width = 0.4)
sns.boxplot( x='sensor', y='Temperature' , data = df , palette='Set3',ax=axes[2] ,orient='v', width = 0.4)


##///////////////// A1 PART DONE ///////////##
#%% Part A2
"""
Plot PMF, PDF and CDF for the 5 sensors Temperature values in 
independent plots (or subplots). 
Describe the behaviour of the distributions, 
are they all similar? what about their tails?

For the Wind Speed values, plot the pdf and the kernel density estimation. 
Comment the differences.
"""
#PDF
grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot , "Temperature", bins= 10 , kde=False );
#grid.fig.subplots_adjust(top=0.85)
grid.fig.suptitle('PDF' , fontsize=16)

#PMF
grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot , "Temperature" , kde=True );
#grid.fig.subplots_adjust(top=-0.85)
grid.fig.suptitle('PMF', fontsize=16)
#CDF
grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
kwargs = {'cumulative': True}
hist_kw = {'cumulative': True,
            'density': True}
grid = grid.map(sns.distplot, 'Temperature',hist_kws=hist_kw , kde_kws=kwargs)
#grid.fig.subplots_adjust(top=0.85)
grid.fig.suptitle('CDF', fontsize=16);
#%%

#WIND SPEED
grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot , "Wind Speed", bins= 10 , kde=False );
grid.fig.suptitle('Windspeed PDF' , fontsize=16)

grid = sns.FacetGrid(df, col="sensor",  palette="Set2",margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot , "Wind Speed", bins= 10 , kde=True );
grid.fig.suptitle('KDE' , fontsize=16)
#%%
for i in df.sensor.unique():
    sns.distplot(df['Wind Speed'].where(df.sensor==str(i)) ,hist=False, kde=True, label=i)

# %%
#A3
"""
Compute the correlations between all the sensors for the variables: 
Temperature, Wet Bulb Globe Temperature (WBGT), Crosswind Speed. 
Perform correlation between sensors with the same variable, 
not between two different variables; for example, 
correlate Temperature time series between sensor A and B. 
Use Pearson’s and Spearmann’s rank coefficients. 
Make a scatter plot with both coefficients with the 3 variables.
"""
#workflow 
#set dataframe for joining
df_tp = df[['FORMATTED DATE-TIME', 'Temperature','sensor']] 
df_tp = df_tp.rename(columns={'FORMATTED DATE-TIME':'Time', 'Temperature':'Temp', 'sensor':'sensor'})
df_tp.set_index('Time', inplace=True)

df_wbgt = df[['FORMATTED DATE-TIME', 'Psychro Wet Bulb Temperature','sensor']] 
df_wbgt = df_wbgt.rename(columns={'FORMATTED DATE-TIME':'Time', 'Psychro Wet Bulb Temperature':'WBGT', 'sensor':'sensor'})
df_wbgt.set_index('Time', inplace=True)


df_cs = df[['FORMATTED DATE-TIME', 'Crosswind Speed','sensor']] 
df_cs = df_cs.rename(columns={'FORMATTED DATE-TIME':'Time', 'Crosswind Speed':'Crosswind_Speed', 'sensor':'sensor'})
df_cs.set_index('Time', inplace=True)

#join dataframes
df_tp_corr = (df_tp[['Temp']][df_tp.sensor == 'A']
            .join(df_tp[['Temp']][df_tp.sensor == 'B'], on='Time', how='left', lsuffix='A', rsuffix='B', sort=True)
            .join(df_tp[['Temp']][df_tp.sensor == 'C'], on='Time', how='left', rsuffix='C')
            .join(df_tp[['Temp']][df_tp.sensor == 'D'], on='Time', how='left', rsuffix='D')
            .join(df_tp[['Temp']][df_tp.sensor == 'E'], on='Time', how='left', rsuffix='E')
            .dropna()
            .rename(columns={'TempA':'TempA', 'TempB':'TempB', 'Temp':'TempC', 'TempD':'TempD', 'TempE':'TempE'})
           )

df_wbgt_corr = (df_wbgt[['WBGT']][df_wbgt.sensor == 'A']
            .join(df_wbgt[['WBGT']][df_wbgt.sensor == 'B'], on='Time', how='left', lsuffix='A', rsuffix='B', sort=True)
            .join(df_wbgt[['WBGT']][df_wbgt.sensor == 'C'], on='Time', how='left', rsuffix='C')
            .join(df_wbgt[['WBGT']][df_wbgt.sensor == 'D'], on='Time', how='left', rsuffix='D')
            .join(df_wbgt[['WBGT']][df_wbgt.sensor == 'E'], on='Time', how='left', rsuffix='E')
            .dropna()
            .rename(columns={'WBGTA':'WBGTA', 'WBGTB':'WBGTB', 'WBGT':'WBGTC', 'WBGTD':'WBGTD', 'WBGTE':'WBGTE'})
           )

df_cs_corr = (df_cs[['Crosswind_Speed']][df_cs.sensor == 'A']
            .join(df_cs[['Crosswind_Speed']][df_cs.sensor == 'B'], on='Time', how='left', lsuffix='A', rsuffix='B', sort=True)
            .join(df_cs[['Crosswind_Speed']][df_cs.sensor == 'C'], on='Time', how='left', rsuffix='C')
            .join(df_cs[['Crosswind_Speed']][df_cs.sensor == 'D'], on='Time', how='left', rsuffix='D')
            .join(df_cs[['Crosswind_Speed']][df_cs.sensor == 'E'], on='Time', how='left', rsuffix='E')
            .dropna()
            .rename(columns={'Crosswind_SpeedA':'Crosswind_SpeedA', 'Crosswind_SpeedB':'Crosswind_SpeedB', 'Crosswind_Speed':'Crosswind_SpeedC', 'Crosswind_SpeedD':'Crosswind_SpeedD', 'Crosswind_SpeedE':'Crosswind_SpeedE'})
           )

df_tp_pear_corr=df_tp_corr.corr(method=  'pearson')
df_wbgt_pear_corr=df_wbgt_corr.corr(method='pearson')
df_cs_pear_corr=df_cs_corr.corr(method=  'pearson')

df_tp_spear_corr=df_tp_corr.corr(method=  'spearman')
df_wbgt_spear_corr=df_wbgt_corr.corr(method='spearman')
df_cs_spear_corr=df_cs_corr.corr(method=  'spearman')


print('Pearson Correlation')
print(df_tp_pear_corr)
print(df_wbgt_pear_corr)
print(df_cs_pear_corr)

print('Spearman Correlation')
print(df_tp_spear_corr)
print(df_wbgt_spear_corr)
print(df_cs_spear_corr)

#plotting : figure out plotting in a pairwise plot before correlation or after
"""df_tp_pear_corr.values.tolist()

sns.pairplot(df_tp_pear_corr)
sns.pairplot(df_wbgt_pear_corr)
sns.pairplot(df_cs_pear_corr)

sns.pairplot(df_tp_spear_corr)
sns.pairplot(df_wbgt_spear_corr)
sns.pairplot(df_cs_spear_corr)
"""

# fix the plots for ab ac ad ae bc bd be cd ce de 
p = df_tp_pear_corr.values.tolist()[0][1:]
p.extend(df_tp_pear_corr.values.tolist()[1][2:])
p.extend(df_tp_pear_corr.values.tolist()[2][3:])
p.extend(df_tp_pear_corr.values.tolist()[3][4:])
q = df_wbgt_pear_corr.values.tolist()[0][1:]
q.extend(df_wbgt_pear_corr.values.tolist()[1][2:])
q.extend(df_wbgt_pear_corr.values.tolist()[2][3:])
q.extend(df_wbgt_pear_corr.values.tolist()[3][4:])
r = df_cs_pear_corr.values.tolist()[0][1:]
r.extend(df_cs_pear_corr.values.tolist()[1][2:])
r.extend(df_cs_pear_corr.values.tolist()[2][3:])
r.extend(df_cs_pear_corr.values.tolist()[3][4:])

df_fix_pear = pd.DataFrame(data=zip(p,q,r) , index = ['ab', 'ac', 'ad', 'ae' ,'bc' ,'bd' ,'be', 'cd', 'ce', 'de'] , columns=['Temperature', 'WBGT','CrossWind Speed'])
sns.scatterplot(data = df_fix_pear)

plt.figure()
p = df_tp_spear_corr.values.tolist()[0][1:]
p.extend(df_tp_spear_corr.values.tolist()[1][2:])
p.extend(df_tp_spear_corr.values.tolist()[2][3:])
p.extend(df_tp_spear_corr.values.tolist()[3][4:])
q = df_wbgt_spear_corr.values.tolist()[0][1:]
q.extend(df_wbgt_spear_corr.values.tolist()[1][2:])
q.extend(df_wbgt_spear_corr.values.tolist()[2][3:])
q.extend(df_wbgt_spear_corr.values.tolist()[3][4:])
r = df_cs_spear_corr.values.tolist()[0][1:]
r.extend(df_cs_spear_corr.values.tolist()[1][2:])
r.extend(df_cs_spear_corr.values.tolist()[2][3:])
r.extend(df_cs_spear_corr.values.tolist()[3][4:])
df_fix_pear = pd.DataFrame(data=zip(p,q,r) , index = ['ab', 'ac', 'ad', 'ae' ,'bc' ,'bd' ,'be', 'cd', 'ce', 'de'] , columns=['Temperature', 'WBGT','CrossWind Speed'])
sns.scatterplot(data = df_fix_pear)


# %% PArt A4 
a4 = df[['FORMATTED DATE-TIME', 'Temperature','Wind Speed','sensor']] 
a4 = a4.rename(columns={'FORMATTED DATE-TIME':'Time', 'Temperature':'Temp', 'Wind Speed':'WS','sensor':'sensor'})
a4.set_index('Time', inplace=True)
a4.dropna(inplace=True)

kwargs = {'cumulative': True}
hist_kw = {'cumulative': True, 'density': True}
grid = sns.FacetGrid(a4, col="sensor",  palette="Set2" ,margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot, 'Temp', hist_kws=hist_kw , kde_kws=kwargs)

grid = sns.FacetGrid(a4, col="sensor",  palette="Set2" ,margin_titles=True, hue='sensor')
grid = grid.map(sns.distplot, 'WS', hist_kws=hist_kw , kde_kws=kwargs)

# %% adding the confidence to table

def confidence95(data):
    """gives back the confidence interval value for df

    Args:
        data ([pandas dataframe]): [df containing the temperature values for a particular sensor]
    """
    confidence = 0.95
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return(h)

conf_df=pd.DataFrame(columns= list('ABCDE') , index=['Temperature','Wind Speed'])

for i in list('ABCDE'):
    conf_df[str(i)] = confidence95(a4[['Temp','WS']][a4.sensor==str(i)])

conf_df.to_csv('./exports/confidence_intervals_A4.csv')
conf_df
#%% A4 hypothesis tests
def hypothesis_tp(dat1,dat2):
    """gives back the p stat 

    Args:
        data,2 ([pandas dataframe]): [df containing the temp and wind spd val for a particular sensor]
    """
    t, p = ttest_ind(dat1,dat2)
    return(p)

hypo_df=pd.DataFrame( index=['Temp' ,'Wind Speed'])

hypo_df['E-D'] = hypothesis_tp(  a4[['Temp','WS']][a4.sensor=='E']  ,  a4[['Temp','WS']][a4.sensor=='D']  )
hypo_df['D-C'] = hypothesis_tp(  a4[['Temp','WS']][a4.sensor=='D']  ,  a4[['Temp','WS']][a4.sensor=='C']  )
hypo_df['C-B'] = hypothesis_tp(  a4[['Temp','WS']][a4.sensor=='C']  ,  a4[['Temp','WS']][a4.sensor=='B']  )
hypo_df['B-A'] = hypothesis_tp(  a4[['Temp','WS']][a4.sensor=='B']  ,  a4[['Temp','WS']][a4.sensor=='A']  )
hypo_df = hypo_df.T
hypo_df.to_csv('./exports/hypo_p_val.csv')
hypo_df
# %%

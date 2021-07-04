import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
def Gaussian_transform(data, variable):
    # function to fill na with a random sample
    df = data.copy()
    
    # random sampling
    df[variable+'_random'] = df[variable]
    
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
    
    return df[variable+'_random']

def Logarithmic(df,variable):
    df['Log_'+variable]=np.log(df[variable]+1)


def Exponential(df,variable):
    df['Exp_'+variable]=df[variable]**(1/5)
    
    
def Reciprocal(df,variable):
    df['Rec_'+variable]=1/(df[variable]+1)
    
    
def Sqaure_Root(df,variable):
    df['Sqr_'+variable]=df[variable]**(1/2)
    
    
    
def Box_Cox(df,variable):
    df[variable+'_boxcox'], param = stats.boxcox(df[variable]+1) # you can vary the exponent as needed
    print('Optimal lambda: ', param)

def qq_plot(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)

    plt.show()

def details_null():
    print('''----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------Descriptions---------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------
1.deling with null values

2.Random Sample Imputation

3.median_impute

4.End of Distribution imputation
-----------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------deling with null values--------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------
1.find the null value .isnull()

2.calculate all null value with the help with the help of .isnull().sum()

3.calculate the % of all null value with the help of .isnull().mean()

4.detacting null values by :where(.isnull(),1,0)


----------------------------------------------------------------------------------------------------------------------------
-------------------------------------Random Sample Imputation---------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------
1.dealing with the null values

2.fill the random sample with the help of you have to take the help of replace method -:hint loc function

Q1.how can you make the random sample -.sample()


Q2.you have to provide the n for sample then how can you select the best n that can  be fit into the function-fill number of 
null values
Q3.if you select best sample?

    if it is wrong then how can you select best sample which have not null values? hint-dropna
3.you have to equal the index number of random sample with the help of .index

4.you have to change the index with the loc

5.make the function
-----------------------------------------------------------------------------------------------------------------------------
----------------------------------------------median_impute------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------
1.calculate the % and null values of the data with k.isnull().mean() and.sum()

2.use fillna to impute the median- hint age column

3.use the dectaing with np.where()

3.make the function for the imputation

4.plot the kde graph by plot
-----------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------End of Distribution imputation---------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------
1.dealing with missing values

2.get the rough area about the distribution(sns.displot())

3.calculate the extreme value with the help of this fourmula ::extreme=df.Age.mean()+3*df.Age.std()

4.fill the extreme value by the fillna

5.plot the graph distribution

op-you can check with the boxplot
6.make the function
------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------''')
def line_chart():
    print(  ''''
    def line_chart(df,x,y):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df[x].plot(kind='kde', ax=ax)
        df[y].plot(kind='kde', ax=ax, color='red')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='best')''')
def impute_random():
    print( '''
    def impute_nan(df,variable,median):
        df[variable+"_median"]=df[variable].fillna(median)
        df[variable+"_random"]=df[variable]
        ##It will have the random sample to fill the na
        random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
        ##pandas need to have same index in order to merge the dataset
        random_sample.index=df[df[variable].isnull()].index
        df.loc[df[variable].isnull(),variable+'_random']=random_sample''')
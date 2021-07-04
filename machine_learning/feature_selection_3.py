def details_feature_selection():
    print('''
=================================================================================================================
===================================Variance Threshold============================================================
=================================================================================================================

1.make x and y 


2.train the data


3.use sklearn,feature selection,variance threshold


4.use get_support and extract the columns
  hint:::x[x.columns[var_.get_support()]]
  
  
  
=================================================================================================================
=================================Correlation=====================================================================
=================================================================================================================

  step to find the less correaltive

1.make x and y
2.correlation 
3.use heat map

================================================================================================================
================================information Gain================================================================
================================================================================================================
1.make x,y (you can use train_test_split)


2.use sklearn,feature selection,mutual_info_classifier


3.set_the column name with the help of index /\ avoid


4.op. plot the barchart /\avoid

5.use k-best for selecting top number 


6.use get support and print name 
  hint(X_train.columns[best_five.get_support()]

7.extracting the columns

===============================================================================================================
====================================Mutal_info gain============================================================
===============================================================================================================
1.make x,y (you can use train_test_split)


2.use sklearn,feature selection,mutual_info_classifier


3.set_the column name with the help of index /\ avoid


4.op. plot the barchart /\avoid

5.use k-best for selecting top number 


6.use get support and print name 
  hint(X_train.columns[best_five.get_support()]

7.extracting the columns

===============================================================================================================
=====================================Chisquare Test For Feature Selection======================================
===============================================================================================================
 Chisquare Test For Feature Selection
 
1.extract catagorical columns
2.from sklearn.feature_selection import chi2
3.p_values=pd.Series(f_p_values[1])
    p_values.index=X_train.columns
4.use sort_index(ascending=False)

===============================================================================================================
=========================================feature importance====================================================
===============================================================================================================
1.make x and y

2.use sklear,ensembel,extratree

3.fit and use feautere importance to show the number

4.plot barchart of top 10 with help of .nlargest()

==============================================================================================================
==============================================================================================================''')

    
    
    
def correlation():
    print('''
    def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr''')
    
    

    
    
def heat_map():
    print('''
cor=x_train.corr()
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = x_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.show()''')
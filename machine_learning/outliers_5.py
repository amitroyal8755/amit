def details_outlires():
    print('''
    Q1. should we remove the outliers?

ans-:if the ouliers is the part of the information then keep it

eg-bank defalter is the part of the main source of the information 
1.if the data is Normal distribution the use 3 standard deviation

formula is :- uppper_boundary=df['Age'].mean() + 3* df['Age'].std()

lower_boundary=df['Age'].mean() - 3* df['Age'].std()

2.if the data is Skewed then use iqr techiqunie

formula is :-

    IQR=df.Fare.quantile(0.75)-df.Fare.quantile(0.25)


    lower_bridge=df['Fare'].quantile(0.25)-(IQR*1.5)


    upper_bridge=df['Fare'].quantile(0.75)+(IQR*1.5)
    
    
    ==========================================================================================================================
    ==========================================================================================================================
    ==========================================================================================================================
    ==========================================================================================================================
    Which Machine LEarning Models Are Sensitive To Outliers
    
    
    Naivye Bayes Classifier--- Not Sensitive To Outliers
    
    
    SVM-------- Not Sensitive To Outliers
    
    
    Linear Regression---------- Sensitive To Outliers
    
    
    Logistic Regression------- Sensitive To Outliers
    
    
    Decision Tree Regressor or Classifier---- Not Sensitive
    
    
    Ensemble(RF,XGboost,GB)------- Not Sensitive
    
    
    KNN--------------------------- Not Sensitive
    
    
    Kmeans------------------------ Sensitive
    
    
    Hierarichal------------------- Sensitive
    
    
    PCA-------------------------- Sensitive
    
    
    Neural Networks-------------- Sensitive
    ''')
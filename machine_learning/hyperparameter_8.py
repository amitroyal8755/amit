def hyperopt():
    print(
    '''
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

=======================================================
=======================================================
space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 10, 1200, 10),#integer
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),#float
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
    }
========================================================
========================================================
def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
    accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }

==========================================================
==========================================================
from sklearn.model_selection import cross_val_score
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 80,
            trials= trials)
best
==========================================================
==========================================================

''')

def optuna():
    print('''
import optuna
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
def objective(trial):
    classifier = trial.suggest_categorical('classifier', ['ARDRegression','Lasso','LinearRegression','Ridge'])
    
    if classifier == 'ARDRegression':
        
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        n_iter=trail.suggest_int('n_iter',[i for i in linspace(100,2000,10)]),
        tol=trail.suggest_float('tol',[0.001,0.01,1,1e-10]),
        alpha_1.suggest_int('alpha_1',[1e-06,1e-07])


        clf_1=ARDRegression(n_estimators=n_estimators,n_iter=n_iter, tol= tol,alpha_1=alpha_1)
    
    elif classifier =='Lasso':
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        n_iter=trail.suggest_int('n_iter',[i for i in linspace(100,2000,10)]),
        tol=trail.suggest_float('tol',[0.001,0.01,1,1e-10]),
        alpha_1.suggest_int('alpha_1',[1e-06,1e-07])
       
        clf_2=Lasso(n_estimators=n_estimators,n_iter=n_iter, tol= tol,alpha_1=alpha_1)
    elif  classifier =='Lasso':
        
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        n_iter=trail.suggest_int('n_iter',[i for i in linspace(100,2000,10)]),
        tol=trail.suggest_float('tol',[0.001,0.01,1,1e-10]),
        alpha_1.suggest_int('alpha_1',[1e-06,1e-07])
        
        
        clf_3=LinearRegression(n_estimators=n_estimators,n_iter=n_iter, tol= tol,alpha_1=alpha_1)
    else:
        n_estimators = trial.suggest_int('n_estimators', 200, 2000,10)
        n_iter=trail.suggest_int('n_iter',[i for i in linspace(100,2000,10)]),
        tol=trail.suggest_float('tol',[0.001,0.01,1,1e-10]),
        alpha_1.suggest_int('alpha_1',[1e-06,1e-07])
        
        
        clf_3=Ridge(n_estimators=n_estimators,n_iter=n_iter, tol= tol,alpha_1=alpha_1)
    
    return sklearn.model_selection.cross_val_score(
        clf,X_train,y_train, n_jobs=-1, cv=3).mean()
        
        =========================================================================================
        =========================================================================================
        =========================================================================================
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        trial = study.best_trial

        print('Accuracy: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
        
        
        ========================================================================================
        ========================================================================================
        ========================================================================================
        study.best_params''')
def TPOTClassifier():
    print('''
    import numpy as np
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,14]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6,8]
# Create the random grid
param = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(param)


=======================================================================================
=======================================================================================
=======================================================================================

from tpot import TPOTClassifier


tpot_classifier = TPOTClassifier(generations= 5, population_size= 24, offspring_size= 12,
                                 verbosity= 2, early_stop= 12,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 4, scoring = 'accuracy')
tpot_classifier.fit(X_train,y_train)

=====================================================================================
=====================================================================================
tpot_classifier.best_param''')


def Random_search():
    print('''
    import numpy as np
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 1000,10)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10,14]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4,6,8]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                  'criterion':['entropy','gini']}
    print(random_grid)
    ==========================================================================================
    ==========================================================================================
    ==========================================================================================
    %%time
    rf=RandomForestClassifier()
    rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
    ### fit the randomized model
    rf_randomcv.fit(X_train,y_train)

    =========================================================================================
    =========================================================================================
    =========================================================================================
    rf_randomcv.best_params_
    best_random_grid=rf_randomcv.best_estimator_''')
    
    
def gridsearchCV():
    print('''
    from sklearn.model_selection import GridSearchCV
    
    it is similier to RandomSearchCV''')
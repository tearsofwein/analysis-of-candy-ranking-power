

   
    
    
import pandas as pd


import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn import linear_model


from sklearn.metrics import r2_score

from pylab import rcParams

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVR

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

## set output format
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Define the figure size
rcParams['figure.figsize'] = 6 ,6
rcParams['axes.linewidth'] = 1.8
#



## data balancing is VIP! it help a lot improving r2 score.
def data_balance( input_d ):
    global remove_n , reap_n , data_new, header_array
    ## transfer to pandas
    input_d = pd.DataFrame( input_d )
    ## assign column name
    input_d.columns = header_array[1:13]
    ## sort 
    input_d = input_d.sort_values( by="winpercent" , ascending = True)
    
    input_d = input_d.to_numpy()
#   
#    ## randomly repeat  data in first and last 20 rows
#    ## data copy to balance the data
     ###!!!! without balancing the data, r2 score is relatively very low.   
#    ## copy the first 20
    data_app = input_d[ 0:21 , 0: ]
#    data_app = data_new[ 0:21 , 0: ]
    input_d = np.vstack( ( data_app, input_d  ) )
    ## copy the last 25
    s_input_d = input_d.shape[0]
#    print( "s_input_d" , s_input_d )
    data_app = input_d[ (s_input_d-23):(s_input_d+2) , 0: ]
#    data_app = data_new[ 60:84 , 0: ]
#    print( input_d[ (s_input_d-23):(s_input_d) , 0: ] )
    input_d = np.vstack( ( input_d  ,data_app ) )
##  
##  
##    ## process of data augumentation
##    ## training set
    output_d = input_d
#    print( output_d )
    ## shuffle the data
    np.random.shuffle( output_d )

    return output_d

################################################################################
## data augumentation, is not used in the end. It provided the similar r2 score as without this technique.
## so this technique is not utilized.
def data_aug( in_d):
    global remove_n , reap_n , data_new
  
    add_n = 200
    alpha = 0.5
    s_input = in_d.shape[0]
#    print( "s_input" , s_input )
    for i in range(add_n):
        ## random lambda
        lam = np.random.beta( alpha , alpha )
        ## generate random rows 
        rnd1 = np.random.randint( 0,s_input + i -1 )
#        print( rnd1 )
        rnd2 = np.random.randint( 0,s_input + i -1 )
        data_add = lam * in_d[rnd1,0:] + ( 1 - lam ) * in_d[rnd2,0:]
        in_d = np.vstack( ( in_d , data_add ) )
    output_d = in_d
    return output_d
################################################################################



   
################################################################################ 
def model_train_test( Model , x_input, y_input , x_test , y_test ) :
    ##################################################################################
    Model.fit(  x_input, y_input   )
#    score_train = cross_val_score(Model, x_input , y_input , cv=5)
    ## cross validation
#    y_pred = cross_val_predict(Model, x_train, y_train, cv=5)
    ## output the coefficients
    if model_name=="GradientBoostingRegressor" :
        coeff = Model.feature_importances_
    else :
        coeff = Model.coef_
    ## trainset
    y_pred = Model.predict( x_input )
    
    
    ################# figure for training
#    fig = sns.scatterplot( y_input , y_pred )
#    sns.lineplot( y_input , y_input )
#    fig.set( xlim = [0.1,0.95] , ylim = [0.1,0.95] , xlabel = "real winpercent" , ylabel = "estimated winpercent" , title = "Trainset in "+model_name[0:9])
#    fig = plt.scatter( y_train , y_pred )
#    plt.xlabel( "real winpercent of trainset"  )
#    plt.ylabel( "estimated winpercent of trainset" )
    
    

    ## r2 score for training
    score_train = r2_score( y_input, y_pred ) 

    ## testset
    y_pred = Model.predict( x_test )
    score_test = r2_score( y_test, y_pred ) 
     
#    ################ figure for testing
#    fig = sns.scatterplot( y_test , y_pred )
#    sns.scatterplot( y_test , y_test )
#    fig.set( xlim = [0.1,0.95] , ylim = [0.1,0.95] , xlabel = "real winpercent" , ylabel = "estimated winpercent" , title = "Testset in "+model_name[0:9])

    
    ##Fake data
    x_fake = np.array([1,	1,	0	,1	,1	,1	,1	,1,	0	,0.96499997	,0.76700002]) #0.33

    x_fake = np.reshape( x_fake , (1,11))
#    print( x_fake.shape )
    y_pred_fake = Model.predict( x_fake )
    
    

    
#    print( "y_" , y_pred_fake )
    return score_train,score_test,y_pred_fake,coeff


def get_name(estimator):
    name = estimator.__class__.__name__
    if name == 'Pipeline':
        name = [get_name(est[1]) for est in estimator.steps]
        name = ' + '.join(name)
    return name



# Utility function to report best scores
def report(results, name , n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model "+name+" with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")




##############################################################################
## load the data
##############################################################################
## the header name 
## competitorname	chocolate	fruity	caramel	peanutyalmondy	nougat	crispedricewafer	hard	bar	pluribus	sugarpercent	pricepercent	winpercent

## load the data
data_org = pd.read_csv("candy-data.csv")
data_org = data_org.sort_values( by = "winpercent"  )
data_org["winpercent"] = data_org["winpercent"] / 100.0
### header 
header_array = data_org.columns
## plot orginal distribution


#######################################################################
## drop the product name
data_org = data_org.drop( columns = "competitorname" )

## transform pandas to numpy
data_org = data_org.to_numpy()

## originla data
data_new = data_org
## original data for x,y
DataX_org = data_org[ 0: , 0:11 ]
DataY_org = data_org[ 0: , 11]





##############################################################################

    
models = [
    (linear_model.LinearRegression( ), {  }),
    ( GradientBoostingRegressor( ), { "n_estimators": np.arange(10,110,11) , "learning_rate":np.logspace(-3, 1, 15) })
]
    
    
names = [get_name(e) for e, g in models]
rnd_itr = 3

linear_score_train = 0.0
boost_score_train = 0.0
SVR_score_train = 0.0

linear_score_test = 0.0
boost_score_test = 0.0
SVR_score_test = 0.0

for i in range(rnd_itr):
    for est_idx, (name, (estimator, param_grid)) in enumerate(zip(names, models)):
    
    
        x_train, x_test, y_train, y_test = train_test_split( DataX_org , DataY_org , test_size=0.2 , shuffle = True )
        ## reshape y_train
        y_train = np.reshape( y_train , ( y_train.size , 1 ) )
        ## join x and y train
        d_train = np.hstack( ( x_train , y_train  ) )
        
        ## balance trainset
        balanced_data = data_balance( d_train )
        auged_data = data_aug( balanced_data )
        
        ## split balanced data into features and lables 
        x_input = auged_data[ 0: , 0:11 ]
        y_input = auged_data[ 0: , 11]
        
      
       #######################################################################
        ## test and train dataset
            
            
        clf = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=5, iid=False)
        with ignore_warnings(category=ConvergenceWarning):
            clf.fit( x_input , y_input )
            y_pred = clf.predict( x_input )
            score_train = clf.score( x_input , y_input )
        report( clf.cv_results_ , name )    




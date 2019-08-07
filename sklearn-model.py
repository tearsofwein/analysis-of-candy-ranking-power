
   
    
    
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
def model_train( Model , x_input, y_input  ) :
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
    
    
    ## r2 score for training
    score_train = r2_score( y_input, y_pred ) 

    
#    print( "y_" , y_pred_fake )
    return score_train,coeff,Model,y_pred





################################################################################ 
def model_test( Model , x_test , y_test ) :
    ##################################################################################
    ## testset
    y_pred = Model.predict( x_test )
    score_test = r2_score( y_test, y_pred ) 
     
    
    ##Fake data
    x_fake = np.array([1,	1,	0	,1	,1	,1	,1	,1,	0	,0.96499997	,0.76700002]) #0.33

    x_fake = np.reshape( x_fake , (1,11))
#    print( x_fake.shape )
    y_pred_fake = Model.predict( x_fake )
    
    

    
#    print( "y_" , y_pred_fake )
    return score_test,y_pred_fake,Model,y_pred






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
model_linear = linear_model.LinearRegression( )    
    
#model_elasticNet = linear_model.ElasticNet( random_state=0 , alpha = 0.01 )

model_boost  = GradientBoostingRegressor( n_estimators = 25   )    

models = {model_linear  , model_boost }
#


for Model in models:
    ## get the model name
    model_name = type(Model).__name__
    print("model_name",model_name)
    ## random iteration 
    rnd_itr = 2
    ## initialize scores, coefficients and importance factors
    fn_score_train = 0.0   
    fn_score_test = 0.0
    fn_y_pred_fake = 0.0
    fn_coeff = 0.0
    ## load data and train the model
    for i in range(rnd_itr): 
        ## load the data

        #######################################################################
        ## test and train dataset
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
         ## train model 
        score_train , coeff, Model , y_pred = model_train( Model , x_input , y_input  )
        
        
        
        ################# figure for training
        fig1 = plt.figure(1)
        sns.scatterplot( y_input , y_pred )
        sns.lineplot( y_input , y_input )
        plt.xlim( 0.1,0.95 )
        plt.ylim( 0.1,0.95 )
        plt.xlabel( "real winpercent" )
        plt.ylabel( "estimated winpercent" )
        plt.title( "Trainset in "+model_name[0:9] )
#        fig1.set( xlim = [0.1,0.95] , ylim = [0.1,0.95] , xlabel = "real winpercent" , ylabel = "estimated winpercent" , title = "Trainset in "+model_name[0:9])
        
   
    
        
        score_test , y_pred_fake , Model , y_pred = model_test( Model , x_test , y_test )
       
        ################ figure for testing
        fig2 = plt.figure(2)
        sns.scatterplot( y_test , y_pred )        
        sns.lineplot( y_test , y_test )
        plt.xlim( 0.1,0.95 )
        plt.ylim( 0.1,0.95 )
        plt.xlabel( "real winpercent" )
        plt.ylabel( "estimated winpercent" )
        plt.title( "Testset in "+model_name[0:9] )
#        fig2.set( xlim = [0.1,0.95] , ylim = [0.1,0.95] , xlabel = "real winpercent" , ylabel = "estimated winpercent" , title = "Testset in "+model_name[0:9])

        
        
        
        
        
        ## accumulative scores, coeffients
        fn_score_train = score_train + fn_score_train
        fn_score_test = score_test + fn_score_test
        fn_y_pred_fake = fn_y_pred_fake + y_pred_fake
        fn_coeff = fn_coeff + coeff
    
    ## average scores and coefficient   
    av_score_train = fn_score_train / rnd_itr
    av_score_test = fn_score_test / rnd_itr
    av_score_fake =  fn_y_pred_fake / rnd_itr 
    av_coeff = fn_coeff / rnd_itr
    
    
    
        

        
        
#    print( x_test, y_test )
    
    print( "av_score_train ", av_score_train )    
    
    print( "av_score_test " , av_score_test )    
    
    print( "av_score_fake " , av_score_fake )   
    
    print( "av_coeff  " , av_coeff )

    fig1.savefig( model_name+"_train.png" , dpi = 300  )
    fig2.savefig( model_name+"_test.png" , dpi = 300  )
    
    plt.clf()
    
    if model_name=="GradientBoostingRegressor":
        # make importances relative to max importance
        feature_importance = 100.0 * av_coeff
        ## sort importance values
        sorted_idx = np.argsort(feature_importance)

        pos = np.arange(sorted_idx.shape[0]) + .5
        ## importance bar
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        ## place labels
        plt.yticks(pos, header_array[sorted_idx+1])
        plt.xlabel('Feature Importance')
        plt.savefig( model_name+"_importance.png" , bbox_inches='tight' , dpi = 300 )
        plt.clf()
        
        
                ## save coefficients to csv
        cof = np.reshape(av_coeff,(1,11))
        cof = pd.DataFrame( cof  )

        ## add header
        cof.to_csv( "boost.csv" , header =  header_array[1:12] )

        
    
    if model_name=="LinearRegression": 
        sorted_idx = np.argsort(av_coeff)
        pos = np.arange(sorted_idx.shape[0]) + .5
        ## importance bar
        plt.barh(pos, av_coeff[sorted_idx], align='center')
        ## place labels
        plt.yticks(pos, header_array[sorted_idx+1])
        plt.xlabel('tangent')
        plt.savefig( model_name+"_parameter.png" , bbox_inches='tight' , dpi = 300 )
        plt.clf()

        ## save coefficients to csv
        cof = np.reshape(av_coeff,(1,11))
        cof = pd.DataFrame( cof  )

        ## add header
        cof.to_csv( "linear.csv" , header =  header_array[1:12] )


###############################################################################

    
    
    
    
    






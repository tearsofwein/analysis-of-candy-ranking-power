import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

## set output format
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Define the figure size
rcParams['figure.figsize'] = 6, 6
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
#    print( data_app )
#    data_app = data_new[ 0:21 , 0: ]
    input_d = np.vstack( ( data_app, input_d  ) )
    ## copy the last 25
    s_input_d = input_d.shape[0]
#    print( "s_input_d" , s_input_d )
    data_app = input_d[ (s_input_d-26):(s_input_d-1) , 0: ]
#    print( data_app )
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



    
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        ## hidden layer
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
#        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer

        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        ## dropout to solve overfitting
#        dp = torch.nn.Dropout(0.1)
#        x = dp(x)
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
#        x = F.relu(self.hidden3(x))      # activation function for hidden layer
#        x = dp(x)
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=11, n_hidden1=11, n_hidden2 = 11,  n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2 )
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
################################################################################


score_test = 0.0
score_train = 0.0
y_pred_fake = 0.0


##  iteration 
rnd_itr = 30
for i in range(rnd_itr): 
    ## data for training and testing
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
    
    x_input = auged_data[ 0: , 0:11 ]
    y_input = auged_data[ 0: , 11]
    
    
    ## transform to tensor format
    x_input = torch.tensor( x_input , dtype = torch.float32 )
    y_input = torch.tensor( [y_input] , dtype = torch.float32 )
    x_test = torch.tensor( x_test , dtype = torch.float32 )
    y_test = torch.tensor( [y_test] , dtype = torch.float32 )
    
#    ## transform shape
    y_input = torch.transpose( y_input , 0 , 1 )
    y_test = torch.transpose( y_test , 0 , 1 )

    
    t_step = 1000
    loss1 = np.zeros( (t_step) )
    loss2 = np.zeros( (t_step) )
    ### train the model
    for t in range(t_step):
        prediction = net(x_input)     # input x and predict based on x

        loss = loss_func( prediction, y_input )          
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        loss1[t] = loss.data.numpy()
        
        
        y_pred = net( x_test )

        loss_test = loss_func( y_pred, y_test )   
        loss2[t] = loss_test.data.numpy()


#####we comment this code, since it will interfere the plot of the next plot
###############################################################################
######### plot the train step vs loss 
        
    fig1 = plt.figure(1)
    plt.plot( np.arange(0,t_step,1) , loss1 , label = "train_set" )
    plt.xlabel( "running steps" )
    plt.ylabel( "losses" )
    plt.title( "lr=0.2" )
    plt.ylim( 0 , 0.04 ) 
###############################################################################
######### plot the test step vs loss 
    fig2 = plt.figure(2)
    plt.plot( np.arange(0,t_step,1) , loss2 , label = "test_set")
    plt.savefig( "pytorch-op-test_steps"+str(i)+".png" , dpi = 300 )  
    plt.clf()
###############################################################################

        

    
    
    ## predict the trainset for figure plotting
    y_pred = net( x_input )
    ## transform to numpy
    x_data = x_input.data.numpy()
    y_data = y_input.data.numpy()
    y_pred        = y_pred.data.numpy()


###############################################################################
#####we comment this code, since it will interfere the plot of the next plot
#    ## plot the real-pred
    fig3 = plt.figure(3)
    sns.scatterplot( y_data.flatten() , y_pred.flatten() )
    sns.lineplot( y_data.flatten() , y_data.flatten()  )
    plt.xlim( 0.1,0.95 )
    plt.ylim( 0.1,0.95 )
    plt.xlabel( "real winpercent" )
    plt.ylabel( "estimated winpercent" )
    plt.title( "Trainset in NN" )
   
###############################################################################
    
    
    ## print the train score

    score_train1 = r2_score( y_data.flatten() , y_pred.flatten() )
    score_train = score_train1 + score_train
    print( "score of trainset" , score_train1)
    
    

#    ## predict the testset
    y_pred = net( x_test )
    ## transform to numpy
    x_test = x_test.data.numpy()
    y_test = y_test.data.numpy()
    y_pred        = y_pred.data.numpy()
    
    
################################################################################    
###    ## plot the real-pred
    fig4 = plt.figure(4)
    sns.scatterplot( y_test.flatten() , y_pred.flatten() )
    sns.lineplot( y_test.flatten() , y_test.flatten() )
    plt.xlim( 0.1,0.95 )
    plt.ylim( 0.1,0.95 )
    plt.xlabel( "real winpercent" )
    plt.ylabel( "estimated winpercent" )
    plt.title( "Testset in NN" )
 ################################################################################
    
    ### print the test score
    score_test1 = r2_score( y_test.flatten() , y_pred.flatten() )
    score_test = score_test1 + score_test
    print( "score of testset" , score_test1 )
    
    ## fake data for real prediction
    x_data_fake = torch.tensor( [[1	,0,	0,	0	,1	,0	,0	,1,	0,	0.60399997,	0.51099998]], dtype = torch.float32 )
    y_pred_fake = net( x_data_fake ) + y_pred_fake
    
    
    
fig3.savefig( "NN_train.png" , dpi = 300  )
fig4.savefig( "NN_test.png" , dpi = 300  )
plt.clf()


## obtain and print the average train and test score
ave_score_test = score_test / rnd_itr
ave_score_train = score_train / rnd_itr
ave_y_pred_fake = y_pred_fake / rnd_itr
print( "ave_score_test" , ave_score_test )
print( "ave_score_train" , ave_score_train )
print( "y_pred_fake" , ave_y_pred_fake )
##
#
#    
    
    
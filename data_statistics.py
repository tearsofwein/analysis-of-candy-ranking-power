
   
    
    
import pandas as pd


import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from pylab import rcParams



## set output format
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# Define the figure size
rcParams['figure.figsize'] = 8 , 8
rcParams['axes.linewidth'] = 1.8
#

def org_plot(data_org) :
    global header_array
    sns.distplot( data_org["winpercent"] , hist=True )
    plt.savefig( "win-hist.png" , dpi = 300   )
    plt.clf()
    
    
#    
#    
    ################################################################################
    #### plot of the original data
    ##################################################################################
    for header_name in header_array[1:10]  : 
        sns.relplot( x="pricepercent", y="winpercent", hue= header_name , style = header_name , data=data_org  );      
        sns.set( font_scale=1.6 )
#        plt.legend(loc='upper left')
        plt.savefig( header_name+".png" , dpi = 300   )
        plt.clf()   
#    
    
    
     ## plot the joint figure of suger or pricecpercent with regarding to the winpercent
    for header_name in header_array[10:12]  : 
    #    fig = sns.relplot( x="pricepercent", y="winpercent", hue= header_name , style = header_name , data=data_org); 
        fig = sns.jointplot( x=header_name, y="winpercent", data=data_org, kind="kde", color="m")
        sns.set( font_scale=1.6 )
        fig.plot_joint(plt.scatter, c="w", marker="+"); 
        fig.ax_joint.collections[0].set_alpha(0)
        plt.savefig( header_name+"_dist.png" , dpi = 300  , bbox_inches='tight'  )
        plt.clf()    
    
    ###############################################################################
    #### conclusion: chocolate, fruity, hard---apparent influence, others not so apparent.
    ###  chocolate and fruity does not appear at the same time, expect "Tootsie Pop". Or they both don't appear at all.            
    ###  hard and bar does not appear at the same time. Or they both don't appear at all.            
    ###  nougut and crispedricewafer does not appear at the same time. Or they both don't appear at all.            
    ###  fruity and caramel does not appear at the same time, expect "Caramel Apple Pops". Or they both don't appear at all.            
    ###############################################################################


#
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
org_plot(data_org)



########################################################################



#######################################################################
# calculate correlations
corr = data_org.corr()
## sort correlations
corr2 = corr.sort_values( by="winpercent" , ascending=False )
## plot correlations
fig = sns.heatmap( corr2 )
fig.set( title="correlation map" )
plt.savefig( "correlation.png" , bbox_inches='tight' , dpi = 300 )
plt.clf()



    
    
    






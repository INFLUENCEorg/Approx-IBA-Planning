
import numpy as np
import yaml
from environments.TrafficGrid.Global_simulator import Traffic_env
from environments.TrafficGrid.agents import traffic_agent
from environments.TrafficGrid.experimentor import Approximator
from environments.TrafficGrid.IALM import IALM
import pylab
import numpy as np
import pandas as pd

def run_TG(parameters):
    

    env=Traffic_env(parameters,verb=0)
    
    '''
    prev_s=np.random.choice([0,1],size=[4,8])
    a=np.random.choice([0,1],size=[4,1])
    
    env.plot_state(prev_s,a)
    local_s=env.get_local_state(prev_s)
    env.plot_local_state(local_s[0],a[0])
    '''

    policies=[]
    for i in range(len(parameters['policies'])):
        policies.append(traffic_agent(parameters['policies'][i]).policy)
    
    env.run_simulation(policies)
   
    Approx=Approximator(env,parameters,policies)

    #Compute statistics
    Norm1={}
    Cross_Entropy={}
    Value={}
    Training_loss={}

    path=parameters['path']

    for i in range(parameters['n_iter_training']):


        N1, CE, V, training_loss = Approx.train_error() 
        

        Norm1['Iter_'+str(i)]=N1
        Cross_Entropy['Iter_'+str(i)]=CE
        Value['Iter_'+str(i)]=V
        Training_loss['Iter_'+str(i)]=training_loss
        
        
        pylab.figure()
        pylab.plot(np.arange(parameters['n_epochs'],step=parameters['frac_epochs']),V, label='Optimal approximate-influence value')
        #pylab.plot(np.arange(parameters['n_epochs']),[Optimal_value]*len(V), label='Optimal exact-influence value')
        pylab.legend(loc='lower right')
        pylab.xticks(np.arange(parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        #pylab.ylim((-0.05,0.75))
        pylab.xlabel('Epochs')
        pylab.ylabel('Value')
        #pylab.show()
        pylab.savefig(path+'/Value '+str(i))
        

        pylab.figure()
        pylab.plot(np.arange(parameters['n_epochs']),N1, label='Norm1')
        pylab.plot(np.arange(parameters['n_epochs']),CE, label='Cross entropy')
        pylab.legend(loc='lower right')
        pylab.xticks(np.arange(parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.xlabel('Epochs')
        pylab.ylabel('Errors')
        #pylab.ylim((0,1.2))
        #pylab.show()
        pylab.savefig(path+'/Test Errors '+str(i))

        pylab.figure()
        pylab.plot(np.arange(parameters['n_epochs']),training_loss)
        pylab.xticks(np.arange(parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.xlabel('Epochs')
        pylab.ylabel('Training loss')
        #pylab.ylim((0,1.2))
        #pylab.show()
        pylab.savefig(path+'/Loss '+str(i))
        
        print('Round ', i+1)

    Norm1=pd.DataFrame(Norm1)
    Cross_Entropy=pd.DataFrame(Cross_Entropy)
    Value=pd.DataFrame(Value)
    Training_loss=pd.DataFrame(Training_loss)

    Norm1['mean']=Norm1.mean(axis=1)
    Cross_Entropy['mean']=Cross_Entropy.mean(axis=1)
    Value['mean']=Value.mean(axis=1)
    Training_loss['mean']=Training_loss.mean(axis=1)

    Norm1['std']=Norm1.std(axis=1)
    Cross_Entropy['std']=Cross_Entropy.std(axis=1)
    Value['std']=Value.std(axis=1)
    Training_loss['std']=Training_loss.std(axis=1)
    
    #Value['Optimal']= [Optimal_value]* len(V)

    Norm1.to_csv(path+'Norm1.csv')
    Cross_Entropy.to_csv(path+'Cross_Entropy.csv')
    Value.to_csv(path+'Value.csv')
    Training_loss.to_csv(path+'/Training_loss.csv')

    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs'],step=parameters['frac_epochs']),Value['mean'], label='Optimal approximate-influence value')
    #pylab.plot(np.arange(parameters['n_epochs']),Value['Optimal'], label='Optimal exact-influence value')
    pylab.fill_between(np.arange(parameters['n_epochs'],step=parameters['frac_epochs']),Value['mean']-Value['std'],Value['mean']+Value['std'],alpha=0.3)
    pylab.legend(loc='lower right')
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Value')
    pylab.savefig(path+'/Value')
    #pylab.show()

        
    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs']),Norm1['mean'], label='Norm1')
    pylab.plot(np.arange(parameters['n_epochs']),Cross_Entropy['mean'], label='Cross entropy')
    pylab.fill_between(np.arange(parameters['n_epochs']), Norm1['mean']-Norm1['std'],Norm1['mean']+Norm1['std'],alpha=0.3)
    pylab.fill_between(np.arange(parameters['n_epochs']),Cross_Entropy['mean']-Cross_Entropy['std'],Cross_Entropy['mean']+Cross_Entropy['std'],alpha=0.3)
    pylab.legend(loc='lower right')
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Errors')
    pylab.savefig(path+'/TestErrors')
    #pylab.show()


    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs']),Training_loss['mean'])
    pylab.fill_between(np.arange(parameters['n_epochs']),Training_loss['mean']-Training_loss['std'],Training_loss['mean']+Training_loss['std'],alpha=0.3)
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Training loss')
    #pylab.show()
    pylab.savefig(path+'/Loss')
    





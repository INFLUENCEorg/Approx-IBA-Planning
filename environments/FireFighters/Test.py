
from environments.FireFighters.simulator import ffg_env
import numpy as np
from environments.FireFighters.agent import agent
from environments.FireFighters.experimentor import Approximator
import pylab
import pandas as pd
import time

def run_FF(parameters):

    path=parameters['path']

    policies=[]

    policies.append(agent(parameters['policies'][0]).policy)
    policies.append(agent(parameters['policies'][1]).policy)

    env=ffg_env(parameters)
    env.run_simulation(policies)

    #TODO: Compute Optimal value
    #Optimal_value=

    #Compute statistics
    Norm1={}
    Cross_Entropy={}
    Value={}
    Training_loss={}
    
    for i in range(parameters['n_iter_training']):
        T=time.time()
        Approx=Approximator(env, parameters)

        N1, CE, V, training_loss = Approx.train_error(policies[0]) 
        
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
        print( 'Time ', time.time()-T)
    
    #Value['Optimal']= [Optimal_value]* len(V)

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

    Norm1.to_csv(path+'/Norm1.csv')
    Cross_Entropy.to_csv(path+'/Cross_Entropy.csv')
    Value.to_csv(path+'/Value.csv')
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
    pylab.savefig(path+'Test_errors.png')
    #pylab.show()


    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs']),Training_loss['mean'])
    pylab.fill_between(np.arange(parameters['n_epochs']),Training_loss['mean']-Training_loss['std'],Training_loss['mean']+Training_loss['std'],alpha=0.3)
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Training loss')
    pylab.savefig(path+'/Training_loss.png')
    #pylab.show()
    


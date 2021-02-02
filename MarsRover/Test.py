
import numpy as np
from MarsRover import MR_env
from agents import agent_satellite, agent_rover
import argparse
from experimentor import Approximator
import pylab
import pandas as pd
from IALM import IALM
import yaml

def main(parameters):

    policies=[]

    policies.append(agent_satellite(parameters).policy)
    policies.append(agent_rover(parameters).policy)
    
    #build the environment
    env=MR_env(parameters)

    #run the simulation
    #save the simulation in env.S, env.A, env.R, env.V
    env.run_simulation(policies)
    
   
    #compute exact influence
    #save the Exact influence as a lenght(hor) list env.Exact_IP
    env.exact_influence(policies[0])

    #built the IALM with the exact influence  
    IALM_exact=IALM(parameters, env.Exact_IP)

    #Local agent optimal policy and optimal value
    Optimal_policy,Optimal_value=IALM_exact.value_iteration()

    #Compute statistics
    Norm1={}
    Cross_Entropy={}
    Value={}
    Training_loss={}

    path=parameters['path_name']

    for i in range(parameters['n_iter_training']):

        Approx=Approximator(env, parameters)

        N1, CE, V, training_loss ,Approx_I, Pi= Approx.train_error(policies[0],env.Exact_IP) 
        
        '''
        print('Value', V[-1])
        print('Optimal value', Optimal_value)
        for t in range(parameters['hor']):
            S_IALM,_=IALM_exact.state_space(parameters['hor']-t)
            print('Time ', t)
            print('Approximate inf', Approx_I[t])
            print('Exact inf', env.Exact_IP[t])
            indexes=np.where(np.array(Optimal_policy[-t-1])!=np.array(Pi[-t-1]))[0]
            if len(indexes)>0:
                print('true optimal action', np.array(Optimal_policy[-t-1])[indexes])
                print('action chosen by approximate policy', np.array(Pi[-t-1])[indexes])
                print('states on which policies are differenti', np.array(S_IALM)[indexes])
                input()
        '''

        Norm1['Iter_'+str(i)]=N1
        Cross_Entropy['Iter_'+str(i)]=CE
        Value['Iter_'+str(i)]=V
        Training_loss['Iter_'+str(i)]=training_loss
        
        pylab.figure()
        pylab.plot(np.arange(parameters['n_epochs']),V, label='Optimal approximate-influence value')
        pylab.plot(np.arange(parameters['n_epochs']),[Optimal_value]*len(V), label='Optimal exact-influence value')
        pylab.legend(loc='lower right')
        pylab.xticks(np.arange(parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.ylim((-0.05,0.75))
        pylab.xlabel('Epochs')
        pylab.ylabel('Value')
        pylab.savefig(path+'Value '+str(i))


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
        pylab.savefig(path+'Test Errors '+str(i))

        pylab.figure()
        pylab.plot(np.arange(parameters['n_epochs']),training_loss)
        pylab.xticks(np.arange(parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.xlabel('Epochs')
        pylab.ylabel('Training loss')
        #pylab.ylim((0,1.2))
        #pylab.show()
        pylab.savefig(path+'Loss '+str(i))
        
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
    
    Value['Optimal']= [Optimal_value]* len(V)

    Norm1.to_csv(path+'Norm1.csv')
    Cross_Entropy.to_csv(path+'Cross_Entropy.csv')
    Value.to_csv(path+'Value.csv')
    Training_loss.to_csv(path+'Training_loss.csv')

    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs']),Value['mean'], label='Optimal approximate-influence value')
    pylab.plot(np.arange(parameters['n_epochs']),Value['Optimal'], label='Optimal exact-influence value')
    pylab.fill_between(np.arange(parameters['n_epochs']),Value['mean']-Value['std'],Value['mean']+Value['std'],alpha=0.3)
    pylab.legend(loc='lower right')
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Value')
    pylab.savefig(path+'Value')

        
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
    pylab.savefig(path+'TestErrors')


    pylab.figure()
    pylab.plot(np.arange(parameters['n_epochs']),Training_loss['mean'])
    pylab.fill_between(np.arange(parameters['n_epochs']),Training_loss['mean']-Training_loss['std'],Training_loss['mean']+Training_loss['std'],alpha=0.3)
    pylab.xticks(np.arange(parameters['n_epochs'],step=1))
    pylab.grid(linestyle='--',alpha=0.8)
    pylab.xlabel('Epochs')
    pylab.ylabel('Training loss')
    pylab.savefig(path+'Loss')
        

def get_config_file():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, help='config file')
    args = parser.parse_args()
    return args.config

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']

if __name__ == "__main__":
    config_file = get_config_file()
    parameters = read_parameters(config_file)
    main(parameters)





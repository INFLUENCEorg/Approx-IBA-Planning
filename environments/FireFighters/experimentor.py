import numpy as np
import os
from environments.FireFighters.prediction import model
import pylab
import pandas as pd
from environments.FireFighters.IALM import IALM
from environments.FireFighters.utility import scomposition
import time

np.random.seed(42)

class Approximator(object):

    def __init__(self, env, parameters):
        self.parameters=parameters
        self.env=env

    
    def run(self):

        #Run the simulation
        #Run the simulation
        #save_obj(self.ffg, filename=self.model_path+"/ffg.bin", protocol=3)

        n_epochs=self.parameters['n_epochs']
        batch_size=self.parameters['batch_size']
        n_iter=self.parameters['n_iter']
        assert batch_size <= n_iter, 'Iterations lower then batch size'

        # Create the dataset (n_iter x hor x dimension of d_set) and the labels (n_iter x hor)
        #TODO: labels (n_iter x hor x dimension sources of influence)
        data=self.env.d_set()
        labels=self.env.sources_influence()

        data=np.array(data)
        labels=np.array(labels)


        self.model = model(self.parameters)

        #Select randomly train_test_split of the experience as train set and the rest as test set
        indexes=np.arange(len(data))
        np.random.shuffle(indexes)
        indexes_train=indexes[:int(self.parameters['train_test_split']*n_iter)]
        indexes_test=indexes[int(self.parameters['train_test_split']*n_iter):]

        train_x = data[indexes_train]
        train_lab = labels[indexes_train]

        test_x = data[indexes_test]
        test_lab = labels[indexes_test]

        total_batches = len(train_x) // batch_size
        loss = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            loss_epoch=0
            for b in range(total_batches):
                index=b*batch_size
                batch_x=train_x[index:index+batch_size]
                batch_lab=train_lab[index:index+batch_size]
                batch={'batch_x':batch_x,'labels':batch_lab}
                loss_step, pred_lab =self.model.update(batch)
                loss_epoch+=loss_step
            loss_epoch=loss_epoch/total_batches
            loss[epoch]=loss_epoch

        pylab.figure()
        pylab.plot(loss)
        pylab.xlabel('Epochs')
        pylab.ylabel('Loss')
        pylab.title('Average Loss')
        #pylab.savefig(self.result_path+'/loss.jpg')
        pylab.show()
                

        cross_entropy, norm_1, accuracy, conf_matrix, precision, recall, k_stat, _, _ , _, _,_=self.model.evaluate(train_x, train_lab)


        print('Cross entropy training set', cross_entropy)

        cross_entropy, norm_1, accuracy, conf_matrix, precision, recall, k_stat, baseline_accuracy, random_accuracy, baseline_norm_1, random_norm_1, random_cross_entropy= self.model.evaluate(test_x, test_lab)

        print('\nReal measures')
        print('test cross entropy', cross_entropy)
        print('norm 1', norm_1)
        print('baseline norm 1', baseline_norm_1)
        print('random cross entropy', random_cross_entropy)
        print('random norm 1', random_norm_1)


    def train_error(self, policy):
        #All possible instatiation of the d_set is a list len= hor dim(D_set[t]) = n_possible d_set at time t x hor x n_factors in the d_set (3)
        self.D_set=self.generate_d_set()

        n_epochs=self.parameters['n_epochs']
        batch_size=self.parameters['batch_size']
        n_iter=self.parameters['n_iter']
        self.dim_d_set=self.parameters['dim_d_set']
        self.n_classes=self.parameters['n_classes']

        # Create the dataset (n_iter x hor x dimension of d_set) and the labels (n_iter x hor)
        data=self.env.extract_d_set()
        labels=self.env.extract_sources_influence()

        #build the network
        self.model = model(self.parameters)
        
        data=np.array(data)
        labels=np.array(labels)

        #Select train_test_split of the experience as train set and the rest as test set
        indexes=np.arange(len(data))
        #np.random.shuffle(indexes)
        indexes_train=indexes[:int(self.parameters['train_test_split']*n_iter)]
        indexes_test=indexes[int(self.parameters['train_test_split']*n_iter):]
        train_x = data[indexes_train]
        train_lab = labels[indexes_train]
        test_x = data[indexes_test]
        test_lab = labels[indexes_test]
        #print(train_x[:10])
        #print(train_lab[:10])
        #input()
        #print(test_x[:10])
        #print(test_lab[:10])
        #input()
        total_batches = len(train_x) // batch_size
        loss = np.zeros(n_epochs)
        
        CE=[]
        N1=[]
        V=[]

     
        for epoch in range(n_epochs):
            loss_epoch=0
            #Compute the approximate influence given the current state of the network
            #Approx_I is a len=hor list. dim(Approx_I[t])=dim(d_set_t) x dim(sources_influence)

            if epoch%self.parameters['frac_epochs']==0:
                Approx_I=self.approximate_influence()

                #Build the approximat influence IALM
                IALM_FF=IALM(self.parameters, Approx_I, policy)
                #Compute the Approx_I optimal policy
                Pi=IALM_FF.value_iteration()
                #Compute the value of the Approx_I optimal policy in the Exact_I IALM
                V.append(IALM_FF.evaluate_IALM_pi_global_env(Pi, policy, 500))
                #Evaluate the performances of the network
            
            cross_entropy, norm_1=self.model.evaluate(test_x,test_lab)
            CE.append(cross_entropy)
            N1.append(norm_1)
            
            #Training step
            for b in range(total_batches):
                index=b*batch_size
                batch_x=train_x[index:index+batch_size]
                #print(batch_x[:10])
                batch_lab=train_lab[index:index+batch_size]
                #print(batch_lab[:10])
                #input()
                batch={'batch_x':batch_x,'labels':batch_lab}
                loss_step, pred_lab =self.model.update(batch)
                loss_epoch+=loss_step
            loss_epoch=loss_epoch/total_batches
            loss[epoch]=loss_epoch
        '''
        print(self.approximate_influence()[0])
        print(CE)
        print(loss)
        input()
        pylab.figure()
        pylab.plot(np.arange(self.parameters['n_epochs']),N1, label='Norm1')
        pylab.plot(np.arange(self.parameters['n_epochs']),CE, label='Cross entropy')
        #pylab.fill_between(np.arange(parameters['n_epochs']), Norm1['mean']-Norm1['std'],Norm1['mean']+Norm1['std'],alpha=0.3)
        #pylab.fill_between(np.arange(parameters['n_epochs']),Cross_Entropy['mean']-Cross_Entropy['std'],Cross_Entropy['mean']+Cross_Entropy['std'],alpha=0.3)
        pylab.legend(loc='lower right')
        pylab.xticks(np.arange(self.parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.xlabel('Epochs')
        pylab.ylabel('Errors')
		#pylab.savefig(path+'Test_errors_'+str(i)+'.png')
        pylab.show()

        pylab.figure()
        pylab.plot(np.arange(self.parameters['n_epochs']),loss)
	    #pylab.fill_between(np.arange(parameters['n_epochs']),Training_loss[i]-Training_loss['std'],Training_loss['mean']+Training_loss['std'],alpha=0.3)
        pylab.xticks(np.arange(self.parameters['n_epochs'],step=1))
        pylab.grid(linestyle='--',alpha=0.8)
        pylab.xlabel('Epochs')
        pylab.ylabel('Training loss')
	    #pylab.savefig(path+'Training__'+str(i)+'.png')
        pylab.show()
        #print(self.approximate_influence()[1])
        input()
        '''
        return N1, CE, V, loss


    def approximate_influence(self):
        #Compute the approximate_influence for any D_set 
        I=[]
        dim_d_set=self.env.f_lev*2*self.env.f_lev
        dim_s_inf=self.env.f_lev
        I.append(self.env.b_init[0])
        for t in range(1,self.env.hor):
            I.append([])
            oss=np.zeros((len(self.D_set[t]),self.env.hor-1,self.dim_d_set))
            oss[:,:t,:]=self.D_set[t]
            a_distr=self.model.predictions(oss)
            a_distr=np.array(a_distr[0])
            a_distr=np.reshape(a_distr,(np.shape(oss)[0],np.shape(oss)[1],self.n_classes))
            I[t]=a_distr[:,t-1,:]
        return I


    def generate_1_step_d_set(self):
    #Generate all the possible instatiation fo one time step d_set in the binary order 0,0,0; 0,0,1...
        D_set_1_step=[]
        for i in range(self.env.f_lev):
            for j in range(2):
                for k in range(self.env.f_lev):
                    D_set_1_step.append(np.array([i,j,k]))
        return np.array(D_set_1_step)

    def index_to_dset(self, i, t):
        dim_d_set=(self.env.f_lev*2*self.env.f_lev)
        dset_index=scomposition(i,dim_d_set,[])
        while(len(dset_index))<=t:
            dset_index.insert(0,0)
        return np.array(dset_index)
        
    def generate_d_set(self):
    #Generate all the possible instantiation of the D_set
    # D set (t) to predict source influence time t (D(0) is empty) 
        T=time.time()
        D_set=[]
        dim_d_set=(self.env.f_lev*2*self.env.f_lev)
        for t in range(self.env.hor-1):
            D_set.append([])
            for i in range(dim_d_set**(t+1)):
                dset_index=self.index_to_dset(i,t)
                D_set[t].append([])
                for j in dset_index:
                    D_set[t][i].append((self.generate_1_step_d_set()[j]))
        T=time.time()-T
        D_set.insert(0,[])
        print('Time to compute all instatiations of the D_set', T)
        return np.array(D_set)
        














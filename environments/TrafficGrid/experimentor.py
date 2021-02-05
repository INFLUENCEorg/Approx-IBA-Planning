import numpy as np
import os
from environments.TrafficGrid.prediction import model
import pylab
import pandas as pd
from environments.TrafficGrid.IALM import IALM
from environments.TrafficGrid.utility import generate_d_set

np.random.seed(42)

class Approximator(object):

    def __init__(self, env, parameters, policies):

        self.parameters=parameters
        self.env=env
        self.D=self.generate_d_set()
        self.policies=policies
       

    def train(self):
        n_epochs=self.parameters['n_epochs']
        batch_size=self.parameters['batch_size']
        n_iter=self.parameters['n_iter']
        #frac_epochs=self.parameters['frac_epochs']

        # Create the dataset (n_iter x hor x dimension of d_set) and the labels (n_iter x hor)
        data=self.env.extract_d_set()
        labels=self.env.extract_sources_influence()
        
        self.model = model(self.parameters)
        
        data=np.array(data)
        labels=np.array(labels)

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

        #pylab.figure()
        #pylab.plot(loss)
        #pylab.xlabel('Epochs')
        #pylab.ylabel('Loss')
        #pylab.title('Average Loss')
        #pylab.savefig('Results/TrainingLoss.jpg')
        #pylab.show()

        
        
    def train_error(self):
        n_epochs=self.parameters['n_epochs']
        batch_size=self.parameters['batch_size']
        n_iter=self.parameters['n_iter']
        frac_epochs=self.parameters['frac_epochs']
        #frac_epochs=(n_epochs-1)/n_VI


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
        
        total_batches = len(train_x) // batch_size
        loss = np.zeros(n_epochs)
        loss_1=np.zeros([n_epochs,2])

        CE=[]
        N1=[]
        V=[]
        

        #Build the exact influence IALM where to evaluate the performances of the approximate-influence optimal policy
        #Exact_IALM=IALM(self.parameters,Exact_I)
     
        for epoch in range(n_epochs):
            loss_epoch=0
            loss_epoch_1=np.array([0,0])

            #Compute the approximate influence given the current state of the network
            #Approx_I is a len=hor list. dim(Approx_I[t])=dim(d_set_t) x dim(sources_influence)
            if epoch%frac_epochs==0:
            #    if epoch==n_epochs-1:
            #        verb=1
            #    else:
            #        verb=0
                #len(Approx_I)=hor np.shape(Approx_I[t])= n_dsets[t] x n_inf_sourse(2) x n_values_inf_sources(2) 
                Approx_I=self.approximate_influence()

                #Build the approximat influence IALM
                IALM_traffic=IALM(self.parameters, Approx_I)

                #Compute the Approx_I optimal policy
                Pi=IALM_traffic.value_iteration()

                #Compute the value of the Approx_I optimal policy in the Exact_I IALM
                V.append(IALM_traffic.evaluate_IALM_policy(1000, Pi, self.policies[1:]))
            
            #Evaluate the performances of the network
            cross_entropy, norm_1=self.model.evaluate(test_x,test_lab)

            CE.append(cross_entropy)
            N1.append(norm_1)

            #Training step
            for b in range(total_batches):
                index=b*batch_size
                batch_x=train_x[index:index+batch_size]
                batch_lab=train_lab[index:index+batch_size]
                batch={'batch_x':batch_x,'labels':batch_lab}
                loss_step, loss_1_step, pred_lab =self.model.update(batch)
                loss_epoch_1=loss_1_step+loss_epoch_1
                loss_epoch+=loss_step

            loss_1[epoch]=loss_epoch_1/total_batches
            
            loss_epoch=loss_epoch/total_batches
            loss[epoch]=loss_epoch

        '''
        pylab.figure()
        pylab.plot(loss)
        pylab.xlabel('Epochs')
        pylab.ylabel('Loss')
        pylab.title('Average Loss')
        pylab.savefig('Results/TrainingLoss.jpg')
        #pylab.show()

        pylab.figure()
        pylab.plot(loss_1[:,0])
        pylab.xlabel('Epochs')
        pylab.ylabel('Loss')
        pylab.title('Average Loss  east')
        pylab.savefig('Results/TrainingLoss.jpg')

        pylab.figure()
        pylab.plot(loss_1[:,1])
        pylab.xlabel('Epochs')
        pylab.ylabel('Loss')
        pylab.title('Average Loss north')
        pylab.savefig('Results/TrainingLoss.jpg')
        pylab.show()
        '''
        return N1, CE, V, loss


    def generate_d_set(self):

        D=np.zeros([4**self.parameters['hor'],self.parameters['hor'],2])
        for i in range(4**self.parameters['hor']):
            D_set = "{0:b}".format(i)
            D_set = np.array([int(j) for j in D_set])
            while(len(D_set)<2*self.parameters['hor']):
                D_set=np.insert(D_set,0,0,axis=0)
            D_set=np.reshape(D_set,[self.parameters['hor'],2])
            D[i]=D_set

        return D

    def approximate_influence(self):
    	#Compute the approximate_influence for any D_set
        I=[]
        influences=self.model.predictions(self.D)
        #print('around[0,1]',influences[198][0][0])

        for t in np.arange(self.parameters['hor']-1,-1, step=-1):
            n_dsets=4**(t+1)
            rev_index=self.parameters['hor']-t-1
            I.append(np.zeros((n_dsets,2,2)))
            indexes=np.arange(0,4**self.parameters['hor'], step=4**rev_index)

            I[rev_index]=influences[indexes,t]
         
        I.reverse()
        return I
        




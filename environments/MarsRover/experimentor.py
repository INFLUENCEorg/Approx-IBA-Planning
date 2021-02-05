from environments.MarsRover.prediction import model
from environments.MarsRover.IALM import IALM
from environments.MarsRover.utility import generate_d_set
import numpy as np

np.random.seed(42)

class Approximator(object):

    def __init__(self, env, parameters):

        self.parameters=parameters
        self.env=env

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

        pylab.figure()
        pylab.plot(loss)
        pylab.xlabel('Epochs')
        pylab.ylabel('Loss')
        pylab.title('Average Loss')
        #pylab.savefig('Results/TrainingLoss.jpg')
        pylab.show()

        
        
    def train_error(self,satellite_policy,Exact_I):
        n_epochs=self.parameters['n_epochs']
        batch_size=self.parameters['batch_size']
        n_iter=self.parameters['n_iter']

        # Create the dataset (n_iter x hor x dimension of d_set) and the labels (n_iter x hor)
        data=self.env.extract_d_set()
        labels=self.env.extract_sources_influence()
        #build the network
        self.model = model(self.parameters)
        
        data=np.array(data)
        labels=np.array(labels)

        #Select train_test_split of the experience as train set and the rest as test set
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

        CE=[]
        N1=[]
        V=[]
        
        #Build the exact influence IALM where to evaluate the performances of the approximate-influence optimal policy
        Exact_IALM=IALM(self.parameters,Exact_I)
     
        for epoch in range(n_epochs):
            loss_epoch=0
            #Compute the approximate influence given the current state of the network
            #Approx_I is a len=hor list. dim(Approx_I[t])=dim(d_set_t) x dim(sources_influence)
            Approx_I=self.approximate_influence(satellite_policy)
            #Build the approximat influence IALM
            IALM_MR=IALM(self.parameters, Approx_I)
            #Compute the Approx_I optimal policy
            Pi=IALM_MR.value_iteration()[0]
            #Compute the value of the Approx_I optimal policy in the Exact_I IALM
            V.append(Exact_IALM.exact_evaluate_policy(Pi))
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
                loss_step, pred_lab =self.model.update(batch)
                loss_epoch+=loss_step
            loss_epoch=loss_epoch/total_batches
            loss[epoch]=loss_epoch
            
        return N1, CE, V, loss, Approx_I, Pi


    def approximate_influence(self, satellite_policy):
    	#Compute the approximate_influence for any D_set 
        I=[]
        #I.append(np.zeros((2,2)))
        #I[0]=self.env.Exact_IP[0]
        for t in range(self.parameters['hor']):
            I.append(np.zeros((2**(t+1),2)))
            for i in range(2**(t+1)):
            	#generate a d_set instatiation at time t according to the binary encoding order
                #Note: the input to the NN must have dim = hor
                D_set = "{0:b}".format(i)
                D_set = np.array([int(j) for j in D_set])
                while(len(D_set)<=t):
                    D_set=np.insert(D_set,0,0,axis=0)
                while (len(D_set)<self.parameters['hor']):
                    D_set=np.append(D_set,0)
                D_set=np.reshape(D_set,(1,self.parameters['hor'],1))
                #Get the probability distribution over sources of influence given the instatiation of the D_set and the current state of the network
                a_distr=self.model.predictions(D_set)[0]
                I[t][i]=a_distr[t]

        return I
        




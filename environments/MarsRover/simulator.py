# -*- coding: utf-8 -*-

import numpy as np
import time
from environments.MarsRover.utility import index_to_d_set, d_set_to_index,  generate_d_set

np.random.seed(42)

class MR_env (object):

    def __init__(self, parameters,verb=0):

        #General parameters
        self.parameters=parameters
        self.n_states=self.parameters['n_states']
        self.max_charge=self.parameters['max_charge']

        #Transition parameters
        self.p_fail=self.parameters['p_fail']
        self.consump=self.parameters['consumption']
        self.recharge=self.parameters['recharge']
        self.p_keep_plan=self.parameters['p_keep_plan']


        #Rewards parameters
        self.max_r=self.parameters['max_r']
        self.r_succ=self.parameters['r_succ']
        self.r_fail=self.parameters['r_fail']
        

        #Initial belief
        self.b_init=np.array(self.parameters['b_init'])

        #State space and action dimensions
        self.X0_dim=self.max_charge+1
        self.X1_dim=2
        self.X2_dim=self.n_states+1
        
        self.n_fact=3
        self.n_agents=2


        #Simulation parameters
        self.hor=self.parameters['hor']
        self.verb = verb
        self.trained=None
        self.n_iter=self.parameters['n_iter']
        
        self.verb=verb

    def get_rewards(self, prev_s, a, s):
        #Output: float reward of Mars Rover
        #Input: previous state tuple prev_s=(x0,x1,x2), tuple of actions a=(satellite action, MR action), tuple next state s=(x0',x1',x2'). 

        if s[2]==self.n_states:
            if prev_s[2]!=self.n_states:
        	    R=self.max_r
            else:
                R=0
        else: 
        	if s[2]==prev_s[2]:
        		if a[1]==0:
        			R=0
        		else:
        			R=self.r_fail
        	else:
        		R=self.r_succ
        	
        return R


    def initial_state(self):
        #Output: sample an initial state according to the initial distribution b_init
        x0=np.random.choice(np.arange(self.X0_dim),p=self.b_init[0])
        x1=np.random.choice(np.arange(self.X1_dim),p=self.b_init[1])
        x2=np.random.choice(np.arange(self.X2_dim),p=self.b_init[2])
        s=np.array([x0,x1,x2])
        return s
   

    def get_transitions_charge(self,prev_charge,a_sat):
        
        T=np.zeros(self.X0_dim)
        '''
        if a_sat==0:
            if prev_charge+self.recharge+1<=self.max_charge:
                T[prev_charge+self.recharge+1]=0.5
                T[prev_charge+self.recharge]=0.5
            elif prev_charge+self.recharge<=self.max_charge:
                T[prev_charge+self.recharge]=1
            else:
                T[self.max_charge]=1
        else:
            if prev_charge>=self.consump:
                T[prev_charge-self.consump]=1
            else:
                T[0]=1

        '''
        if a_sat==0:
            if prev_charge+self.recharge<=self.max_charge:
                T[prev_charge+self.recharge]=1
            else:
                T[self.max_charge]=1
        else:
            if prev_charge>=self.consump:
                T[prev_charge-self.consump]=1
            else:
                T[0]=1

        return T


    def get_transitions_plan(self,prev_plan,a_sat):

    	T=np.zeros(self.X1_dim)
    	if a_sat==0:
    		T[0]=1
    	else:
    		if prev_plan==1:
    			T[1]=self.p_keep_plan
    			T[0]=1-self.p_keep_plan
    		else:
    			T[1]=1

    	return T
    
    def get_transitions_pos(self,prev_pos, a_mr, plan):

        T=np.zeros(self.X2_dim)

        if a_mr==0:
            T[prev_pos]=1

        else:
            if prev_pos!=self.n_states:
                if plan==1:
                    T[prev_pos+1]=1
                else:
                    T[prev_pos]=self.p_fail
                    T[prev_pos+1]=1-self.p_fail
            else:
                T[prev_pos]=1
        
        return T

    def sample_new_state(self,prev_s,a):

        prev_charge, prev_plan, prev_pos = prev_s
        a_sat, a_mr = a
        
        T_charge=self.get_transitions_charge(prev_charge,a_sat)
        T_plan=self.get_transitions_plan(prev_plan,a_sat)

        charge = np.random.choice(self.X0_dim, p=T_charge)
        plan = np.random.choice(self.X1_dim, p=T_plan)

        T_pos=self.get_transitions_pos(prev_pos,a_mr,plan)
        pos= np.random.choice(self.X2_dim, p=T_pos)
        
        s=np.array([charge,plan,pos])

        return s

    def run_simulation(self, policies):
        #Input: list of policies of the two agents a0=satellite, a1=MR
        #Output: save arrays for the visited states S, action performed A, rewards received R and cumulative reward V
        # dim(S) = n_factors x n_iterations x hor
        # dim(A) = n_agents x n_iterations x hor
        # dim(R) = n_iterations x hor (only MR receives the reward
        # dim(V) = n_iteration
        T=time.time()

        target_reached=0

        self.S=np.zeros((self.n_fact, self.n_iter, self.hor))
        self.A=np.zeros((self.n_agents, self.n_iter, self.hor))
        self.R=np.zeros((self.n_iter, self.hor))
        self.V=np.zeros((self.n_iter))

        for i in range(self.n_iter):

            s=self.initial_state()
            a=np.zeros(self.n_agents)
            a_distr= [None] * self.n_agents

            for t in range(self.hor):
                for k , pi in enumerate(policies):
                    a_distr[k] = pi(s)
                    a[k]=np.random.choice(np.arange(len(a_distr[k])), p=a_distr[k])

                self.S[:, i, t] = s
                self.A[:, i, t] = a   
                prev_s=np.copy(s)
                
                s=self.sample_new_state(prev_s, a)
                
                r=self.get_rewards(prev_s,a,s)

                if s[2]==self.n_states and prev_s[2]!=self.n_states:
                    target_reached=+1
                
                self.R[i, t]=r
                if self.verb:
                    print(t,'Previous state',prev_s)
                    print (t, 'State   ', s)
                    print (t, 'Actions ', a)
                    print(t, 'Rewards', r)
                    print ('\n')
                    input()
                self.V[i]+=r
        T=time.time()-T
        self.trained=True
        

        print ('\nRunning time ', T)
        print ('Times the target has been reached before the horizon ', target_reached)
    
    
    def exact_influence(self, satellite_policy):
        #satellite_policy (input: factored state. output: a probability distribution over the action of the satellite)
        #b_init (input: factor index, output: initial probability distribution of the selected factor)
        #Output: I  a list with lenght hor. For each t<=hor I[t] has dim(D_set) x dim(sources_influence) components 
        T=time.time()
        I=[]
        b=[]
        for t in range(self.hor):
            I.append(np.zeros((2**(t+1),2)))
            b.append(np.zeros((2**(t+1),self.X0_dim)))

            if t==0:

                for i in range(2):
                    if self.b_init[1][i]>0:
                       b[0][i]=self.b_init[0]
                       I[0][i]=np.dot(self.b_init[0],np.array([satellite_policy([x,0,0]) for x in range(self.X0_dim)]))
                    else:
                        b[0][i]=np.zeros(self.X0_dim)
                        I[0][i]=np.zeros(2)
            else:

                D_set =generate_d_set(t)
              
                for i in range(2**(t+1)):
                    d_set=D_set[i]
                    prev_d_set=d_set[:-1]
                    prev_d_set_index=d_set_to_index(t-1,prev_d_set)              
                    prev_b=b[t-1][prev_d_set_index]
                    
                    if prev_b.any():
                    #if the previous d_set is not compatible then leave the distribution as zeros
                        b_app=np.zeros(self.X0_dim)
                        x1=d_set[-1]
                        prev_x1=d_set[-2]

                        for x0 in range(self.X0_dim):
                            c=0
                            for a in range(2):
                                p1=self.get_transitions_plan(prev_x1,a)[x1]
                                p0=np.array([self.get_transitions_charge(prev_x0,a)[x0] for prev_x0 in range(self.X0_dim)])
                                pol=np.array([satellite_policy([prev_x0,0,0])[a] for prev_x0 in range(self.X0_dim)])
                                c+=p1*np.sum(p0*pol*prev_b)
                            b_app[x0]=c

                        if b_app.any():
                            b_app = b_app/np.sum(b_app)      

                        b[t][i]=b_app
                        
                        I[t][i,0]=np.sum(b_app*np.array([satellite_policy([x0,0,0])[0] for x0 in range(self.X0_dim)]))
                        I[t][i,1]=np.sum(b_app*np.array([satellite_policy([x0,0,0])[1] for x0 in range(self.X0_dim)]))

        
        print('Time for exact influence inference', time.time()-T)
        self.Exact_IP=I





    def extract_d_set (self):
    #Output: generated d_sets during the simulation dim(D_set) = n_iter x hor 
        D_set=[]
        for i in range(self.n_iter):
            D_set.append([])
            for j in range(self.hor):
                D_set[i].append(np.array([self.S[1,i,j]]))
        return D_set


    def extract_sources_influence(self):
        #Output: generated sources of influence during simulation dim(s_inf) =  n_iter
        s_inf=[]
        for i in range(self.n_iter):
            s_inf.append(self.A[0,i,:])
        return s_inf
    








   

    


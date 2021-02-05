# -*- coding: utf-8 -*-

import numpy as np
import time
from environments.TrafficGrid.Global_simulator import Traffic_env
import pdb
from environments.TrafficGrid.utility import index_to_d_set, d_set_to_index, generate_d_set
np.random.seed(42)

class IALM (Traffic_env):

    def __init__(self, parameters,  Influence):

        Traffic_env.__init__(self, parameters)
        self.Influence=Influence
        self.D=self.generate_d_set()


    def IALM_get_rewards(self, s):
        #Output: (float) reward of the local agent
        if len(s)==2:
            d_set, local_s = s
        else:
            local_s=s
        R=-np.sum(local_s[np.array([1,2])])
        	
        return R

    def IALM_get_transitions(self, prev_s, a, local_s, t):

        prev_d_set, prev_local_s = prev_s
        
        prev_south,prev_east,prev_north,prev_west=prev_local_s
        south,east,north,west=local_s
        
        south_, east_, north_, west_=self.compatibility_conditions(prev_s, a)
        
        I=self.Influence[self.parameters['hor']-t-1][self.d_set_to_index(t,prev_d_set)]
        

        if east_ is None:
            P_east=I[0][int(east)]
        else:
            P_east=1

        if north_ is None:
            P_north=I[1][int(north)]
        else:
            P_north=1

        return P_east*P_north


    def value_iteration(self):
    	#Output: policy, value
    	#policy is list hor x dim augmented state space for each t
    	# t stands for the actions left
        T=time.time()
        V=[]
        Pi=[]

        for t in range(self.hor):
            #still t+1 actions to perform
            print('value iteration ', t)
            V.append([])
            Pi.append([])
            S, dim_S=self.state_space(t)
            if t==0:
                for prev_s in S:
                    Qval=[np.sum([self.IALM_get_transitions(prev_s, a,local_s,t)*self.IALM_get_rewards(local_s) for local_s in self.local_space(prev_s,a)]) for a in range(2)]
                    V[t].append(np.max(Qval))
                    Pi[t].append(np.argmax(Qval))
            else:
                prev_V=V[t-1]
                #S_next, dim_S_next=self.state_space(t-1)
                '''
                if t==self.parameters['hor']-1:
                    print('state space', S)
                    print('state ', S[13])
                    input()
                    prev_s=S[13]
                    a=0
                    feasible_next_local=self.local_space(prev_s,a)
                    print('feasible next', feasible_next_local)
                    input()
                    for local_s in feasible_next_local:
                            print('next local state ', local_s)
                            next_index= self.local_to_next_state_index(prev_s,local_s)
                            print('local state index ', next_index)
                            input()
                            Qval[a]+=self.IALM_get_transitions(prev_s,a,local_s,t)*(self.IALM_get_rewards(local_s)+prev_V[next_index])
                            print(Qval[a])
                            input()
                    a=1
                    feasible_next_local=self.local_space(prev_s,a)
                    print('feasible next', feasible_next_local)
                    input()
                    for local_s in feasible_next_local:
                            print('next local state ', local_s)
                            next_index= self.local_to_next_state_index(prev_s,local_s)
                            print('local state index ', next_index)
                            input()
                            Qval[a]+=self.IALM_get_transitions(prev_s,a,local_s,t)*(self.IALM_get_rewards(local_s)+prev_V[next_index])
                            print(Qval[a])
                            input()
                            '''
                S_next,dim_S_next=self.state_space(t-1)
                for prev_s in S:
                    Qval=np.zeros(2)
                    for a in range(2):
                        feasible_next_local=self.local_space(prev_s,a)
                        for local_s in feasible_next_local:
                            #print('next local state ', local_s)
                            next_index= self.local_to_next_index(prev_s,local_s,S_next)
                            #print('local state index ', next_index)
                            Qval[a]+=self.IALM_get_transitions(prev_s,a,local_s,t)*(self.IALM_get_rewards(local_s)+prev_V[next_index])
                    V[t].append(np.max(Qval))
                    Pi[t].append(np.argmax(Qval))
                #S_next, dim_S_next=self.state_space(t-1)
                #for prev_s in S:
                #    Qval=[np.sum([self.IALM_get_transitions(prev_s,a,s,t)*(self.IALM_get_rewards(s)+prev_V[j]) for j,s in enumerate(S_next)]) for a in range(2)]
                #    V[t].append(np.max(Qval))
                #    Pi[t].append(np.argmax(Qval))                  


        Pi.reverse()
        print('Time for value iteration',time.time()-T)
        return Pi

    def local_to_next_state_index(self,prev_s, local_s):
        d_set=prev_s[0]
        d_set=np.concatenate([d_set,[local_s[np.array([0,3])]]])
        next_state=[d_set,local_s]
        index=self.state_to_index(next_state, self.parameters['hor']-len(d_set))
        return index

    def local_to_next_index(self,prev_s, local_s, S):
        d_set=prev_s[0]
        d_set=np.concatenate([d_set,[local_s[np.array([0,3])]]])
        next_state=[d_set,local_s]
        for i in range(len(S)):
            if np.array_equal(S[i][0],next_state[0]) and np.array_equal(S[i][1],next_state[1]):
                break
        return i

    def update_local_state(self,s, d_set):
        local_s=self.get_local_state(s)[0]
        if len(d_set)!=0:
            d_set=np.concatenate([d_set,[local_s[np.array([0,3])]]])
        else:
            d_set=np.array([local_s[np.array([0,3])]])
        local_s=[d_set,local_s]
        #print('IALM state', local_s)
        #print('d set', d_set)
        #input()
        return local_s, d_set

    def evaluate_IALM_policy(self, n, Pi_IALM, policies,verb=0):

        T=time.time()
        V=np.zeros(n)
       
        for i in range(n):
            
            s=self.initial_state()
            a=np.zeros(self.n_agents)
            a_distr= [None] * (self.n_agents-1)
            dset=[]
            
            for t in range(self.hor):
                local_s_0, dset=self.update_local_state(s,dset)
                local_s=self.get_local_state(s)
                action_index=self.state_to_index(local_s_0,self.parameters['hor']-t-1)
                a[0]=int(Pi_IALM[t][action_index])
 
                for k , pi in enumerate(policies):
                    a_distr[k] = pi(local_s[k+1])
                    a[k+1]=np.random.choice(np.arange(len(a_distr[k])), p=a_distr[k])
 
                prev_s=np.copy(s)
                
                s=self.sample_s(prev_s, a)
                r=self.get_rewards(s)
                
                if verb:
                    self.plot_state(prev_s,a)

                V[i]+=r
            #pylab.show(block=True)

        T=time.time()-T
        self.trained=True

        return np.mean(V)
    

    def local_space(self, prev_s, a):
        local_S=[]
        
        south, east, north, west=self.compatibility_conditions(prev_s,a)

        if east is None and north is None:
            for east in range(2):
                for north in range(2):
                    local_S.append([south,east,north,west])
        elif east is None:
            for east in range(2):
                local_S.append([south,east,north,west])
        else:
            for north in range(2):
                local_S.append([south,east,north,west])

        return np.array(local_S)

    def compatibility_conditions(self, prev_s, a):
        prev_d_set, prev_local_s = prev_s
        prev_south,prev_east,prev_north,prev_west=prev_local_s
        
        south, east, north, west=[None]*4
        if a==0:
            south=0
            if prev_east==1:
                west=1
            else:
                west=0
            #if prev_north==1:
            #    north=1
        else:
            west=0
            if prev_north==1:
                south=1
            else:
                south=0
            #if prev_east==1:
            #    east=1

        return south, east, north, west



    def state_to_index(self, s, t):
        #t+1 actions to perform
    	# t is the number of steps/actions left
        #output index of the state according to d set binary enumeration at time t
        S, dim_S=self.state_space(t)
        for i in range(dim_S):
            if np.array_equal(S[i][0],s[0]) and np.array_equal(S[i][1],s[1]):
                break
        return i


    def index_to_d_set(self, t, index):
        D_t=self.d_set(t)
        return D_t[index]

    def d_set_to_index(self,t, dset):
        D_t=self.d_set(t)
        for i in range(len(D_t)):
            if np.array_equal(D_t[i],dset):
                break
        return i

    def d_set(self,t):
        indexes=np.arange(0,4**self.parameters['hor'], step=4**t)
        rev_index=self.parameters['hor']-t
        D_t=self.D[indexes,:rev_index]
        return D_t

    def state_space(self, t):
        #State space when t+1 actions to perform
        S=[]
        D_t=self.d_set(t)
        for d in D_t:
            local_s=np.zeros(4)
            local_s[np.array([0,3])]=d[-1]
            for i in range(2):
                for j in range(2):
                    local_s[np.array([1,2])]=i,j
                    app=[d,np.copy(local_s)]
                    S.append(app)
        return S, len(S)

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




   

    


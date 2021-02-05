# -*- coding: utf-8 -*-

import numpy as np
import time
from environments.FireFighters.utility import index_to_inst

class ffg_env (object):

    def __init__(self, parameters):

        #Number of agents - number of variable factors
        self.parameters=parameters
        self.n_agents= self.parameters['n_agents']
        self.n_fact = self.parameters['n_fact']
        self.f_lev=self.parameters['f_lev']
        #Transition parameters
        self.t1=self.parameters['t1']
        self.t2=self.parameters['t2']
        self.t3=self.parameters['t3']

        #Initial distribution
        self.b_init=np.array(self.parameters['b_init'])
        
        #State space dimensions
        self.X_dim=self.f_lev

        #Simulation parameters
        self.hor=self.parameters['hor']
        self.verb = 0
        self.trained=None
        self.n_iter=self.parameters['n_iter']
        
    def initial_state(self):
        #Output: sample an initial state according to the initial distribution b_init
        s0=[]
        for x in range(self.n_fact):
            s0.append(np.random.choice(np.arange(self.X_dim),p=self.b_init[x]))
        s0=np.array(s0)
        return s0

    
    def get_rewards(self, prev_s, a, s):
        #Output: array rewards len = n_agents
        #Input: previous state tuple prev_s=(x0,x1,x2,..), tuple of actions a=(a_1, a_2, ...), tuple next state s=(x0',x1',x2', ...)
        R = []
        for fact in range(self.n_agents):
            R.append(-prev_s[fact]-prev_s[fact+1])
        return np.array(R)

    def get_transitions(self, prev_s, a , fact):

        #Input: the factor whose CPT must be computed, an array of actions a at time t, an array of states factors at time t
        #Output: the CPT (a distribution over the firelevels) of the specified factor given the current input actions and state a, s.
        if not prev_s.any():
            T=np.zeros(self.X_dim)
            T[0]=1
            return T

        if fact==0:
            T=self.get_factor_transitions(prev_s[fact:fact+2],a[fact],fact) 

        elif fact == self.n_fact - 1:
            T=self.get_factor_transitions(prev_s[fact-1:fact+1],a[fact-1],fact)

        else:
            T=self.get_factor_transitions(prev_s[fact-1:fact+2],a[fact-1:fact+1],fact) 

        return T


    def get_factor_transitions(self, prev_fact, a, fact):
        T=np.zeros(self.X_dim)

        if not prev_fact.any():
        #the house and its neighbouring houses are not burning
            T[0]=1
            return T

        if fact==0:
            prev_x = np.nan
            x , next_x = prev_fact
            prev_a = np.nan
            next_a = a

        elif fact == self.n_fact-1:
            prev_x , x = prev_fact
            next_x = np.nan
            prev_a= a
            next_a = np.nan

        else:
            prev_x, x , next_x =prev_fact
            prev_a, next_a = a 

        if (prev_a == 0 and next_a == 1) or (np.isnan(prev_a) and  next_a==1) or (prev_a==0  and  np.isnan(next_a)):
        #No agents fighting fire
            if x == self.f_lev - 1:
            #maximum level of fire
                T[-1] = 1
            elif prev_x>0 or next_x>0:
                #Neighboring houses are burning
                T[x]=1-self.t2
                T[x+1]=self.t2
            else:
                #Neighboring house are not burning but the house is burning
                T[x]=1-self.t3
                T[x+1]=self.t3

        elif prev_a==1 and next_a==0:
        # Both agents fighting fire
            T[0]=1

        else:
        # only one agent fighting fire 
            if x==0:
                #The house is not burning
                T[0]=1
            else:
                #The house is burning
                if prev_x>0 or next_x>0:
                    #At least one of its neigbors is burning
                    T[x-1]= 1-self.t1
                    T[x]=self.t1
                else:
                    # None of its neigbors is burning
                    T[x-1]=1
        return T

    def run_simulation(self, policies):
        #Input: list of policies of the FFs 
        #Output: save arrays for the visited states S, action performed A, rewards received R and cumulative reward V
        # dim(S) = n_factors x n_iterations x hor
        # dim(A) = n_agents x n_iterations x hor
        # dim(R) = n_agents x n_iterations x hor 
        # dim(V) = n_agents x n_iteration
        T=time.time()

        fire_off=0

        self.S=np.zeros((self.n_fact, self.n_iter, self.hor))
        self.A=np.zeros((self.n_agents, self.n_iter, self.hor))
        self.R=np.zeros((self.n_agents, self.n_iter, self.hor))
        self.V=np.zeros((self.n_agents, self.n_iter))

        for i in range(self.n_iter):

            s=self.initial_state()
            a=np.zeros(self.n_agents)
            a_distr= [None] * self.n_agents

            for t in range(self.hor):
                for k , pi in enumerate(policies):
                    o = s[k:k+2]
                    a_distr[k] = pi(o)
                    a[k]=np.random.choice(np.arange(len(a_distr[k])), p=a_distr[k])

                self.S[:, i, t] = s
                self.A[:, i, t] = a   
                prev_s=np.copy(s)

                p=np.array([self.get_transitions(prev_s, a, k ) for k in range(self.n_fact)])

                for k in range(self.n_fact):
                    s[k] = np.random.choice(np.arange(len(p[k])), p=p[k])
                
                r=self.get_rewards(prev_s,a,s)

                if not s.any():
                    fire_off=+1
                
                self.R[:, i, t]=r
                if self.verb:
                    print(t,'Previous state',prev_s)
                    print (t, 'State   ', s)
                    print (t, 'Actions ', a)
                    print(t, 'Rewards', r)
                    print ('\n')
                self.V[:, i]+=r
            if self.verb:
                print ('Value', self.V[:,i])
        T=time.time()-T
        self.trained=True

        print ('\nRunning time ', T)
        print ('Times the fire has been extinguished before the horizon ', fire_off)


    def extract_d_set (self):
    #Output: generated d_sets during the simulation dim(D_set) =  n_iter x hor x dim(d_set) 
        D_set=[]
        for i in range(self.n_iter):
            D_set.append([])
            for j in range(1,self.hor):
                D_set[i].append(np.array([self.S[1,i,j],self.A[1,i,j-1],self.S[2,i,j-1]]))
        return D_set


    def extract_sources_influence(self):
        #Output: generated sources of influence during simulation dim(s_inf) =  n_iter
        s_inf=[]
        for i in range(self.n_iter):
            s_inf.append(self.S[0,i,1:])
        return s_inf


    def infer_influence(self,slot=None,verb=0):
        #Empirical computation of the influences based on the simulations
        #If slot is not None, compute empirical conditional distributions of the source of influence
        # given a subset of the d-separating including only variables up to stage t-slot in the past.

        T=time.time()

        assert self.trained==True, ('Run the simulations first')

        self.inf_source=self.n_agents-self.n_loc_agents-1
        self.slot=slot

        if self.slot is None:
            self.influence=self._compute_inf(slot=None, verb=verb)
        else:
            self.influence = self._compute_inf(slot=None, verb=verb)
            self.influence_app = self._compute_inf(slot=self.slot, verb=verb)

        print ('Running time to compute influences ', time.time()-T)


    def _compute_inf(self,slot, verb):

        if verb:
            print ('\n')
            print ('Inference on influence')

        Influence=[]

        for t in range(self.hor):

            if t == 0:
                H = self.X[self.inf_source, :, t]
                Influence0 = [float((H == np.array([i])).sum()) / self.n_iter for i in range(self.f_lev)]
                if verb:
                    print ('Influence for t=0')
                    print ('Freq of souc_inf X_0 as 0, 1, ... ,', self.f_lev-1)
                    print (Influence0)
                Influence.append(np.array(Influence0))

            else:
                if verb:
                    print ('\n\n Influence for t=', t)

                if slot is None:
                    D = self._d_set(t)
                else:
                    D= self._d_set_app(t, slot)

                D.insert(0, self.X[self.inf_source, :, t])
                H = np.transpose(D)

                vars=H.shape[1]-1
                instan = self.f_lev ** (vars)

                comb = np.zeros((instan, vars))
                Influence.append(np.zeros((instan, self.f_lev)))

                for i in range(instan):

                    comb[i] = index_to_inst(vars, base=self.f_lev)(i)
                    space = (H[:, 1:] == comb[i]).all(axis=1)

                    if (space.sum()) != 0:
                        Influence[t][i] = [
                            float((H[:, 0][space] == np.array([k])).sum()) / (space).sum() for k
                            in range(self.f_lev)]

                        if verb:
                            print ('D set as', comb[i])
                            print ('state space size = ', (space).sum())
                            c = np.round(1.96 * np.sqrt((Influence[t][i, -1] * (1 - Influence[t][i, -1])) / space.sum()), 4)
                            I = np.round(Influence[t][i, -1], 4)
                            print ('Conf Int [' + str(np.round(I - c, 4)) + ',' + str(np.round(I + c, 4)) + ']')
                            print ('Freq of souc_inf X as 0, 1, ... ,',self.f_lev-1)
                            print (Influence[t][i])

                            print ('\n')

        return Influence


    def exact_infl(self, pi0):

        T = time.time()
        Influence = []
        for t in range(self.hor):
            if t == 0:
                Influence.append(np.ones(self.f_lev)/self.f_lev)
            elif t==1:
                M = self.d_matrix(t)
                b=np.array(Influence[0])
                Influence.append(np.zeros((M.shape[0], self.f_lev)))
                for i in range(M.shape[0]):
                    x1,a1,x2,x1_= M[i]
                    app=np.sum([self.get_transitions(1,np.array([pi0(x0,x1),a1]),np.array([x0,x1,x2]))[int(x1_)]*self.get_transitions(0,np.array([pi0(x0,x1),a1]),np.array([x0,x1,x2]))*b[x0] for x0 in range(self.f_lev)],axis=0)
                    if np.sum(app)!=0:
                        Influence[t][i]=app/np.sum(app)
            else:
                M = self.d_matrix(t)
                _M=self.d_matrix(t-1)
                Influence.append(np.zeros((M.shape[0], self.f_lev)))
                for i in range(M.shape[0]):
                    D_= M[i]
                    D = D_[:-3]
                    x1,a1,x2,x1_ = D_[-4:]
                    D_index =int(np.where(np.all(_M==D,axis=1))[0])
                    b = Influence[t-1][D_index]
                    app=np.sum([self.get_transitions(1,np.array([pi0(x0,x1),a1]),np.array([x0,x1,x2]))[int(x1_)]*self.get_transitions(0,np.array([pi0(x0,x1),a1]),np.array([x0,x1,x2]))*b[x0] for x0 in range(self.f_lev)],axis=0)
                    if np.sum(app) != 0:
                        Influence[t][i]=app/np.sum(app)
        return Influence
        #self.Ex_influence=Influence
        print('Running time to compute exact influences ', time.time() - T)


    def _d_set(self,t):
        #Input: time
        #Output: Sample of d-separating set at time t
        index=self.inf_source+1

        if t==1:
            D=[self.X[index,:,0], self.A[index,:,0], self.X[index+1,:,0], self.X[index,:,1]]
        else:
            D=self._d_set(t-1)
            D.append(self.A[index,:,t-1])
            D.append(self.X[index+1,:,t-1])
            D.append(self.X[index,:,t])

        return D

    def _d_set_app(self,t, slot):
        #Input: time
        #Output: Sample of subset of d-separating set at time t containing only variables from stages t-slot up to t.
        indexes=1+(slot-1)*3
        D=self._d_set(t)
        D_app=D[-indexes:]
        return D_app


    def d_matrix(self,t):
        #To be checked
        #Input: time
        #Output: matrix M whose for each row provides an instatiation of the d-separating set at time t.
        vars=3*t+1
        act=self.n_loc_agents*t
        states=vars-act
        instan=2**(act)*(self.f_lev)**(states)
        tot_instan=(self.f_lev)**(vars)
        m=[]

        for i in range(tot_instan):
            app =index_to_inst(vars,self.f_lev)(i)
            check = np.array([app[j] for j in range(1,len(app),3)])
            if (check > 1).sum()==0:
                m.append(app)
        return np.array(m)


    def d_set (self):
        agent_indexes  = - self.n_loc_agents
        factor_indexes = - self.n_loc_agents -1
        D_set=[]
        for i in range(self.n_iter):
            D_set.append([])
            for j in range(self.hor):
                D_set[i].append(np.concatenate((self.X[factor_indexes:,i,j],self.A[agent_indexes:,i,j])))
        return D_set

    def sources_influence(self):
        #TODO: make shape(s_inf) = (n_iter, hor, 1)
        #now shape(s_inf) = (n_iter, hor)
        factor_indexes = - self.n_loc_agents -2
        s_inf=[]
        for i in range(self.n_iter):
            s_inf.append(self.X[factor_indexes,i,:])
        return s_inf



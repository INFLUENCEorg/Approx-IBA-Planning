
import numpy as np

class agent_satellite(object):

    def __init__(self, parameters):

        self.min_level=parameters['min_level']
        self.policy_fail=parameters['policy_fail']
        self.consumption=parameters['consumption']

        policy=parameters['pi_satellite']

        if policy=='rand_pi':
            self.policy=self.rand_pi

        if policy=='min_level':
            self.policy=self.min_level_pi
        
        if policy=='rand_min_level':
            self.policy=self.rand_min_level_pi

    def rand_pi(self, s):
        if s[0]<=self.consumption:
            action_distr=np.array([1,0])
        else:
            action_distr=np.array([0.5,0.5])
        return action_distr

    def min_level_pi(self, s):
    
        if s[0]>=self.min_level:
            action_distr= np.array([0,1])
        else:
            action_distr=np.array([1,0])

        return action_distr

    def rand_min_level_pi(self,s):

        if s[0]>=self.min_level:
            action_distr=np.array([self.policy_fail,1-self.policy_fail])
        else:
            action_distr=np.array([1,0])

        return action_distr
        

class agent_rover(object):

    def __init__(self, parameters):

        policy=parameters['pi_rover']

        if policy=='rand':
            self.policy=self.rand_pi

        if policy=='opt_det':
        	self.policy=self.opt_det

    def opt_det(self,s):
    	plan=s[1]
    	if plan==0:
    	    action_distr=np.array([0.,1.])
    	else:
    		action_distr=np.array([1.,0.])
    	return action_distr

    def rand_pi(self, s):
        action_distr=np.array([0.5,0.5])
        return action_distr



'''
Created on Aug 20, 2020

@author: Atrisha
'''

class AgentTote:
    
    def __init__(self, agent_id, time_ts, relev_agents, emp_action):
        self.agent_id = agent_id
        self.time_ts = time_ts
        self.relev_agents = {r:None for r in relev_agents}
        self.emp_action = emp_action
        
class AgentInfo:
    
    def __init__(self, agent_id, time_ts, emp_action):
        self.agent_id = agent_id
        self.time_ts = time_ts
        self.emp_action = emp_action
    
    def set_s1_traj(self, s1_traj):
        self.s1_trajectory = s1_traj
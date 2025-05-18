import numpy as np
from map import Map
import json
import os
import random

class Instance:
    def __init__(self,map:Map,num_agents:int,num_tasks:int,seed:int):
        self.map=map
        self.num_agents=num_agents
        self.num_tasks=num_tasks
        self.starts=[]
        self.tasks=[]
        self.seed=seed
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.random_gen()
    
    def random_gen(self):
        rows,cols=np.where(self.map.graph==0)
        
        num=len(rows)
        
        # samples start locations
        idxs=np.random.choice(a=num,size=self.num_agents,replace=False)
        self.starts=[(rows[idx],cols[idx]) for idx in idxs]
        
        # sample tasks
        idxs=np.random.randint(low=0,high=num,size=self.num_tasks)
        self.tasks=[(rows[idx],cols[idx]) for idx in idxs]
        
    def save(self,suffix):
        folder,subfolder=os.path.split(os.path.split(self.map.fp)[0])
        assert subfolder=="maps"
        map_fn=self.map.fn
        map_name=os.path.splitext(map_fn)[0]
        name="{}_{}_{}_{}".format(map_name,self.num_agents,self.num_tasks,suffix)
        agent_fn="{}.agents".format(name)
        task_fn="{}.tasks".format(name)
        instance_fn="{}.json".format(name)
        
        os.makedirs(os.path.join(folder,"agents"),exist_ok=True)
        os.makedirs(os.path.join(folder,"tasks"),exist_ok=True)
        os.makedirs(os.path.join(folder,"instances"),exist_ok=True)
        
        agent_fp=os.path.join(folder,"agents",agent_fn)
        task_fp=os.path.join(folder,"tasks",task_fn)
        instance_fp=os.path.join(folder,"instances",instance_fn)
        
        with open(agent_fp,"w") as f:
            f.write(str(self.num_agents)+"\n")
            for i in range(len(self.starts)):
                loc=self.starts[i]
                loc_idx=loc[0]*self.map.width+loc[1]
                f.write(str(loc_idx)+"\n")
                
        with open(task_fp,"w") as f:
            f.write(str(self.num_tasks)+"\n")
            for i in range(len(self.tasks)):
                loc=self.tasks[i]
                loc_idx=loc[0]*self.map.width+loc[1]
                f.write(str(loc_idx)+"\n")
                
        cfg={
            "mapFile": "../maps/"+map_fn,
            "agentFile": "../agents/"+agent_fn,
            "teamSize": self.num_agents,
            "taskFile": "../tasks/"+task_fn,
            "numTasksReveal": 1,
            "taskAssignmentStrategy": "roundrobin"
        }
        
        with open(instance_fp,"w") as f:
            json.dump(cfg,f,indent=4)
            
if __name__=="__main__":
    fp="../our_problems/game.domain/maps/brc202d.map"
    map=Map(fp)
    map.print_graph(map.graph)
    instance=Instance(map,num_agents=400,num_tasks=11000,seed=0)
    instance.save("test")
    
        
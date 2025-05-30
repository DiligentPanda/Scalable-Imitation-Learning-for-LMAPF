class EpisodeLog:
    def __init__(self, num_robots, enabled=True):
        self.log={
            "actionModel": "MAPF",
            # "AllValid": None,
            "teamSize": num_robots,
            "start": [],
            "numTaskFinished": -1,
            # "sumOfCost": None,
            "makespan": -1,
            "actualPaths": [[] for i in range(num_robots)],
            # "plannerTimes": None,
            # "errors": None,
            "events": [[] for i in range(num_robots)],
            "tasks": [],
        }
        
        self.tasks=[]
        self.agent_tasks=[None]*num_robots
        
        # NOTE this need to follow the same order as movements in the env
        self.actions=["R","D","L","U","W"]
        
        self.enabled=enabled
        
    def add_starts(self, locs):
        if not self.enabled:
            return
        self.log["start"]=[loc+["E"] for loc in locs.cpu().numpy().tolist()]
        
    def add_actions(self, actions):
        if not self.enabled:
            return
        actions=actions.cpu().numpy().tolist()
        for agent_idx in range(len(actions)):
            self.log["actualPaths"][agent_idx].append(self.actions[actions[agent_idx]])
            
    def add_completed_tasks(self, step, reached):
        if not self.enabled:
            return
        reached=reached.cpu().numpy().tolist()
        for agent_idx in range(len(reached)):
            if reached[agent_idx]:
                self.log["events"][agent_idx].append(
                    [step, self.agent_tasks[agent_idx][0], "finished"]
                )
    
    def add_new_tasks(self, step, reached, target_locs):
        if not self.enabled:
            return
        if reached is not None:
            reached=reached.cpu().numpy().tolist()
        target_locs=target_locs.cpu().numpy().tolist() 
        for agent_idx in range(len(target_locs)):
            if reached is None or reached[agent_idx]:
                task=[len(self.log["tasks"]), *target_locs[agent_idx]]
                self.agent_tasks[agent_idx]=task
                self.log["tasks"].append(task)
                self.log["events"][agent_idx].append(
                    [step, self.agent_tasks[agent_idx][0], "assigned"]
                )
    
    def summarize(self):
        numTaskFinsihed=0
        for agent_events in self.log["events"]:
            for event in agent_events:
                if event[-1]=="finished":
                    numTaskFinsihed+=1
        
        self.log["numTaskFinished"]=numTaskFinsihed
        self.log["makespan"]=len(self.log["actualPaths"][0])
    
    def dump(self, path):
        self.summarize()
        
        for i in range(len(self.log["actualPaths"])):
            self.log["actualPaths"][i]=",".join(self.log["actualPaths"][i])

        import json
        with open(path,"w") as f:
            json.dump(self.log,f)
            
    def __str__(self):
        self.summarize()
        return str(self.log)
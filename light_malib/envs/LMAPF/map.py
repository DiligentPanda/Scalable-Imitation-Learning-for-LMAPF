import numpy as np
import queue
import os
import yaml

class Map:
    def __init__(self,fp=None, agent_bins=None):
        self.fp=fp
        self.name=None
        self.height=None
        self.width=None
        self.graph=None
        self.agent_bins=[]
        self.s_locations=[]
        self.e_locations=[]
        
        if fp is not None:
            self.load(self.fp)
        
        if agent_bins is not None:
            self.agent_bins=agent_bins
            
    def set_agent_bins(self, agent_bins:list):
        self.agent_bins=agent_bins
            
    def load_learn_to_follow_map(self, name:str, map_str:str):
        self.name=name
        lines=map_str.split("\n")
        self.height=len(lines)
        self.width=len(lines[0])
        self.graph=np.zeros((self.height,self.width),dtype=int)
        for row in range(self.height):
            line = lines[row]
            assert len(line)==self.width
            for col, loc in enumerate(line):
                # obstacle
                if loc=="#":
                    self.graph[row,col]=1

        self.only_keep_the_main_connected_component()
    
    def load(self,fp:str):
        self.fp=fp
        self.name=os.path.splitext(os.path.split(self.fp)[-1])[0]
        with open(fp,"r") as f:
            # skip the type line
            f.readline()
            self.height=int(f.readline().split()[-1])
            self.width=int(f.readline().split()[-1])
            self.graph=np.zeros((self.height,self.width),dtype=int)
            # skip the map line
            f.readline()
            for row in range(self.height):
                line=f.readline().strip()
                assert len(line)==self.width
                for col,loc in enumerate(line):
                    # obstacle
                    if loc=="@" or loc=="T":
                        self.graph[row,col]=1
                    if loc=="S":
                        self.s_locations.append((row,col))
                    if loc=="E":
                        self.e_locations.append((row,col))
        # self.print_graph(self.graph)
        
        self.s_locations=np.array(self.s_locations)
        self.e_locations=np.array(self.e_locations)
        
        self.only_keep_the_main_connected_component()
        
        
    def generate_task_files(self, num, fp, uniform=False, seed=None):
        if seed is None:
            import time
            seed=int(time.time())
        np.random.seed(seed)
        if len(self.s_locations)+len(self.e_locations)>0 and not uniform:
            s_locations=self.s_locations[:,0]*self.width+self.s_locations[:,1]
            e_locations=self.e_locations[:,0]*self.width+self.e_locations[:,1]
            prob=np.random.uniform(0,1,size=num)
            mask=(prob<0.5).astype(int)
            s_tasks = np.random.choice(s_locations,size=num,replace=True)
            e_tasks = np.random.choice(e_locations,size=num,replace=True)
            tasks = mask*s_tasks+e_tasks*(1-mask)
        else:
            task_locations=np.stack(np.nonzero(self.graph==0),axis=1)
            task_locations=task_locations[:,0]*self.width+task_locations[:,1]
            tasks = np.random.choice(task_locations,size=num,replace=True)
        
        with open(fp,'w') as f:
            f.write(str(num)+"\n")
            for i in range(len(tasks)):
                f.write(str(tasks[i])+"\n")  
                
    def generate_agent_files(self, num, fp, seed=None):
        if seed is None:
            import time
            seed=int(time.time())
        np.random.seed(seed)
        agent_locations=np.stack(np.nonzero(self.graph==0),axis=1)
            
        agent_locations=agent_locations[:,0]*self.width+agent_locations[:,1]
        agents = np.random.choice(agent_locations,size=num,replace=False)
        
        with open(fp,'w') as f:
            f.write(str(num)+"\n")
            for i in range(len(agents)):
                f.write(str(agents[i])+"\n")  
    
    def only_keep_the_main_connected_component(self):
        component_idx_map=np.zeros((self.height,self.width),dtype=int)
        
        max_component_size=0
        max_component_idx=0
        
        component_idx=0
        for row in range(self.height):
            for col in range(self.width):
                if self.graph[row,col]==0 and component_idx_map[row,col]==0:
                    component_idx+=1
                    size=self.bfs_count((row,col),component_idx,component_idx_map)
                    # print(component_idx,size)
                    if size>max_component_size:
                        max_component_size=size
                        max_component_idx=component_idx

        self.graph[max_component_idx!=component_idx_map]=1
        
                
    def get_loc_neighbors(self,loc:tuple):
        neighbors=[]
        
        # up
        if loc[0]>0:
            neighbor=(loc[0]-1,loc[1])
            neighbors.append(neighbor)
        
        # down
        if loc[0]<self.height-1:
            neighbor=(loc[0]+1,loc[1])
            neighbors.append(neighbor)
            
        # left
        if loc[1]>0:
            neighbor=(loc[0],loc[1]-1)
            neighbors.append(neighbor)
        
        # right
        if loc[1]<self.width-1:
            neighbor=(loc[0],loc[1]+1)
            neighbors.append(neighbor)
            
        return neighbors
    
            
    def bfs_count(self,start:tuple,component_idx:int,component_idx_map:np.ndarray):
        visited=np.zeros((self.height,self.width),dtype=bool)
        
        ctr=0
        
        q=queue.Queue()
        visited[start[0],start[1]]=True
        component_idx_map[start[0],start[1]]=component_idx
        ctr+=1
        q.put(start)
        
        while not q.empty():
            curr=q.get()
            neighbors=self.get_loc_neighbors(curr)
            for neighbor in neighbors:
                if not visited[neighbor[0],neighbor[1]] and self.graph[neighbor[0],neighbor[1]]==0:
                    visited[neighbor[0],neighbor[1]]=True
                    component_idx_map[neighbor[0],neighbor[1]]=component_idx
                    ctr+=1
                    q.put(neighbor)
                    
        return ctr
    
    def print_graph(self,graph:np.ndarray=None):
        if graph is None:
            graph=self.graph     
        map=""
        height,width=graph.shape
        for i in range(height):
            for j in range(width):
                map+=str(graph[i,j])
            map+="\n"
        print(map)
        
        
    def zoom_save(self, ofp, new_h, new_w):
        from scipy import ndimage
        scale_h = new_h/self.height
        scale_w = new_w/self.width
        graph = ndimage.zoom(self.graph, (scale_h, scale_w), order=0)
        
        with open(ofp,'w') as f:
            f.write("type octile\n")
            f.write("height "+str(new_h)+"\n")
            f.write("width "+str(new_w)+"\n")
            f.write("map\n")
            map=""
            height,width=graph.shape
            for i in range(height):
                for j in range(width):
                    map+="@" if graph[i,j] else '.'
                map+="\n"
            f.write(map)
        
class MapManager:
    def __init__(self, map_filter_keep=None, map_filter_remove=None):
        self.maps_list=[]
        self.maps_dict={}
        self.instances_list=[]
        self.map_filter_keep=set(map_filter_keep) if map_filter_keep else None
        self.map_filter_remove=set(map_filter_remove) if map_filter_remove else None
        
    def __len__(self):
        return len(self.maps_list)
        
    def load_learn_to_follow_maps(self, maps_path, agent_bins):
        with open(maps_path) as f:
            maps_data = yaml.safe_load(f)

        for k, v in maps_data.items():
            assert k not in self.maps_dict
            
            if self.map_filter_keep is not None and k not in self.map_filter_keep:
                continue
            
            if self.map_filter_remove is not None and k in self.map_filter_remove:
                continue
            
            m = Map()
            m.load_learn_to_follow_map(k,v)
            # TODO make it configurable
            m.set_agent_bins(agent_bins)
            self.add_map(m)
            
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.maps_list[idx]
        elif isinstance(idx, str):
            return self.maps_dict[idx]
        else:
            raise NotImplementedError

    def get_instance(self, sample_idx):
        sample_idx = sample_idx%len(self.instances_list)
        return self.instances_list[sample_idx]
    
    def sample_instance(self):
        idx = np.random.randint(0, len(self.maps_list))
        m:Map = self.maps_list[idx]
        if len(m.agent_bins)>0:
            idx = np.random.randint(0, len(m.agent_bins))
            num_agents = m.agent_bins[idx]
        return m, num_agents
    
    def add_map(self, m:Map):
        assert m.name not in self.maps_dict
        self.maps_dict[m.name] = m
        self.maps_list.append(m)
        
        for agent_num in m.agent_bins:
            self.instances_list.append((m.name,agent_num))


from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch

@dataclass
class Progress:
    step_ctr: int
    curr_positions: torch.Tensor
    target_positions: torch.Tensor
    priorities: torch.Tensor
    
    kwargs: Dict = field(default_factory=lambda: {})

from collections import defaultdict
class ProgressManager:
    def __init__(self, map_manager: MapManager):
        self.map_manager=map_manager
        
        # (map_name, num_agents): progress
        self.instance_progresses=defaultdict(None)
        
        # (num_agents, map_h, map_w): [map_name]
        self.instance_groups=defaultdict(list)
        
        for name, agent_num in self.map_manager.instances_list:
            m: Map = self.map_manager[name]
            self.instance_groups[(agent_num, m.height, m.width)].append(name)

        self.group_infos=list(self.instance_groups.keys())

    def get_group_info(self, idx):
        idx = idx%len(self.group_infos)
        group_info=self.group_infos[idx]
        return group_info
    
    def record_progress(
        self, 
        map_name,
        num_agents,
        progress: Progress = None
        ):
        self.instance_progresses[(map_name,num_agents)]=progress  
        
    def retrieve_progress(self, map_name, num_agents) -> Progress:
        return self.instance_progresses(map_name,num_agents)
                
    def get_instance(self, group_info, idx):
        map_names = self.instance_groups[group_info]
        idx = idx%len(map_names)
        map_name = map_names[idx]
        num_agents=group_info[0]
        
        return map_name, num_agents
    

if __name__=="__main__":
    import json
    
    map_names= [
    # "den_520d_small",
    "warehouse_large",
    "sortation_large",
    # "Paris_1_256_small",
    # "Berlin_1_256_small"
    ]
    task_num=200000
    agent_nums=[
        #600,
        # 8000,
        10000,
        # 12000
    ]
    uniform=True
    
    for map_name in map_names:
        map_fp=f"/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/large/{map_name}.map"
        map=Map(map_fp)
        map.print_graph(map.graph)
        print(len(np.nonzero(map.graph==0)[0]))
        
        task_folder=f"/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/tasks/{map_name}_uniform"
        agent_folder=f"/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/agents/{map_name}_uniform"
        config_folder=f"/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/configs/{map_name}_uniform"
        os.makedirs(task_folder, exist_ok=True)
        os.makedirs(agent_folder, exist_ok=True)
        os.makedirs(config_folder, exist_ok=True)
    
        for idx in range(25):
            task_fp=f"{task_folder}/{map_name}_uniform_{idx}.tasks"
            map.generate_task_files(
                task_num,
                task_fp,
                seed=idx*1997+1234,
                uniform=True
            )
            
            for jdx, agent_num in enumerate(agent_nums):
                agent_fp=f"{agent_folder}/{map_name}_uniform_{idx}_{agent_num}.agents"
                map.generate_agent_files(
                    agent_num,
                    agent_fp,
                    seed=idx*19971+1234+jdx
                )
                
                config = {
                    "mapFile": f"../../small/{map_name}.map",
                    "agentFile": f"../../agents/{map_name}/{map_name}_uniform_{idx}_{agent_num}.agents",
                    "teamSize": agent_num,
                    "taskFile": f"../../tasks/{map_name}/{map_name}_uniform_{idx}.tasks",
                    "numTasksReveal": 1,
                    "taskAssignmentStrategy": "roundrobin"
                }
                        
                config_fp = f"{config_folder}/{map_name}_uniform_{idx}_{agent_num}.json"
                
                with open(config_fp,'w') as f:
                    json.dump(config, f)
    
    # map_fp="/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/large/Berlin_1_256.map"
    # m=Map(map_fp)
    
    # ofp="/root/GRF_MARL/lmapf_lib/data/paper_exp_v3/small/Berlin_1_256_small.map"
    # graph=m.zoom_save(ofp, 64, 64)
    # m.print_graph(graph)
        
import numpy as np
import queue
import os

class Map:
    def __init__(self,fp):
        self.fp=fp
        self.fn=os.path.split(self.fp)[-1]
        self.height=None
        self.width=None
        self.graph=None
        
        self.load(self.fp)
    
    def load(self,fp:str):
        self.fp=fp
        with open(fp,"r") as f:
            # skip the type line
            f.readline()
            self.height=int(f.readline().split()[-1])
            self.width=int(f.readline().split()[-1])
            self.graph=np.zeros((self.height,self.width),dtype=np.int)
            # skip the map line
            f.readline()
            for row in range(self.height):
                line=f.readline().strip()
                assert len(line)==self.width
                for col,loc in enumerate(line):
                    # obstacle
                    if loc=="@" or loc=="T":
                        self.graph[row,col]=1
        # self.print_graph(self.graph)
                        
        self.only_keep_the_main_connected_component()
    
    def only_keep_the_main_connected_component(self):
        component_idx_map=np.zeros((self.height,self.width),dtype=np.int)
        
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
        visited=np.zeros((self.height,self.width),dtype=np.bool)
        
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
    
    def print_graph(self,graph:np.ndarray):
        map=""
        height,width=graph.shape
        for i in range(height):
            for j in range(width):
                map+=str(graph[i,j])
            map+="\n"
        print(map)
        
if __name__=="__main__":
    fp="../our_problems/random.domain/maps/random-32-32-20.map"
    map=Map(fp)
    map.print_graph(map.graph)
        
                    
            
        
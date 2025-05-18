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
        
                    
            
        
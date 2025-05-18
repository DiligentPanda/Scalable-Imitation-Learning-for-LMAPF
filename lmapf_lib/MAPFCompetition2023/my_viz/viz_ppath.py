from map import Map
import numpy as np
import os

class PPATH:
    def __init__(self):
        self.map_file = None
        self.height = None
        self.width = None
        self.paths = None    

def read_file(fp):
    ppath=PPATH()
    with open(fp) as f:
        line = f.readline()
        ppath.map_file = line.strip()
        line = f.readline()
        h,w = line.strip().split()
        ppath.height=int(h)
        ppath.width=int(w)
        ppath.paths = []
        for idx,line in enumerate(f):
            if idx%2==0:            
                continue
            else:
                path = []
                locs=line.strip().split()
                for loc in locs:
                    y, x=loc.split(',')
                    path.append((int(y),int(x)))
                ppath.paths.append(path)
    return ppath

map_folder="../example_problems/random.domain/maps/"
ppath=read_file("../analysis/ppaths/lacam2_50.ppath")
map_fp = os.path.join(map_folder,"random-32-32-20.map")
m = Map(map_fp)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
ax=plt.subplot(1,1,1)
ax.matshow(m.graph)

for i in range(1,m.height):
    ax.axhline(i-0.5, linestyle='--', color='k') # horizontal lines
    
for j in range(1,m.width):
    ax.axvline(j-0.5, linestyle='--', color='k') # vertical lines    


for path in ppath.paths:
    y,x=zip(*path)
    print(x,y)
    ax.plot(x,y)

plt.show()
plt.savefig('test.png')
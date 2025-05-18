import json
import matplotlib.pyplot as plt
import numpy as np

path="../analysis_dev.json"
h=32
w=32

with open(path, 'r') as f:
    analysis = json.load(f)
    
vertex_usage=np.array(analysis["vertexUsage"],dtype=float)
edge_usage=np.array(analysis["edgeUsage"],dtype=float)

print(vertex_usage.sum())
print(edge_usage.sum())

heatmap=np.zeros((h,w),dtype=float)

for i in range(h):
    for j in range(w):
        heatmap[i,j]=vertex_usage[i*w+j]
        
plt.imshow(heatmap, cmap='hot')
plt.xlim(0-10,w+10)
plt.ylim(h+10,0-10)
plt.colorbar()

plt.show()

plt.savefig('random_vertex_usage.png')
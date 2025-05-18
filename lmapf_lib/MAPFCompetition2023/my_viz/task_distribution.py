import numpy as np
import matplotlib.pyplot as plt

path="../example_problems/city.domain/tasks/paris_8000.tasks"
#path="../example_problems/random.domain/tasks/random-32-32-20-100.tasks"
#path="../example_problems/warehouse.domain/tasks/sortation_large.tasks"

h=256
w=256

plt.figure(figsize=(10,10))
points=[]
with open(path) as f:
    f.readline()
    for line in f:
        pos=int(line.strip())
        x=pos%w
        y=pos//w
        points.append((x,y))

points=np.array(points)

plt.scatter(points[:,0],points[:,1],s=1)
plt.xlim(0-10,w+10)
plt.ylim(h+10,0-10)

plt.show()
plt.savefig('test.png')
from map import Map
from instance import Instance
import time

init_random_seed=0

# number to generate, map_fp, num_agents, num_tasks
configs=[
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",20,2000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",50,5000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",100,10000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",200,20000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",400,40000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",600,60000),
    (10,"../our_problems/random.domain/maps/random-32-32-20.map",800,80000)
]

random_seed=init_random_seed
for config_idx,config in enumerate(configs):
    num_instances,map_fp,num_agents,num_tasks=config
    for instance_idx in range(num_instances):
        random_seed=random_seed+19
        map=Map(map_fp)
        # map.print_graph(map.graph)
        instance=Instance(map,num_agents,num_tasks,seed=random_seed)
        instance.save("{}_rs{}".format(instance_idx,random_seed))
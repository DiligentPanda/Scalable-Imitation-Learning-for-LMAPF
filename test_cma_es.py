import cma
import numpy as np

def f(x):
    return np.sum((np.array(x)-np.arange(8,dtype=float))**2)

es = cma.CMAEvolutionStrategy(8 * [0], 0.5)

while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [f(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()

es.result_pretty()

cma.plot() 
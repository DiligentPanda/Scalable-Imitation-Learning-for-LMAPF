import sys
sys.path.append('build')

import py_driver

print(py_driver.playground())

# how do we specify OMP_NUM_THREADS=1, may be directly set in environ?
import os
os.environ["OMP_NUM_THREADS"] = "1"
cmd="./build/lifelong --inputFile example_problems/random.domain/random_400.json -o test_py_driver.json --simulationTime 10 --planTimeLimit 1 --fileStoragePath large_files/"

ret=py_driver.run(cmd=cmd)

import json

analysis=json.loads(ret)

print(analysis["throughput"])
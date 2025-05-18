# Guided-PIBT

The repository contains the implementation of the paper: Zhe Chen, Daniel Harabor, Jiaoyang Li, Peter J. Stuckey Traffic, [Flow Optimisation for Lifelong Multi-Agent Path Finding](https://arxiv.org/abs/2308.11234). The paper was accepted for publication at AAAI 2024.

## Dependency

- Boost >= 1.83.0 (https://www.boost.org/)
- CMake >= 3.10 (https://cmake.org/)

## Instructions for Lifelong Planning with Guided-PIBT

The implementation is built on top of the League of Robot Runners start kit. The following command will build the project for lifelong planning with Guided-PIBT. The codes use a few compiling flags to enable the desired features. Here list the flags and their meanings:

- GUIDANCE: ON/OFF, enable/disable the guide paths
- GUIDANCE_LNS: int, the number of LNS iterations for guide path refinement. If 0, the path refinement is turned off.
- FLOW_GUIDANCE: ON/OFF, enable/disable Traffic Cost Heuristic
- INIT_PP: ON/OFF, when ON, initialising the guide paths one by one and considering the congestion cost from the planned paths (just like PP). When OFF, use the individual shortest paths as initial guide paths.
- RELAX: int, limit the guide path/heuristic table initialisation to a certain number of agents. Agents without guide paths will follow the shortest distance heuristics.
- OBJECTIVE: int, the objective function for the A* that computes the guide path. 0: each move cost constant 1, 1: two-part objective (Contra Flow, 1+Vertex Flow), 2:  1+Vertex Flow, 3: 1+Vertex Flow+Contra Flow, 4. SUI Cost to Come, 5: SUI Cost to Go.
- FOCAL_SEARCH: float, if OFF, the focal search is turned off. If given a number > 1.0, the focal search is turned on with the given focal suboptimality weight.

The path refinement is currently only constructed for OBJECTIVE 1, 2, and 3.

### Example CMake flags to recreate the algorithms in the paper

**GP-R100-Re10-F2**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=10 -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=1 -DFOCAL_SEARCH=2 -DCMAKE_BUILD_TYPE=RELEASE
```

**GP-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=1 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**GP-R100-F1.2**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=1 -DFOCAL_SEARCH=1.2 -DCMAKE_BUILD_TYPE=RELEASE
```

**GP-R100-F2**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=1 -DFOCAL_SEARCH=2 -DCMAKE_BUILD_TYPE=RELEASE
```

**SP-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=OFF -DRELAX=100 -DOBJECTIVE=1 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**GP<sub>v</sub>-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=2 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**GP<sub>v</sub>-R100-F2**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=2 -DFOCAL_SEARCH=2 -DCMAKE_BUILD_TYPE=RELEASE
```

**GP<sub>svc</sub>-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=3 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**GP<sub>sui</sub>-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=5 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**GP<sub>sui</sub>-R100-F2**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=ON -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DRELAX=100 -DOBJECTIVE=5 -DFOCAL_SEARCH=2 -DCMAKE_BUILD_TYPE=RELEASE
```

**TH<sub>v<sub>**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=OFF -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=ON -DINIT_PP=OFF -DRELAX=OFF -DOBJECTIVE=2 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

**TH<sub>v</sub>-R100**
```
cmake -B guided-pibt-build ./guided-pibt -DGUIDANCE=OFF -DGUIDANCE_LNS=OFF -DFLOW_GUIDANCE=ON -DINIT_PP=OFF -DRELAX=100 -DOBJECTIVE=2 -DFOCAL_SEARCH=OFF -DCMAKE_BUILD_TYPE=RELEASE
```

### Compiling

Then compile the code with the following command:
```
make -C guided-pibt-build
```

### Running the code

The following command is an example of running the code with the benchmark file `sortation_small_0_800.json` and the time limit of 10 seconds: 
```
./guided-pibt-build/lifelong --inputFile guided-pibt/benchmark-lifelong/sortation_small_0_800.json --planTimeLimit 10 --output output.json -l event_log.txt 
```


## Instructions for One-shot Planning with Guided-Lacam2

Guided-Lacam uses flat GUID_T to specify the max time limit for the guide path computation. This time limit is not accurate as I need to estimate the time to compute the final guide path heuristic table based on how many agents have guide paths computed.

The Lacam2 code is also modified to optimise the Sum of Individual Costs, with cli flag -O 3.

### Example CMake flags to recreate the algorithms in the paper

**F1.2**
```
cmake -B guided-lacam2-build ./guided-lacam2 -DGUIDANCE=ON -DGUIDANCE_LNS=5000 -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DOBJECTIVE=1  -DCMAKE_BUILD_TYPE=RELEASE -DFOCAL_SEARCH=1.2  -DGUID_T=30  
```


**F2**
```
cmake -B guided-lacam2-build ./guided-lacam2 -DGUIDANCE=ON -DGUIDANCE_LNS=5000 -DFLOW_GUIDANCE=OFF -DINIT_PP=ON -DOBJECTIVE=1 -DCMAKE_BUILD_TYPE=RELEASE -DFOCAL_SEARCH=2  -DGUID_T=30  
```

### Compiling
    
Then compile the code with the following command:
```
make -C guided-lacam2-build
```

### Running the code

```
./guided-lacam2-build/main -m guided-lacam2/assets/random-32-32-10.map -i guided-lacam2/assets/random-32-32-10-random-1.scen -t 60 -N 100 -v 3 -O 3 
```

-t for timelimit
-N for number of agents
-v for verbosity
-O for objective function, 3: Sum of Individual Costs, 2: Sum of Loss, 1: Makespan

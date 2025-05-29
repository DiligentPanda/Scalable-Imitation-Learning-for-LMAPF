# Deploying Ten Thousand Robots: Scalable Imitaion Learning for Lifelong Multi-Agent Path Finding

## NEWS
1. (2025-05-27) I have uploaded pretrained weights for static guidance on the benchmark of this paper. Please refer to the line 4 of the Table IV in the appendix for reproduction. (using `eval.sh`.)
2. (2025-05-27) I have uploaded pretrained weights for Backward Dijkstra heuristics on the learn-to-follow benchmark. Please refer to the Figure 9 in the appendix for reproduction. (using `eval_ltf.sh`.)

## Introduction
This repo maintains the code for the paper. There are some other amazing repos involved and maintained in the `lmapf_lib` folder. 
1. [Guided-PIBT](https://github.com/nobodyczcz/Guided-PIBT)
2. [learn-to-follow](https://github.com/AIRI-Institute/learn-to-follow)
3. [MAPFCompetition2023](https://github.com/DiligentPanda/MAPF-LRR2023): The Winning Solution of the League of Robot Runner Competition 2023. The League of Robot Runner Competition 2024 has a stronger winner: [EPIBT](https://github.com/Straple/LORR24), if you are interested in search-based approaches.
4. [RHCR](https://github.com/Jiaoyang-Li/RHCR)

Examples for training and evaluation are provided as scripts below.  All training and evaluation heavily rely on experiment configs in the `expr_configs`. 

I usually train models with 4 RTX4090D (24GB) and roughly 64 vCPUs. But since it is imitation learning fundamentally, less computational resources also work (need to modifythe experiment configs). 

But this repo is quite messy now for those who need to modify the internal code, because it involves both complex search- and learning-based methods, builds upon a distributed framework for large-scale training, and contains much more than what we have in the paper.

On the hand, the ideas conveyed by the paper are actually straightforward and it is easy to implement for any other works.

Please contact me (reverse:
moc.liamxof@rivers
) if you have any questions.

## Install Libs & Compile PIBT and LNS
```
./compile.sh
```

## Train
See `train.sh`.

## Eval
See `eval.sh` for how to evaluate on the benchmark of this paper.
See `eval_ltf.sh` for how to evaluate on the benchmark of the learn-to-follow paper.

## Benchmark
The benchmark data in this paper is in the `lmapf_lib/data/papere_exp_v3` folder. They use the same data format as the competition [League of the Robot Runner 2023](https://github.com/MAPF-Competition/Benchmark-Archive/tree/main/2023%20Competition).

## TODO
1. recompile everything in an empty env to check the dependencies.
2. upload pretrained model weights.
3. add reproduction scripts.
4. add documentation.
5. organize/re-write code.
    1. remove redundant code.

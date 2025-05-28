#!/bin/bash
set -ex

if [ ! -d large_files ]
then
    mkdir large_files
fi

cd build && make -j16 lifelong && cd ..

# export OMP_NUM_THREADS=1
ARGS="--planTimeLimit 1 --fileStoragePath large_files/"

# # random:random
# # ./build/lifelong --inputFile example_problems/random.domain/random_20.json $ARGS
# # ./build/lifelong --inputFile example_problems/random.domain/random_50.json $ARGS
# # LOAD_FP="random_100_lns_step_500.snapshot" 
# ./build/lifelong --inputFile example_problems/random.domain/random_200.json $ARGS -o test_random_200.json --simulationTime 100
# ./build/lifelong --inputFile example_problems/random.domain/random_200.json $ARGS -o offline_eval/1124/test_random_200.json --simulationTime 500 
./build/lifelong --inputFile example_problems/random.domain/random_400.json $ARGS --simulationTime 1000 --evaluationMode true -o ../../logs/LMAPF/LMAPF_test_BC_1024samples_basic_cnn_re_embed_perm5_bn_1block_random_32_32_20_400agents_2iter_random_perm/2024-05-22-05-15-58/eval/2024-05-23-18-36-17_mode_PIBT-RL_roll_1000steps_lns_5000iters_1threads/log/log_0.json
# ./build/lifelong --inputFile example_problems/random.domain/random_600.json $ARGS -o offline_eval/1124/test_random_600.json --simulationTime 1000 
# ./build/lifelong --inputFile example_problems/random.domain/random_800.json $ARGS -o offline_eval/1124/test_random_800.json --simulationTime 2000 

# # warehouse:warehouse_small
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_10.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_50.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_60.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_100.json $ARGS
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_200.json $ARGS -o offline_eval/1210/test_warehouse_small_200.json --simulationTime 5000
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_400.json $ARGS -o offline_eval/1210/test_warehouse_small_400.json --simulationTime 200
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_600.json $ARGS -o offline_eval/1210/test_warehouse_small_600.json --simulationTime 5000 
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_800.json $ARGS -o offline_eval/1210/test_warehouse_small_800.json --simulationTime 200 
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_small_1000.json $ARGS -o offline_eval/1210/test_warehouse_small_1000.json --simulationTime 5000 

# # warehouse:warehouse_large
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_200.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_400.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_600.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_800.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_1000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_2000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_3000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_4000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_5000.json $ARGS
# ./build/lifelong --inputFile example_problems/warehouse.domain/warehouse_large_8000.json $ARGS -o offline_eval/1124/test_warehouse.json --simulationTime 5000 

# # warehouse:sortation
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_400.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_600.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_800.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_1200.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_1600.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_2000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_2400.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_3000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_5000.json $ARGS
# # ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_8000.json $ARGS
# ./build/lifelong --inputFile example_problems/warehouse.domain/sortation_large_10000.json $ARGS -o offline_eval/1124/test_sortation.json --simulationTime 5000 

# # game:brc202_d
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_200.json $ARGS
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_300.json $ARGS
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_400.json $ARGS 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_500.json $ARGS 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_1000.json $ARGS 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_2000.json $ARGS 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_3000.json $ARGS 
# ./build/lifelong --inputFile example_problems/game.domain/brc202d_4000.json $ARGS  -o offline_eval/1124/test_brc202d.json --simulationTime 5000 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_5000.json $ARGS 
# # ./build/lifelong --inputFile example_problems/game.domain/brc202d_8000.json $ARGS 

# # city:paris
# # ./build/lifelong --inputFile example_problems/city.domain/paris_200.json $ARGS
# # ./build/lifelong --inputFile example_problems/city.domain/paris_300.json $ARGS
# # ./build/lifelong --inputFile example_problems/city.domain/paris_400.json $ARGS
# # ./build/lifelong --inputFile example_problems/city.domain/paris_500.json $ARGS
# ./build/lifelong --inputFile example_problems/city.domain/paris_1000.json $ARGS -o offline_eval/1124/test_paris_1000.json --simulationTime 2000 
# # ./build/lifelong --inputFile example_problems/city.domain/paris_2000.json $ARGS
# ./build/lifelong --inputFile example_problems/city.domain/paris_3000.json $ARGS -o offline_eval/1124/test_paris_3000.json --simulationTime 4000 
# # ./build/lifelong --inputFile example_problems/city.domain/paris_5000.json $ARGS
# # ./build/lifelong --inputFile example_problems/city.domain/paris_8000.json $ARGS
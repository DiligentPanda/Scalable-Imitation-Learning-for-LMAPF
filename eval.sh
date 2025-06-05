# The results of evaluation will appear in the folder,
# exp/eval/.., for example.
# you can take a look at the summaries.json for the mean throughput and inference step time.

## NOTE: For Backward Dijkstra or Static Guidance, the code will precompute the shortest path distances before evaluation.
## It takes a while for the large maps, please be patient. Since it is precomputed only once, its time is not included in the inference step time.
## A variant of implementation could compute the shortest path distances on the fly.

## NOTE: The evaluation on the large maps takes a while. Please be patient.
## NOTE: setting agent number is not supported currently, because the problem instances is generated separately for each number of agents. 
## Most numbers have no instances generated. 
## (TODO) We can just use the problem instances for the largest number of agents and load the required number of agents.

# the root folder for output
OUTPUT_FOLDER=exp/eval
# the root folder for pretrained weights
# IL is trained with our Scalable Imitation Algorithm
# RL is trained with MAPPO
# set to pretrained_models/static_guidance/v3/IL to eval the static guidance.
# set to pretrained_models/backward_dijkstra/v3/IL to eval the backward_dijkstra guidance.
MODEL_FOLDER=pretrained_models/static_guidance/v3/IL 
MAP_WEIGHTS_PATH=DEFAULT
if [[ ${MODEL_FOLDER} == *"backward_dijkstra"* ]]; then
    MAP_WEIGHTS_PATH=NONE
fi

echo "MAP weights path is" ${MAP_WEIGHTS_PATH}

# PIBT, PIBT-RL, PIBT-LNS, PIBT-RL-LNS. 
# PIBT-RL will load pretrained weights, PIBT will just call the original PIBT.
# The LNS version will call LNS after PIBT initialization, used for training.
WPPL_mode=PIBT-RL 
NUM_TESTS=8
NUM_PROCESS=1
NUM_DEVICE=1

######### small maps for training ##########

## NOTE: there is another sortation_small without uniform. 
## It means that the start and goal locations are not generated uniformaly but 
## by rules in the League of Robot Runner 2023 Competition.
# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name sortation_small_uniform \
#  --model_path ${MODEL_FOLDER}/sortation/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 600 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE} \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# ## NOTE: there is another warehouse_small without uniform. 
# ## It means that the start and goal locations are not generated uniformaly but 
# ## by rules in the League of Robot Runner 2023 Competition.
python evaluate.py \
 --output_folder ${OUTPUT_FOLDER} \
 --exp_name warehouse_small_uniform \
 --model_path ${MODEL_FOLDER}/warehouse/best \
 --WPPL_mode ${WPPL_mode} \
 --num_agents 600 \
 --num_processes ${NUM_PROCESS} \
 --num_episodes_per_instance ${NUM_TESTS} \
 --num_device ${NUM_DEVICE}  \
 --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name paris_small \
#  --model_path ${MODEL_FOLDER}/paris/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 600 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name berlin_small \
#  --model_path ${MODEL_FOLDER}/berlin/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 600 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name random_256_10_small \
#  --model_path ${MODEL_FOLDER}/random_256_10/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 600 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name random_256_20_small \
#  --model_path ${MODEL_FOLDER}/random_256_20/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 600 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# ######### large maps for evaluation ##########

# # NOTE: there is another sortation_large without uniform. 
# # It means that the start and goal locations are not generated uniformaly but 
# # by rules in the League of Robot Runner 2023 Competition.
# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name sortation_large_uniform \
#  --model_path ${MODEL_FOLDER}/sortation/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# ## NOTE: there is another warehouse_large without uniform. 
# ## It means that the start and goal locations are not generated uniformaly but 
# ## by rules in the League of Robot Runner 2023 Competition.
# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name warehouse_large_uniform \
#  --model_path ${MODEL_FOLDER}/warehouse/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name paris_large \
#  --model_path ${MODEL_FOLDER}/paris/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name berlin_large \
#  --model_path ${MODEL_FOLDER}/berlin/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name random_256_10_large \
#  --model_path ${MODEL_FOLDER}/random_256_10/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}

# python evaluate.py \
#  --output_folder ${OUTPUT_FOLDER} \
#  --exp_name random_256_20_large \
#  --model_path ${MODEL_FOLDER}/random_256_20/best \
#  --WPPL_mode ${WPPL_mode} \
#  --num_agents 10000 \
#  --num_processes ${NUM_PROCESS} \
#  --num_episodes_per_instance ${NUM_TESTS} \
#  --num_device ${NUM_DEVICE}  \
#  --map_weights_path ${MAP_WEIGHTS_PATH}
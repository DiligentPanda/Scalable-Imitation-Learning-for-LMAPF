# the root folder for pretrained weights
# IL is trained with our Scalable Imitation Algorithm
# RL is trained with MAPPO
MODEL_PATH=pretrained_models/ltf_reeval/v3/IL/best
NUM_TESTS=10
NUM_PROCESS=16
NUM_DEVICE=1

# PIBT, PIBT-RL, PIBT-LNS, PIBT-RL-LNS. 
# PIBT-RL will load pretrained weights, PIBT will just call the original PIBT.
# The LNS version will call LNS after PIBT initialization, used for training.
for WPPL_mode in PIBT PIBT-RL PIBT-IL; do
    for EXP_NAME in ltf_maze ltf_pico ltf_den520d ltf_paris; do
        if [ "$EXP_NAME" == "ltf_pico" ]; then
            NUM_AGENTS_LIST=(8 16 32 64)
        fi

        if [ "$EXP_NAME" == "ltf_maze" ]; then
            NUM_AGENTS_LIST=(32 64 128 256)
        fi

        if [ "$EXP_NAME" == "ltf_den520d" ]; then
            NUM_AGENTS_LIST=(32 64 128 256)
        fi

        if [ "$EXP_NAME" == "ltf_paris" ]; then
            NUM_AGENTS_LIST=(32 64 128 256)
        fi

        for NUM_AGENTS in "${NUM_AGENTS_LIST[@]}"; do
            echo "Evaluating $EXP_NAME with $WPPL_mode and $NUM_AGENTS agents"
            # Run the evaluation script with the specified parameters
            python evaluate.py --exp_name $EXP_NAME --model_path $MODEL_PATH --WPPL_mode $WPPL_mode --num_agents $NUM_AGENTS --num_processes $NUM_PROCESS --num_episodes_per_instance $NUM_TESTS --num_device $NUM_DEVICE
        done
    done
done

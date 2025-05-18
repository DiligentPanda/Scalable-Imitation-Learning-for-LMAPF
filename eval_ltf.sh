MODEL_PATH="logs/ltf_reeval/v3/RL/best"
WPPL_MODE="PIBT"
# PIBT
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 8
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 16
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 256

WPPL_MODE="PIBT"
# PIBT-RL
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 8
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 16
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 256

MODEL_PATH="logs/ltf_reeval/v3/IL/best"
# PIBT-IL
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_maze $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 8
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 16
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_pico $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_den520d $MODEL_PATH $WPPL_MODE 256
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 32
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 64
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 128
python evaluate.py ltf_paris $MODEL_PATH $WPPL_MODE 256

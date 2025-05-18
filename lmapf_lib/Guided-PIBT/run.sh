set -ex

./compile.sh

OUTPUT_DIR="output/"
mkdir -p $OUTPUT_DIR

LNS_NUM_THREADS=32 LNS_MAX_ITERATIONS=40000 ./guided-pibt-build/lifelong --inputFile "/root/GRF_MARL/lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/room-64-64-8_1_3000.json" --planTimeLimit 10 --output ${OUTPUT_DIR}output.json  # -l ${OUTPUT_DIR}event_log.txt 
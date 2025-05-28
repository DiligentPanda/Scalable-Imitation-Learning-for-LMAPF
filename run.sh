set -ex

#./compile.sh

OUTPUT_DIR="output/"
mkdir -p $OUTPUT_DIR


./lmapf_lib/Guided-PIBT/guided-pibt-build/lifelong --inputFile lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/sortation_small_0_600.json --planTimeLimit 10 --output ${OUTPUT_DIR}output.json -l ${OUTPUT_DIR}event_log.txt
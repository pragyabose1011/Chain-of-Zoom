#----------------- Nearest Neighbor -----------------#
# REQUIRED ENVIRONMENT: coz

INPUT_FOLDER="samples"
OUTPUT_FOLDER="inference_results/nearest"

CUDA_VISIBLE_DEVICES=0, python inference_coz.py \
-i $INPUT_FOLDER \
-o $OUTPUT_FOLDER \
--rec_type nearest;


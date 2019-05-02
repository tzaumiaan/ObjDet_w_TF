INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=config/ssd_mobilenet_v1_racecar.config
TRAINED_CKPT_PREFIX=train_racecar_0/model.ckpt-63164
EXPORT_DIR=ssd_mobilenet_racecar_model_63164
TF_MODEL_ROOT=~/workspace/models/research
python ${TF_MODEL_ROOT}/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}

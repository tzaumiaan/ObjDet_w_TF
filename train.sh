python model_main.py \
--pipeline_config_path=config/ssd_mobilenet_v1_coco.config \
--model_dir=train \
--num_train_steps=20000 \
--alsologtostderr \
--sample_1_of_n_eval_examples=1 

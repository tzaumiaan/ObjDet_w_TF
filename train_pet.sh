python model_main.py \
    --model_dir=train_pet_0 \
    --pipeline_config_path=config/ssd_mobilenet_v1_pets.config \
    --num_train_steps=1000 \
    --num_eval_steps=200 \
    --logalsotostderr

python dataset_tools/create_racecar_tf_record.py \
--data_dir=data/racecar \
--set=train \
--output_path=record/racecar_train.record
python dataset_tools/create_racecar_tf_record.py \
--data_dir=data/racecar \
--set=val \
--output_path=record/racecar_val.record

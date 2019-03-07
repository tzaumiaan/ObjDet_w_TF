python dataset_tools/create_fisheye_tf_record.py \
--data_dir=data/fisheye \
--clip=FAU_DriveA \
--set=train \
--output_path=record/fisheye_train.record
python dataset_tools/create_fisheye_tf_record.py \
--data_dir=data/fisheye \
--clip=FAU_DriveA \
--set=val \
--output_path=record/fisheye_val.record

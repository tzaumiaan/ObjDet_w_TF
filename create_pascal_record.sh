python dataset_tools/create_pascal_tf_record.py \
--data_dir=data/VOCdevkit \
--year=VOC2007 \
--set=train \
--output_path=record/pascal_train.record
python dataset_tools/create_pascal_tf_record.py \
--data_dir=data/VOCdevkit \
--year=VOC2007 \
--set=val \
--output_path=record/pascal_val.record

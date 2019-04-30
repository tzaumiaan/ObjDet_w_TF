# ObjDet_w_TF

## Environment setup
Make sure the Tensorflow object detection API is installed beforehand. 

Check if the locations of `tensorflow/models/research` and 
`tensorflow/models/research/slim` are included in `PYTHONPATH`
```
$ echo $PYTHONPATH
```
If no, add it. (Note: here the models are downloaded in `~/workspace/`)
```
$ source env.sh
```

## Necessary package installation
Basic packages include:
```
$ pip install numpy matplotlib opencv-python opencv-contrib-python Pillow
```
Additional packages required by the object detection API:
```
$ pip install lxml Cython pycocotools
```

## Dataset downloading
To download the VOC2007 dataset, run:
```
$ cd data
$ source download_voc2007.sh
$ cd ..
```
The dataset will then be in folder `data/VOCdevkit/VOC2007`.
Now use conversion script to turn the dataset to `TFRecord` format for training.
```
$ source create_pascal_record.sh
```
In `record` folder there will be two `TFRecord` files ready.
```
$ tree record
record
├── pascal_train.record
└── pascal_val.record

0 directories, 2 files
```

## Download pre-trained model
To download the pre-trained model provided by TensorFlow model zoo.
```
$ cd pretrained
$ source download_models.sh
```
Now the checkpoint should be available in `pretrained` folder.
```
$ tree pretrained
pretrained
├── download_models.sh
├── ssd_mobilenet_v1_coco_2018_01_28
│   ├── checkpoint
│   ├── frozen_inference_graph.pb
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   ├── model.ckpt.meta
│   ├── pipeline.config
│   └── saved_model
│       ├── saved_model.pb
│       └── variables
└── ssd_mobilenet_v1_coco_2018_01_28.tar.gz

3 directories, 9 files
```

## Run the training
Use the script to run
```
$ source train_pascal.sh
```
The training folder is `train_pascal_0`, 
and we can use `tensorboard` to monitor the training status.
```
$ tensorboard --logdir=train_pascal_0
```


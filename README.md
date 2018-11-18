# ObjDet_w_TF

## Environment setup
Make sure the Tensorflow object detection API is installed beforehand. 

Check if the locations of `tensorflow/models/research` and 
`tensorflow/models/research/slim` are included in `PYTHONPATH`
```
$ echo $PYTHONPATH
```
If no, add it
```
$ export PYTHONPATH=$PYTHONPATH:~/workspace/models/research/
$ export PYTHONPATH=$PYTHONPATH:~/workspace/models/research/slim/
```

## Necessary package installation
Basic packages include:
```
$ pip install numpy matplotlib opencv-python opencv-contrib-python Pillow
```

Since `model_lib.py` requires `pycocotools` package, 
creating `TFrecord` will requires `lxml`,
we install them by:
```
$ pip install lxml Cython pycocotools
```

## Dataset downloading
To download the VOC2007 dataset, run:
```
$ 
```

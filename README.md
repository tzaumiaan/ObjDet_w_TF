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
Since `model_lib.py` requires `pycocotools` package, 
we have to install it by:
```
$ pip install pycocotools
```

## Dataset downloading
To download the VOC2007 dataset, run:
```
$ 
```

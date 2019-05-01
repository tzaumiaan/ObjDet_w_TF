# Convert racecar dataset to TFRecord for object_detection.
# Example usage:
#   python dataset_tools/create_racecar_tf_record.py \
#       --data_dir=data/racecar \
#       --set=train \
#       --label_map_path=data/racecar_label_map.pbtxt \
#       --output_path=record/racecar_train.record

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
from collections import defaultdict
import PIL.Image
import tensorflow as tf

from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data/racecar', 'Root directory to racecar dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', 'record/racecar.record', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/racecar_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def etree_to_dict(t):
  d = {t.tag: {} if t.attrib else None}
  # traverse children
  children = list(t)
  if children:
    dd = defaultdict(list)
    for dc in map(etree_to_dict, children):
      for k, v in dc.items():
        dd[k].append(v)
    d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
  # append attributes to dictionary
  if t.attrib:
    d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
  # append text part to dictionary 
  if t.text:
    text = t.text.strip()
    if children or t.attrib:
      if text:
        d[t.tag]['#text'] = text
    else:
      d[t.tag] = text
  return d

def parse_cvat_annotation(anno_dict, label_map_dict=None):
  # generate label map, if not given
  if label_map_dict is None:
    label_map_dict, idx = {}, 1
    for l_ in anno_dict['annotations']['meta']['task']['labels']['label']:
      label_map_dict[str(l_['name'])] = int(idx)
      idx += 1
  assert label_map_dict is not None, 'label map invalid'

  # generate object
  object_dict = {}
  for i_ in anno_dict['annotations']['image']:
    if type(i_['box']) != list:
      i_['box'] = [i_['box']]
    object_dict[str(i_['@name'])] = {
        'width': int(i_['@width']),
        'height': int(i_['@height']),
        'box': i_['box']}
  return label_map_dict, object_dict

def process_one_clip(clip, example_writer):
  # parsing CVAT annotation file
  anno_file = os.path.join(FLAGS.data_dir, clip + '.xml')
  assert os.path.exists(anno_file), 'annotation file {} does not exist'.format(anno_file)
  anno_xml_root = etree.parse(anno_file).getroot()
  anno_dict = etree_to_dict(anno_xml_root)
  label_map_dict, object_dict = parse_cvat_annotation(anno_dict)
  del anno_dict
  
  # iterate through images
  img_dir = os.path.join(FLAGS.data_dir, clip)
  img_list = [i_ for i_ in os.listdir(img_dir) if i_.endswith('.jpg')]
  img_list = sorted(img_list)
  assert len(img_list)>0, 'no image files found in {}'.format(img_dir)
  for i_ in img_list:
    # read in one image
    img_file = os.path.join(img_dir, i_)
    with tf.gfile.GFile(img_file, 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width, height = image.size
    
    xmin, ymin, xmax, ymax = [], [], [], []
    classes, classes_text = [], []
    truncated, poses, difficult_obj = [], [], []
    
    if i_ in object_dict:
      assert image.size == (object_dict[i_]['width'], object_dict[i_]['height'])
      diff, trunc, pose = False, False, 'Unspecified' # not used in cvat annotation
      for obj in list(object_dict[i_]['box']):
        xmin.append(float(obj['@xtl']) / width)
        ymin.append(float(obj['@ytl']) / height)
        xmax.append(float(obj['@xbr']) / width)
        ymax.append(float(obj['@ybr']) / height)
        classes_text.append(obj['@label'].encode('utf8'))
        classes.append(label_map_dict[obj['@label']])
        difficult_obj.append(int(0)) # not used in cvat annotation
        truncated.append(int(0)) # not used in cvat annotation
        poses.append('Unspecified'.encode('utf8')) # not used in cvat annotation
    else: 
      tf.logging.debug('{} has no annotation!'.format(i_))
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(i_.encode('utf8')),
        'image/source_id': bytes_feature(i_.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }))
    example_writer.write(example.SerializeToString())
  
  tf.logging.info('clip {} processed successfully'.format(clip))

def get_clips(data_dir, set_name):
  clip_list = []
  clip_list_file = os.path.join(data_dir, set_name+'_list.txt')
  assert os.path.exists(clip_list_file), '{} does not exist'.format(clip_list_file)
  f = open(clip_list_file, 'r')
  for line in f:
    clip_list.append(str(line).strip())
  f.close()
  return clip_list

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  clip_list = get_clips(FLAGS.data_dir, FLAGS.set)
  for c_ in clip_list:
    process_one_clip(c_, writer)
  writer.close()
  tf.logging.info('{} generated!'.format(FLAGS.output_path))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

# adapted from Tensorflow/models/research/object_detection/legacy/eval.py
"""
Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt
"""

import functools
import os
import tensorflow as tf

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import evaluator
from object_detection.utils import config_util
from object_detection.utils import label_map_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string(
    'checkpoint_dir', '',
    'Directory containing checkpoints to evaluate, typically '
    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', '', 'Directory to write eval summaries to.')
flags.DEFINE_string(
    'pipeline_config_path', '',
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
    'file. If provided, other configs are ignored')
flags.DEFINE_boolean(
    'run_once', False, 'Option to only run a single pass of '
    'evaluation. Overrides the `max_evals` parameter in the '
    'provided config.')
FLAGS = flags.FLAGS

def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  configs = config_util.get_configs_from_pipeline_file(
      FLAGS.pipeline_config_path)
  tf.gfile.Copy(
      FLAGS.pipeline_config_path,
      os.path.join(FLAGS.eval_dir, 'pipeline.config'),
      overwrite=True)

  model_config = configs['model']
  eval_config = configs['eval_config']
  input_config = configs['eval_input_config']
  if FLAGS.eval_training_data:
    input_config = configs['train_input_config']

  model_fn = functools.partial(
      model_builder.build, model_config=model_config, is_training=False)

  def get_next(config):
    return dataset_builder.make_initializable_iterator(
        dataset_builder.build(config)).get_next()

  create_input_dict_fn = functools.partial(get_next, input_config)

  categories = label_map_util.create_categories_from_labelmap(
      input_config.label_map_path)

  if FLAGS.run_once:
    eval_config.max_evals = 1

  graph_rewriter_fn = None
  if 'graph_rewriter_config' in configs:
    graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=False)

  evaluator.evaluate(
      create_input_dict_fn,
      model_fn,
      eval_config,
      categories,
      FLAGS.checkpoint_dir,
      FLAGS.eval_dir,
      graph_hook_fn=graph_rewriter_fn)


if __name__ == '__main__':
  tf.app.run()
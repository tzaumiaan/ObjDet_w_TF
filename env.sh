#!/bin/sh
if [ -z $(echo $PYTHONPATH | grep "models/research") ]; then
  echo "Put Tensorflow models path to PYTHONPATH ..."
  export PYTHONPATH=~/workspace/models/research:${PYTHONPATH}
  export PYTHONPATH=~/workspace/models/research/slim:${PYTHONPATH}
else
  echo "Everything set, nothing changes"
fi

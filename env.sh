# environment setup

# change based on the name of your virtual environment
workon ve-py3 

# path of tensorflow API
export TFAPIPATH=~/workspace/models/research
export PYTHONPATH=$PYTHONPATH:$TFAPIPATH/:$TFAPIPATH/slim/

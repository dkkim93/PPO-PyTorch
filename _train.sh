#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# # Conda
# source ~/anaconda3/bin/activate pytorch_p36
# pip install -r requirements_conda.txt

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Begin experiment
python3.6 PPO_continuous.py

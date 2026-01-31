!/bin/bash

conda create -n mit26 python=3.11
conda activate mit26
conda install pip
pip install -r rmsynth/requirements.txt
pip install qiskit

python -m pip install -U pip setuptools wheel
python -m pip install -e rmsynth/


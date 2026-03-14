#bin/bash

. .venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cu126.html

pip install -r requirements.txt

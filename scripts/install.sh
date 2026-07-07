conda create -n dawn python==3.10 -y
conda activate dawn
pip install uv
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
conda install pyhash multicore-tsne -y
sh install.sh
cd ..
uv pip install -r requirements.txt
uv pip install "numpy<2"
uv pip install torch torchvision -U


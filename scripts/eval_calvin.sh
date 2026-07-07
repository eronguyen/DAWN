# WEIGHTS=/home/colligo/Codes/HiVA/DAWN/outputs/DAWN-action-expert/ft-edp-hsv-bs128-skip10-lr1e-4-total50k-fixed/2026-06-17_14-37/checkpoints/model_0009000.pth

accelerate launch --main_process_port=19500 inference.py  #weights.model=${WEIGHTS}
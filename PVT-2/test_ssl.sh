# No SSL
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/None.py --data-path ./ --dataset CIFAR100 --eval

#BYOL
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset CUB --finetune ./checkpoints/BYOL/CUB/pretrain/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset CUB --finetune ./checkpoints/BYOL/CUB/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset SCAR --finetune ./checkpoints/BYOL/SCAR/pretrain/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset SCAR --finetune ./checkpoints/BYOL/SCAR/checkpoint.pth --eval

#SimSiam
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CUB --finetune ./checkpoints/SimSiam/CUB/pretrain/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset SCAR --finetune ./checkpoints/SimSiam/SCAR/pretrain/checkpoint.pth --eval

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset SCAR --finetune ./checkpoints/SimSiam/SCAR/checkpoint.pth --eval

#MAE
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CUB --finetune ./mae_cub.pth --eval

#DINO
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset CUB --finetune ./dino_cub.pth --eval

#MOCO_v3
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MOCOv3.py --data-path ./ --dataset CUB --finetune ./moco_cub.tar --eval

#MOCOv3
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MOCOv3.py --data-path ./ --finetune ./vit-b-300ep-sn.pth.tar

#MAE
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --finetune ./mae_pretrain_vit_base.pth

#BYOL
python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/BYOL.py --data-path ./ --dataset CUB --lr 0.0003 --epoch 500

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset CUB --finetune ./checkpoints/BYOL/CUB/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/BYOL.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset SCAR --finetune ./checkpoints/BYOL/SCAR/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/BYOL.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset CIFAR10 --finetune ./checkpoints/BYOL/CIFAR10/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/BYOL.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/BYOL.py --data-path ./ --dataset CIFAR100 --finetune ./checkpoints/BYOL/CIFAR100/pretrain/checkpoint.pth

#SimSiam
python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CUB --lr 0.0003 --epoch 500

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CUB --finetune ./checkpoints/SimSiam/CUB/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/SimSiam.py --data-path ./ --dataset SCAR

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset SCAR --finetune ./checkpoints/SimSiam/SCAR/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CIFAR10

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CIFAR10 --finetune ./checkpoints/SimSiam/CIFAR10/pretrain/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 --use_env byol.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CIFAR100

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/SimSiam.py --data-path ./ --dataset CIFAR100 --finetune ./checkpoints/SimSiam/CIFAR100/pretrain/checkpoint.pth




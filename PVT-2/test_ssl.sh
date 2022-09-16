# No SSL - using default imagenet_pretrained weights
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/None.py --data-path ./ --dataset CUB --eval 
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/None.py --data-path ./ --dataset NABird --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

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
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CUB --finetune ./pretrained_weights/mae_pretrain_vit_base.pth --eval

#DINO
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset CUB --finetune ./checkpoints/vit_base/CUB/finetune_mae_nabird_pretrain/checkpoint.pth --eval --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset NABird --finetune ./checkpoints/vit_base/NABird/finetune_mae_nabird_pretrain/checkpoint.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset SCAR --finetune ./checkpoints/vit_base/SCAR/finetune_mae_nabird_pretrain/checkpoint.pth --eval --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset CUB --finetune ./checkpoints/vit_base/CUB/finetune_mae_imagenet_pretrain/checkpoint.pth --eval --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset NABird --finetune ./checkpoints/vit_base/NABird/finetune_mae_imagenet_pretrain/checkpoint.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset SCAR --finetune ./checkpoints/vit_base/SCAR/finetune_mae_imagenet_pretrain/checkpoint.pth --eval --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49

#MOCO_v3
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MOCOv3.py --data-path ./ --dataset CUB --finetune ./pretrained_weights/vit-b-300ep.pth.tar --eval

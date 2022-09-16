#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --finetune ./checkpoints/vit_base/NABird/1.0_0.0_12_08_2022_11_54_12/checkpoint.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset NABird --finetune ./checkpoints/vit_base/NABird/1.0_0.0_13_08_2022_21_59_40/checkpoint.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --finetune ./checkpoints/vit_base/SCAR/1.0_0.0_15_08_2022_17_44_22/checkpoint.pth --eval --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset SCAR --finetune ./checkpoints/vit_base/SCAR/1.0_0.0_15_08_2022_11_32_03/checkpoint.pth --eval --num_train_classes 98 --num_aux_classes 49 --num_test_classes 49

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --finetune ./checkpoints/vit_base/CUB/1.0_0.1_17_08_2022_19_45_52/checkpoint.pth --eval --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/backbones/vit_base.py --data-path ./ --dataset CUB --finetune ./checkpoints/vit_base/CUB/finetune_imagenet_supervised_pretrain/checkpoint.pth --eval --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --teacher_ssl

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --student_ssl

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --teacher_ssl --student_ssl

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 0 --wclu	1

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 0 --wclu	1 --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 1 --wclu	1

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 1 --wclu	1 --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 7 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 1 --wclu	1

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 7 --warmup_teacher_temp_epochs 30 --dataset CUB  --ce_warmup_epoch 0 --num_train_classes 80 --num_aux_classes 20 --num_test_classes 100 --wce 0 --wdino 1 --wclu	1 --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset NABird  --ce_warmup_epoch 300 --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50 --wce 0 --wdino 1 --wclu	0 --pretrained_weights checkpoint.pth --batch_size 4 --lr 0.00003 --teacher_ssl --student_ssl

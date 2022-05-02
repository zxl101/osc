#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --teacher_ssl

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --student_ssl

#python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --teacher_ssl --student_ssl

python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --arch vit_base --data_path ./ --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --dataset CUB --pretrained_weights dino_vitbase16_pretrain_full_checkpoint.pth --ce_warmup_epoch 0 --num_train_classes 50 --num_aux_classes 50 --num_test_classes 100

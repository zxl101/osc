#MAE
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/MAE.py --data-path ./ --dataset CUB --finetune ./ssl_pretrained_imagenet/mae_pretrain.pth --eval

#DINO
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset NABird --finetune ./ssl_pretrained_imagenet/dino_pretrain.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset NABird --finetune ./pretrained_weights/freeze/checkpoint0000.pth --eval --num_train_classes 304 --num_aux_classes 50 --num_test_classes 50

#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/SSL/DINO.py --data-path ./ --dataset CUB --finetune ./ssl_pretrained_imagenet/checkpoint.pth --eval --num_train_classes 100 --num_aux_classes 50 --num_test_classes 50


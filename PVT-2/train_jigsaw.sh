#sh dist_train.sh configs/pvt/pvt_small.py 1 --data-path ./ --resume ./ --eval


#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small.py --data-path ./ --finetune ./pvt_small.pth
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small.py --data-path ./ --finetune ./checkpoints/pvt_small/jigsaw_checkpoint.pth
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small2.py --data-path ./ --finetune ./pvt_small.pth
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small2.py --data-path ./ --finetune ./checkpoints/pvt_small2/jigsaw_checkpoint.pth
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small3.py --data-path ./ --finetune ./pvt_small.pth
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small3.py --data-path ./ --finetune ./checkpoints/pvt_small3/jigsaw_checkpoint.pth
    
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small4.py --data-path ./ --finetune ./pvt_small.pth
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small4.py --data-path ./ --finetune ./checkpoints/pvt_small4/jigsaw_checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small5.py --data-path ./ --finetune ./pvt_small.pth
    
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small5.py --data-path ./ --finetune ./checkpoints/pvt_small5/jigsaw_checkpoint.pth
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main2.py --config configs/pvt/pvt_small7.py --data-path ./ --finetune ./pvt_small.pth   
    
#python -m torch.distributed.launch --nproc_per_node=1 \
#    --use_env main.py --config configs/pvt/pvt_small7.py --data-path ./ --finetune ./checkpoints/pvt_small7/jigsaw_checkpoint.pth




#python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b0.py --data-path ./ --finetune ./pvt_v2_b0.pth   
    
#python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b0.py --data-path ./ --finetune ./checkpoints/pvt_v2_b0/jigsaw_checkpoint.pth
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b1.py --data-path ./ --finetune ./pvt_v2_b1.pth   
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b1.py --data-path ./ --finetune ./checkpoints/pvt_v2_b1/jigsaw_checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b2.py --data-path ./ --finetune ./pvt_v2_b2.pth   
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b2.py --data-path ./ --finetune ./checkpoints/pvt_v2_b2/jigsaw_checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b3.py --data-path ./ --finetune ./pvt_v2_b3.pth   
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b3.py --data-path ./ --finetune ./checkpoints/pvt_v2_b3/jigsaw_checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b4.py --data-path ./ --finetune ./pvt_v2_b4.pth   
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b4.py --data-path ./ --finetune ./checkpoints/pvt_v2_b4/jigsaw_checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=1 --use_env main2.py --config configs/pvt_v2/pvt_v2_b5.py --data-path ./ --finetune ./pvt_v2_b5.pth   
    
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config configs/pvt_v2/pvt_v2_b5.py --data-path ./ --finetune ./checkpoints/pvt_v2_b5/jigsaw_checkpoint.pth

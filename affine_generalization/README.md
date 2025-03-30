python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=152.2.132.94 --master_port=1234 unigradicon_train_parallel.py

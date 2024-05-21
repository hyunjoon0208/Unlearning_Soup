### pruning 

#omp
python -u main_imp.py --data ./data --dataset cifar10 --prune_type rewind_lt --rewind_epoch 8 --save_dir result/pruning/omp --pruning_times 2 --num_workers 8
#imp
python -u main_imp.py --data ./data --dataset cifar10 --prune_type rewind_lt --rewind_epoch 8 --save_dir result/pruning/imp --rate 0.2 --pruning_times 2 --num_workers 8
#synFlow
python -u main_imp.py --data ./data --dataset cifar10 --prune_type rewind_lt --rewind_epoch 8 --save_dir result/pruning/synFlow --pruning_times 1 --num_workers 8
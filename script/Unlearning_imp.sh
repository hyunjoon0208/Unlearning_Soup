#### Retrain

python -u main_forget.py --save_dir result/unlearning/imp/ --mask result/pruning/imp --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1

#### FT

python -u main_forget.py --save_dir result/unlearning/imp/ --mask result/pruning/imp --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10

#### GA

python -u main_forget.py --save_dir result/unlearning/imp/ --mask result/pruning/imp --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5

#### FF

python -u main_forget.py --save_dir result/unlearning/imp/ --mask result/pruning/imp --unlearn fisher_new --num_indexes_to_replace 4500 

#### IU

python -u main_forget.py --save_dir result/unlearning/imp/ --mask result/pruning/imp --unlearn wfisher --num_indexes_to_replace 4500 
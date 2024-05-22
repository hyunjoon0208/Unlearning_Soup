#### Retrain

python -u unlearning.py --save_dir result/unlearning/synFlow/ --mask result/pruning/synFlow --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1

#### FT

python -u unlearning.py --save_dir result/unlearning/synFlow/ --mask result/pruning/synFlow --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10

#### GA

python -u unlearning.py --save_dir result/unlearning/synFlow/ --mask result/pruning/synFlow --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5

#### FF

python -u unlearning.py --save_dir result/unlearning/synFlow/ --mask result/pruning/synFlow --unlearn fisher_new --num_indexes_to_replace 4500 

#### IU

python -u unlearning.py --save_dir result/unlearning/synFlow/ --mask result/pruning/synFlow --unlearn wfisher --num_indexes_to_replace 4500 
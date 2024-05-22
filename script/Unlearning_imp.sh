#### Retrain

python -u unlearning.py --save_dir result/unlearning/imp/ --mask result/pruning/imp/0checkpoint.pth.tar --unlearn retrain --num_indexes_to_replace 4500 --unlearn_epochs 160 --unlearn_lr 0.1

#### FT

python -u unlearning.py --save_dir result/unlearning/imp/ --mask result/pruning/imp/0checkpoint.pth.tar --unlearn FT --num_indexes_to_replace 4500 --unlearn_lr 0.01 --unlearn_epochs 10

#### GA

python -u unlearning.py --save_dir result/unlearning/imp/ --mask result/pruning/imp/0checkpoint.pth.tar --unlearn GA --num_indexes_to_replace 4500 --unlearn_lr 0.0001 --unlearn_epochs 5

#### FF

python -u unlearning.py --save_dir result/unlearning/imp/ --mask result/pruning/imp/0checkpoint.pth.tar --unlearn fisher_new --num_indexes_to_replace 4500 

#### IU

python -u unlearning.py --save_dir result/unlearning/imp/ --mask result/pruning/imp/0checkpoint.pth.tar --unlearn wfisher --num_indexes_to_replace 4500 
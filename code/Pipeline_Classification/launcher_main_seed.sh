#!/bin/bash
seeds=(94 118 10 82)
selecs=(10 20 50 100)
clfs=(10 20 50 100 200 500 1000)
ks=(5 6 7 8 9 10 11 12)

for see in ${seeds[@]};
do
for n_iter_sele in ${selecs[@]};
do
for k_fea in ${ks[@]};
do
for n_iter_classi in ${clfs[@]};
do
sbatch --export=n_iter_selec=$n_iter_sele,k_feat=$k_fea,pre_filter=0,seed="$see",clf_only=0,n_iter_classif=$n_iter_classi main.sh
done
done
done
done

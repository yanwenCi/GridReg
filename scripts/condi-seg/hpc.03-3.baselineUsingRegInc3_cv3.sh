#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=96:0:0
#$ -j y
#$ -N cbtRegIn3Cv3
#$ -cwd
hostname
date
python3 -u train.py \
--project WeakSup \
--exp_name hpc.03-3.cbtRegIn3Cv3 \
--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
--batch_size 8 \
--inc 3 \
--cv 3 \
--input_shape 64 101 91 \
--voxel_size 2.0 2.0 2.0 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 100 \
--num_epochs 10000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16
                   
                   
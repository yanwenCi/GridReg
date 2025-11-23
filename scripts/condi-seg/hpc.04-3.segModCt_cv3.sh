#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=96:0:0
#$ -j y
#$ -N segMCTCV3
#$ -cwd
hostname
date
python3 -u train.py \
--project CBCTUnetSeg \
--exp_name hpc.04-3.segModeCTCV3 \
--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
--batch_size 8 \
--input_mode ct \
--inc 1 \
--outc 2 \
--cv 3 \
--input_shape 64 101 91 \
--lr 1e-5 \
--affine_scale 0.15 \
--save_frequency 100 \
--num_epochs 10000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16 \
--two_stage_sampling 0 
                   
                   
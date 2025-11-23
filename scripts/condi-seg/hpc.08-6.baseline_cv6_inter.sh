#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=96:0:0
#$ -j y
#$ -N CS_cv6_inter
#$ -cwd
hostname
date
python3 -u train.py \
--project ConditionalSeg \
--exp_name hpc.08-6.CondisegCBCT_cv6_nc16_inter \
--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
--batch_size 8 \
--cv 6 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 200 \
--num_epochs 20000 \
--w_dce 1.0 \
--using_HPC 1 \
--nc_initial 16 \
--patient_cohort inter
                   
                   

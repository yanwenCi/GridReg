#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=240:0:0
#$ -j y
#$ -N RCnIT_cv1
#$ -cwd
hostname
date
python3 -u train.py \
	--project ConditionalSeg \
	--exp_name hpc.10-1.CS_inter+RC_cv1 \
	--data_path ../data/CBCT/fullResCropIntensityClip_resampled \
	--batch_size 8 \
	--cv 1 \
	--input_shape 64 101 91 \
	--lr 3e-5 \
	--affine_scale 0.15 \
	--save_frequency 500 \
	--num_epochs 50000 \
	--w_dce 1.0 \
	--using_HPC 1 \
	--nc_initial 16 \
	--crop_on_seg_aug 1 \
	--patient_cohort inter

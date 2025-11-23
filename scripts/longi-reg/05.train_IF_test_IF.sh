python -u train.py --project Longitudinal \
	          --exp_name 05.train_IF_test_IF \
		  --data_path ../data/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
		  --key_file key-train-IF-val-IF-test-IF.pkl \
		  --gpu 0 \
		  --batch_size 8 \
		  --patient_cohort intra \
		  --input_shape 128 128 102 \
		  --voxel_size 0.7 0.7 0.7 \
		  --affine_scale 0.1 \
		  --lr 1e-5 \
		  --num_epochs 3000 \
		  --w_bde 50 \
		  --w_ssd 1.0 \
		  --w_dce 1.0 \
		  --w_mmd 0


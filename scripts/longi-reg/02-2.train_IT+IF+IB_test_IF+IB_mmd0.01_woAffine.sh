python -u train.py --project Longitudinal \
	          --exp_name 02-2.train_IT+IF+IB_test_IF+IB_mmd0.01_woAffine \
		  --data_path ../data/AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
		  --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
		  --gpu 1 \
		  --batch_size 8 \
		  --patient_cohort inter+intra \
		  --input_shape 128 128 102 \
		  --voxel_size 0.7 0.7 0.7 \
		  --affine_scale 0 \
		  --lr 1e-5 \
		  --num_epochs 3000 \
		  --w_bde 50 \
		  --w_ssd 1.0 \
		  --w_dce 1.0 \
		  --w_mmd 0.01
		  

python -u train.py --project Icn\
                   --exp_name 13.icn5_grid10_base16\
                   --cropped 0 \
		             --grid_size 10 10 10 \
                   --data_path ../AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 1\
                   --batch_size 8 \
                   --num_epochs 400 \
                   --affine_scale 0 \
                   --lr 1e-4 \
                   --w_bde 100000 \
                   --w_Idsc 1 \
                   --w_Gdsc 1 \
                   --patient_cohort inter+intra \
                   --nc_initial 16 \
                   --num_layers 5 \
                   --in_nc 2\
      

                #    --key_file random-key.pkl \
                #    --voxel_size 0.7 0.7 0.7 \
                #    --affine_scale 0.1 \
                #    --lr 1e-5 \
                #    --num_epochs 3000 \
                #    --ddf_energy_type bending \
                #    --w_bde 50 \
                #    --w_ssd 1.0 \
                #    --w_dce 1.0 \
                #    --w_mmd 0


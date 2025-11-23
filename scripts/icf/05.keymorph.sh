python -u train.py --project Icn\
                   --model KeyMorph \
                   --cropped 0 \
                   --exp_name 14.keymorph6\
		             --grid_size 64 \
                   --crop_size 128 128 96 \
                   --data_path ../AS-morph-interp-ver/0.7-0.7-0.7-64-64-51 \
                   --key_file key-train-IFIB-val-IFIB-test-IFIB.pkl \
                   --gpu 0\
                   --batch_size 4 \
                   --num_epochs 200 \
                   --affine_scale 0 \
                   --lr 1e-4 \
                   --w_bde 1 \
                   --w_Idsc 1 \
                   --w_Gdsc 1 \
                   --patient_cohort inter+intra \
                   --num_control_points 64 \
                   --num_layers 6 \
      

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


# train IF, test on IF
# python train.py --exp_name IF_IF --patient_cohort intra --batch_size 8 --gpu 2 --key_file exp1-key-ordered.pkl --w_mmd 0

# train IF+IB, test on IF
python train.py --exp_name IF+IB_IF --patient_cohort intra --batch_size 8 --gpu 2 --key_file exp1-key-random.pkl --w_mmd 0 

# train IT+IF+IB, test on IF
python train.py --exp_name IT+IF+IB_IF --patient_cohort inter+intra --batch_size 8 --gpu 2 --key_file exp1-key-random.pkl --w_mmd 0

# train IT+IF+IB, test on IF+IB
# python train.py --exp_name IT+IF+IB_IF+IB --patient_cohort inter+intra --batch_size 8 --gpu 2 --key_file exp1-key-random-testIFIB.pkl --w_mmd 0

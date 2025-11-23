import os
from glob import glob
from tqdm import tqdm


log_files = glob('./*.o*')
print(log_files)


failed_exps = []

for lf in tqdm(log_files):
    with open(lf, 'r') as f:
        content = f.readlines()[::-1]
    for c in content:
        if 'Traceback' in c:
            failed_exps.append(lf)
            break

print("following exps might failed, have to check manually")
[print(i) for i in failed_exps]
    
    
import os
from glob import glob 
import sys


models = glob(os.path.join('./logs/**/*.pt'), recursive=True)
models = [i for i in models if 'best' not in i]

for i in models:
    if 'best' in i:
        print("find trying to delete best model")
        sys.exit()
    else:
        print('deleting', i)
        os.remove(i)
        
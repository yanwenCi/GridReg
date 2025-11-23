import os
import sys
from glob import glob

pattern = sys.argv[1]
scripts = glob(pattern, recursive=True)

print('Going to submit following jobs:')
[print(i) for i in scripts]

def run_script(script_path):
    assert os.path.exists(script_path), f"file {script_path} not exists"
    os.system(f"qsub {script_path}")

while(1):
    ans = input("Sure to do that? [Y/N]")
    if ans == "Y":
        [run_script(i) for i in scripts]
        print("All jobs submitted!")
        break
    elif ans == "N":
        sys.exit()
    else:
        print("Only input Y or N")
        
        
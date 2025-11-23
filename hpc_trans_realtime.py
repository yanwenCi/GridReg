import os
from glob import glob
import sys

if len(sys.argv) > 1:
    model_type = sys.argv[1]
    print(f'transfering {model_type} models...')
    assert model_type in ['all', 'best'], "model type not right"
else:
    model_type = None

if len(sys.argv) > 2:
    target_folder_pattern = [
        sys.argv[2]
    ]
else:
    target_folder_pattern = [
        './logs/*/*'
    ]


def scp_to_local(file_path, del_source=False):
    '''
    file_path = './logs/projectName/expname/checkpoints/xxxx.pt'
    '''
    local_project_folder = "/media/yipeng/data/mpmrireg"
    target_path = local_project_folder + os.path.dirname(file_path)[1:]

    command = f"ssh lab \"mkdir -p {target_path}\" && scp {file_path} lab:{target_path}"
    os.system(command)

    if del_source:
        os.remove(file_path)
        print(f'source file {file_path} deleted')




print('Will transfer the following experiment models to local:')

folders = []
for t in target_folder_pattern:
    folders.extend(glob(t))
[print(i) for i in folders]


while(1):
    ans = input("make sure the experiment folders are correct? [y/n]")
    if ans == 'n':
        print("abort.")
        sys.exit()
    elif ans == 'y':
        break
    else:
        print("need to input y or n")

while(1):
    ans = input("please make sure that the local folder structures and versions are correct. [y/n]")
    if ans == 'n':
        print("abort.")
        sys.exit()
    elif ans == 'y':
        break
    else:
        print("need to input y or n")


for f in folders:
    models = glob(os.path.join(f, 'checkpoints', '*.pt'))
    if model_type == 'best':
        models = [i for i in models if 'best' in i]
    elif model_type is None:
        models = [i for i in models if 'best' not in i]
    else:  # all
        models += glob(os.path.join(f, '*.pkl'))

    for i in models:
        print(f'send {i} to local....')
        scp_to_local(i, del_source=True)


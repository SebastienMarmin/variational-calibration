import subprocess
import glob
import os
pdf_viewer = "evince"


exe_files = glob.glob('*.py')
exe_files.remove(os.path.basename(os.path.realpath(__file__)))

for exe_file in exe_files:
    if exe_file == "test_gp.py":
        command = "--dataset simple2 --dataset_dir /home/sebastien/Datasets/ --nmc_train 100 --nmc_test 200 --batch_size 100 --lr 5e-3 --iterations_fixed_noise 1000 --iterations_free_noise 0 --model fgp --test_interval 1000 --nfeatures 50 --verbose --noise_std 2e-2 --full_cov_W 1".split()
    else:
        command = [""]
    
    subprocess.run(["python3",exe_file]+command)



pdf_files = glob.glob('./*.py')
for pdf_file in pdf_files:
    subprocess.run([pdf_viewer,pdf_file])

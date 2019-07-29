import subprocess
import glob
import os
pdf_viewer = "evince"


exe_files = glob.glob('*.py')
exe_files.remove(os.path.basename(os.path.realpath(__file__)))

for exe_file in exe_files:
    if exe_file == "test_gp.py":
        command = "--dataset simple2 --dataset_dir ./test_datasets/ --nmc_train 20 --nmc_test 25 --batch_size 100 --lr 5e-2 --iterations_fixed_noise 50 --iterations_free_noise 0 --model fgp --test_interval 50 --nfeatures 15 --verbose --noise_std 2e-2 --full_cov_W 1".split()
    else:
        command = [""]
    
    subprocess.run(["python3",exe_file]+command)



pdf_files = glob.glob('./*.py')
for pdf_file in pdf_files:
    subprocess.run([pdf_viewer,pdf_file])

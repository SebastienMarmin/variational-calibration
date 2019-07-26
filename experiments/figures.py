import subprocess
import os
import pandas as pd
import numpy as np
import glob
import json
import torch

    # Fetch data
def build_frame(glob_folder, only_most_recent=False):
    bckup_wd = os.getcwd()
    os.chdir(glob_folder)
    json_files = glob.glob('**/results.json', recursive=True)
    print("nb_files:"+str(len(json_files)))
    print(json_files)
    if only_most_recent:
        json_files = (max(json_files, key=os.path.getctime),)

    dfs = []
    for file in json_files:
        try:
            run_time =  os.stat(os.path.dirname(file)).st_mtime 
            newpd = pd.DataFrame(pd.read_json(file, typ='series')).T
            newpd['birth_time']=int(run_time)
            dfs.append(newpd)
        except json.JSONDecodeError:
            print("Error reading "+file+". Skipping.")
        
        #dfs['run'] = "int(os.path.basename(os.path.dirname(file))[-4:])"
    
    df = pd.concat(dfs,sort=False).reset_index()
    del df['index']
    os.chdir(bckup_wd)
    return df

def make():
    scriptpath = os.path.dirname(os.path.realpath(__file__))+"/"
    input_folder = scriptpath+"workspace/"
    results_folder = scriptpath+"results/"
    gen_fig_folder = scriptpath+"figures/gen_fig/"
    output_folder = scriptpath+"figures/"
    data_set_folder = "/home/sebastien/Datasets/"

    ############ Nevada ################
    dataset_name = "calib_nevada"
    gen_fig_path = gen_fig_folder+dataset_name+".R"
    dataset_path = data_set_folder+dataset_name+"/"
    results_path = results_folder+dataset_name+"/"
    df = build_frame(input_folder+dataset_name+"/",only_most_recent=False)


    df_shallow =  df[df["nlayers_run"]==1]
    df_deep    =  df[df["nlayers_run"]>1]

    youngest_run_shallow_indx = int(df_shallow["birth_time"].idxmax(0))
    youngest_run_deep_indx    = int(df_deep["birth_time"].idxmax(0))
    youngest_run_shallow = df[df.index.values==youngest_run_shallow_indx]
    youngest_run_deep    = df[df.index.values==youngest_run_deep_indx]
    folder_shallow=youngest_run_shallow["outdir"]
    folder_deep=youngest_run_deep["outdir"]

    name_shallow = "postMeanShallow"
    name_deep    = "postMeanDeep"
    Y_mean_shallow = torch.Tensor(list(youngest_run_shallow["y_mean"])).t()
    Y_mean_deep    = torch.Tensor(list(youngest_run_deep["y_mean"])).t()

    np.savetxt(results_path+name_shallow+".csv", Y_mean_shallow.numpy(),delimiter=";")
    np.savetxt(results_path+name_deep+".csv", Y_mean_deep.numpy(),delimiter=";")


    # build graph
    command = ["Rscript",gen_fig_path, results_path, output_folder,dataset_path]+[*folder_shallow]+[*folder_deep]
    subprocess.run(command)



    ############ Borehole ################
    dataset_name = "calib_borehole"
    gen_fig_path = gen_fig_folder+dataset_name+".R"
    dataset_path = data_set_folder+dataset_name+"/"
    results_path = results_folder+dataset_name+"/"
    df = build_frame(input_folder+dataset_name+"/",only_most_recent=True)
    folder=df["outdir"]

    # build graph
    mean_list = [str(p) for p in tuple(*df["calib_mean"])]
    sd_list =  [str(p) for p in tuple(*df["calib_stddev"])]
    command = ["Rscript",gen_fig_path, results_path, output_folder,dataset_path]+mean_list+sd_list+[*folder]
    subprocess.run(command)


    ############ Case 2 ################
    dataset_name = "calib_case2"
    gen_fig_path = gen_fig_folder+dataset_name+".R"
    dataset_path = data_set_folder+dataset_name+"/"
    results_path = results_folder+dataset_name+"/"
    df = build_frame(input_folder+dataset_name+"/",only_most_recent=False)


    df_additive =  df[df["additive"]==1]
    df_general  =  df[df["additive"]==0]


    youngest_run_additive_indx = int(df_additive["birth_time"].idxmax(0))
    youngest_run_general_indx    = int(df_general["birth_time"].idxmax(0))
    youngest_run_additive = df[df.index.values==youngest_run_additive_indx]
    youngest_run_general    = df[df.index.values==youngest_run_general_indx]
    folder_additive=youngest_run_additive["outdir"]
    folder_general=youngest_run_general["outdir"]

    # build graph
    mean_list = [str(p) for p in tuple(*youngest_run_additive["calib_mean"])]
    sd_list =  [str(p) for p in tuple(*youngest_run_additive["calib_stddev"])]
    mean_list_NA = [str(p) for p in tuple(*youngest_run_general["calib_mean"])]
    sd_list_NA =  [str(p) for p in  tuple(*youngest_run_general["calib_stddev"])]
    command = ["Rscript",gen_fig_path, results_path, output_folder,dataset_path]+mean_list+sd_list+mean_list_NA+sd_list_NA+[*folder_additive]+[*folder_general]

    subprocess.run(command)

import time
from datetime import datetime, timedelta
t0 = datetime.now()
t = datetime.now()
tf = t0 + timedelta(days=7)
period = timedelta(seconds=5)
flag_loc = "workspace/flag_new.txt"
while t <tf:
    with open(flag_loc, "r") as f:
        if f.read(1)=="1":
            new=True
        else:
            new=False
    if new:
        make()
        print("Figures_done")
        with open(flag_loc, "w") as f:
            f.write('0')
    time.sleep(period.seconds)
    

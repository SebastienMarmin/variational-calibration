import subprocess
import os
import pandas as pd
import numpy as np
import glob
import json
import torch


kw = "" # additive/

external_link =  "/run/user/1000/gvfs/sftp:host=ssh.zoe.eurecom.fr,user=marmin/mnt/cephfs/zoe-workspaces/prod/marmin/"

    # Fetch data
def build_frame(glob_folder, only_most_recent=False):
    bckup_wd = os.getcwd()
    os.chdir(glob_folder)
    json_files = glob.glob('**/results.json', recursive=True)
    print("nb_files:"+str(len(json_files)))
    if only_most_recent:
        json_files = (max(json_files, key=os.path.getctime),)

    dfs = []
    for file in json_files:
        try:
            run_time =  os.stat(os.path.dirname(file)).st_mtime 
            print(file)
            try:
                newpd = pd.DataFrame(pd.read_json(file, typ='series')).T
                newpd['birth_time']=int(run_time)
                dfs.append(newpd)
            except ValueError:
                print("Error reading it.")
        except json.JSONDecodeError:
            print("Error reading "+file+". Skipping.")
        
        #dfs['run'] = "int(os.path.basename(os.path.dirname(file))[-4:])"
    
    df = pd.concat(dfs,sort=False).reset_index()
    del df['index']
    os.chdir(bckup_wd)
    if not kw == "":
        df = df[df["model"]==kw]
    return df

def make(run_id):
    scriptpath = os.path.dirname(os.path.realpath(__file__))+"/"
    input_folder = scriptpath+"workspace/"
    results_folder = scriptpath+"results/"
    gen_fig_folder = scriptpath+"figures/gen_fig/"
    output_folder = run_id#scriptpath+"figures/"+kw+"/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_set_folder = "~/Datasets/"

    if m_nevada:
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
        print("Excecute :"+" ".join(command))
        subprocess.run(command)


    if m_borehole:
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
        print("Excecute :"+" ".join(command))
        subprocess.run(command)

    if m_case2:
        ############ Case 2 ################
        dataset_name = "calib_case2"
        gen_fig_path = gen_fig_folder+dataset_name+".R"
        dataset_path = data_set_folder+dataset_name+"/"
        results_path = results_folder+dataset_name+"/"
        df = build_frame(input_folder+dataset_name+"/",only_most_recent=False)


        df_additive =  df[df["additive"]==1]
        df_general  =  df[df["additive"]==0]

        print("ll")
        print(df["additive"])
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
        print("Excecute :"+" ".join(command))
        subprocess.run(command)

#make()
m_case2 = m_borehole = m_nevada = False
import time
from datetime import datetime, timedelta
t0 = datetime.now()
t = datetime.now()
tf = t0 + timedelta(days=7)
period = timedelta(seconds=2)
flag_loc = "workspace/flag_new.txt"
print("Start watching out for new runs.")
while t <tf:
    with open(flag_loc) as f:
        run_id = f.readline()
        if run_id!="":
            if external_link != "":
                run_id = run_id[15:]
                run_id = external_link+run_id
            new=True
            print("Start analysing "+str(run_id))
        else:
            new=False
    if new:
        print("rid")
        print(run_id)
        if "calib_nevada" in run_id:
            m_nevada = True
            make(run_id)
        elif "calib_borehole" in run_id:
            m_borehole = True
            make(run_id)
        elif "calib_case2" in run_id:
            m_case2 = True
            make(run_id)
        else:
            print("dataset not recognised")
        
        print(str(t)+": figures done (run "+run_id+").\n\n")
        with open(flag_loc, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(flag_loc, 'w') as fout:
            fout.writelines(data[1:])
        
    t = datetime.now()
    time.sleep(period.seconds)
    

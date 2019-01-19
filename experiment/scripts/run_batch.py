import os
import json
import subprocess, shlex
from multiprocessing import Pool
import shutil

def single_run_helper(arg):
    single_run(*arg)

def single_run(work_dir, chain_length, hidden_state_size, window_size, training_size, idx, cpt_selection):
    child_dir = "%s/length_%s_card_%s_window_%s_training_%s_id_%s_cpt_selection_%s"%(work_dir, chain_length, hidden_state_size, window_size, training_size, idx, cpt_selection)
    if not os.path.exists(child_dir):
        try:
            os.mkdir(child_dir)
        except:
            pass
    config = {"training_size": training_size,
              "tac_compiler": "./tbn_compiler",
              "window_length": window_size,
              "working_dir": child_dir,
              "hmm_net_fname_tac_prefix": "%s/tac" % child_dir,
              "testing_size": 10000,
              "hmm_net_fname_ac_prefix": "%s/ac" % child_dir,
              "use_map_variable": False,
              "thmm_net_fname": "%s/thmm.net" % child_dir,
              "hmm_net_fname": "%s/hmm.net" % child_dir,
              "hidden_state_size": hidden_state_size,
              "chain_length": chain_length,
              "cpt_selection": cpt_selection,}
    config_fname = "%s/config.json" % child_dir
    with open(config_fname, "w") as fp:
        json.dump(config, fp, indent=2)
    result_filename = "%s/result.txt" % child_dir
    result = subprocess.check_output(shlex.split("python pipeline.py %s %s" % (config_fname, idx)))
    with open (result_filename, "w") as fp:
        fp.write(result.decode("utf-8"))


if __name__ =="__main__":
    import sys
    core = int(sys.argv[1])
    work_dir = sys.argv[2]
    total_args = []
    for cl in range(8,11):
        for hs in range(2,4):
            for ws in range(2,4):
                for ts in range(8,9):
                    for idx in range(0, 5):
                        for cpt_selection in ["random","gamma"]:
                            cur_arg = (work_dir, cl, hs, ws, 1<<ts, idx, cpt_selection)
                            total_args.append(cur_arg)
    p = Pool(core)
    p.map(single_run_helper, total_args)


import os
import json
import subprocess, shlex
from multiprocessing import Pool
import shutil
import math
import sys

def single_run_helper(arg):
    single_run(*arg)

def single_run(work_dir, chain_length, hidden_state_size, window_size, training_size, idx, parameter_mode):
    child_dir = "%s/length_%s_card_%s_window_%s_training_%s_id_%s_parameter_mode_%s"%(work_dir, chain_length, hidden_state_size, window_size, training_size, idx, parameter_mode)
    if not os.path.exists(child_dir):
        try:
            os.mkdir(child_dir)
        except:
            pass
    config = {"training_size": training_size,
              "tac_compiler": "./tbn_compiler",
              "window_length": window_size,
              "working_dir": child_dir,
              "testing_size": 10000,
              "hidden_state_size": hidden_state_size,
              "chain_length": chain_length,
              "emission_error": 0.1,
              "missing_pr": 0.2,
              "parameter_mode": parameter_mode
              }
    config_fname = "%s/config.json" % child_dir
    with open(config_fname, "w") as fp:
        json.dump(config, fp, indent=2)
    result_filename = "%s/result.txt" % child_dir
    result = subprocess.check_output(shlex.split("python3 pipeline.py %s %s" % (config_fname, idx)))
    with open (result_filename, "w") as fp:
        fp.write(result.decode("utf-8"))

def run_gamma_regression(min_value, max_value, child_dir):
    for i in range(min_value, max_value):
        for j in range(0,5):
            cur_value = math.pow(2, i)
            cur_gamma_config = {"min":cur_value, "max": (cur_value+0.001), "default": cur_value}
            config = {"training_size": 512,
                    "tac_compiler": "./tbn_compiler",
                    "window_length": 2,
                    "working_dir": child_dir,
                    "testing_size": 10000,
                    "hidden_state_size": 2,
                    "chain_length": 6,
                    "emission_error": 0.2,
                    "missing_pr": 0.2,
                    "parameter_mode": "det_transition",
                    "gamma_mode": cur_gamma_config
                }
            with open ("%s/gamma_test_config.json" % child_dir, "w") as fp:
                json.dump(config, fp, indent=2)
            result_filename = "%s/result_%s_%s.txt" % (child_dir,i,j)
            result = subprocess.check_output(shlex.split("python3 pipeline.py %s/gamma_test_config.json %s" % (child_dir, j)))
            with open (result_filename, "w") as fp:
                fp.write(result.decode("utf-8"))

if __name__ == "__main__":
    gamma_min = int(sys.argv[1])
    gamma_max = int(sys.argv[2])
    child_dir = sys.argv[3]
    run_gamma_regression(gamma_min, gamma_max,child_dir)

"""
if __name__ =="__main__":
    import sys
    core = int(sys.argv[1])
    work_dir = sys.argv[2]
    total_args = []
    for cl in range(8, 10):
        for hs in range(2,6):
            for ws in range(2,4):
                for ts in range(9,10):
                    for idx in range(0, 5):
                        for parameter_mode in ["det_transition","peak"]:
                            cur_arg = (work_dir, cl, hs, ws, 1<<ts, idx, parameter_mode)
                            total_args.append(cur_arg)
    p = Pool(core)
    p.map(single_run_helper, total_args)

"""

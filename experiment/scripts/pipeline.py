import subprocess, shlex
import generate_hmm
#import generate_hmm_for_simulate
import json
import learn.data
import learn.tac
import simulate_hmm
import merge_hmm_parameters
import math

def TrainCircuit(fname_prefix, evidence_vars, training_examples, training_labels, testing_examples, test_labels, config):
    gamma_min     = 8
    gamma_default = 16
    gamma_max     = 32
    learn.tac.gamma_min = gamma_min
    learn.tac.gamma_max = gamma_max
    learn.tac.gamma_default = gamma_default
    node_ac = learn.tac.read_tac(fname_prefix+".tac",fname_prefix+".lmap")
    # preprocessing data
    training_inputs = learn.data.binary_dataset_to_tac_inputs(evidence_vars,training_examples,node_ac)
    testing_inputs = learn.data.binary_dataset_to_tac_inputs(evidence_vars,testing_examples,node_ac)
    predictions,weights  = learn.tac.learn(node_ac,training_inputs,training_labels,testing_inputs, rate=.005,thresh=1e-5,iterations=10000)
    # report error
    N_testing  = len(testing_examples)
    N_training = len(training_examples)
    error_mae  = 0.0
    error_mse  = 0.0
    for p1, p2 in zip(predictions,testing_labels):
        error_mae = error_mae + abs(p1-p2)
        error_mse = error_mse + (p1-p2)**2
    error_mae  = error_mae/N_testing
    error_mse  = math.sqrt(error_mse/N_testing)
    return error_mae, error_mse, predictions


def ConstructSimHmmConfig(config, num_example, parameter_seed = 0, simulation_seed = 0):
    sim_config = {}
    sim_config["window_size"] = config["window_length"]
    sim_config["chain_size"] = config["chain_length"]
    sim_config["parameter_seed"] = parameter_seed
    sim_config["hidden_state_size"] = config["hidden_state_size"]
    sim_config["simulate_seed"] = simulation_seed
    sim_config["num_examples"] = num_example
    return sim_config

def save_dataset(examples, chain_size, fp, pr=True):
    fp.write(",".join(["E%s" % i for i in range(0,chain_size)])+"\n")
    for cur_data in examples:
        fp.write(",".join([str(e) for e in cur_data["input"]]))
        if pr :
            fp.write(", %s\n" % cur_data["value"][0])
        else:
            fp.write(", %s\n" % cur_data["label"])

if __name__ == "__main__":
    # Load Config
    import sys
    if len(sys.argv) <= 2:
        print ("Usage: <config> <seed>")
        sys.exit(1)
    config = sys.argv[1]
    seed = int(sys.argv[2])
    with open(config, "r") as fp:
        config = json.load(fp)
    chain_length = config["chain_length"]
    hidden_state_size = config["hidden_state_size"]
    hmm_net_fname = config["hmm_net_fname"]
    ac_fname_prefix = config["hmm_net_fname_ac_prefix"]
    thmm_net_fname = config["thmm_net_fname"]
    tac_fname_prefix = config["hmm_net_fname_tac_prefix"]
    compiler_path = config["tac_compiler"]
    working_dir = config["working_dir"]
    cpt_selection = config["cpt_selection"]
    # simulator config
    window_length = config["window_length"] # window length for simulation
    use_map_variable = config["use_map_variable"]
    # learning
    training_size = config["training_size"]
    testing_size = config["testing_size"]
    # Generate hmm and thmm net
    generate_hmm.GenerateHmmNet(chain_length, hidden_state_size, hmm_net_fname, False) # is_test = False
    generate_hmm.GenerateHmmNet(chain_length, hidden_state_size, thmm_net_fname, True) # is_test = True
    subprocess.check_call(shlex.split("%s %s %s" % (compiler_path, hmm_net_fname, ac_fname_prefix)));
    subprocess.check_call(shlex.split("%s %s %s" % (compiler_path, thmm_net_fname, tac_fname_prefix)));
    # tie parameters
    merge_hmm_parameters.MergeParameters(ac_fname_prefix+".tac", ac_fname_prefix+".lmap", ac_fname_prefix+".tac", ac_fname_prefix+".lmap")
    merge_hmm_parameters.MergeParameters(tac_fname_prefix+".tac", tac_fname_prefix+".lmap", ac_fname_prefix+".tac", ac_fname_prefix+".lmap")
    # Simulate data
    sim_config = ConstructSimHmmConfig(config, training_size + testing_size, seed, seed)
    #generated_examples, _, _, _ = simulate_hmm.GenerateRandomHardDatasetByMissing(sim_config)
    #generated_examples, _, _, _ = simulate_hmm.GenerateRandomHardDatasetByMissingFromHidden(sim_config)
    generated_examples, _, _, _ = simulate_hmm.GenerateRandomHardDatasetWithMCAR(sim_config, cpt_selection)
    #generated_examples, _, _, _ = simulate_hmm.GenerateRandomHardDataset(sim_config)
    training_data = generated_examples[:training_size]
    testing_data = generated_examples[training_size:]
    training_dataset_fname = "%s/training.csv" % working_dir
    testing_dataset_fname = "%s/testing.csv" % working_dir
    with open(training_dataset_fname, "w") as fp:
        save_dataset(training_data, chain_length, fp)
    with open(testing_dataset_fname, "w") as fp:
        save_dataset(testing_data, chain_length, fp)
    # Preprocessing data
    header,training_examples,training_labels = learn.data.read_csv(training_dataset_fname)
    header,testing_examples,testing_labels = learn.data.read_csv(testing_dataset_fname)
    ac_mae,ac_mse, ac_prediction = TrainCircuit(ac_fname_prefix, ["E%s"% i for i in range(0, chain_length)], training_examples, training_labels, testing_examples, testing_labels, config)
    tac_mae, tac_mse, tac_prediction = TrainCircuit(tac_fname_prefix, ["E%s"% i for i in range(0, chain_length)], training_examples, training_labels, testing_examples, testing_labels, config)
    #print (tac_mse/ac_mse)
    #print (tac_mae/ac_mae)
    print ("MSE : TAC, %s, AC, %s" % (tac_mse, ac_mse))
    print ("MAE : TAC, %s, AC, %s" % (tac_mse, ac_mse))
    with open("%s/prediction.txt"%working_dir,'w') as f:
	    f.write("evidence BN  AC  TAC\n")
	    for z in zip(testing_examples,testing_labels,ac_prediction,tac_prediction):
		    f.write("%-20s\t%.6f\t%.6f\t%.6f\n" % z)

def java_sim():
    ## Generates simulating network
    _, evidence_var, query_var, testing_vars, map_vars = generate_hmm_for_simulate.generate_hmm(chain_length, hidden_state_size, window_length, use_map_variable, hmm_sim_fname)
    evidence_filename = "%s/evidence.txt"%working_dir
    training_dataset_fname = "%s/training.dataset" % working_dir
    testing_dataset_fname = "%s/testing.dataset" % working_dir
    with open (evidence_filename, "w") as fp:
        fp.write("%s %s" % (len(evidence_var), " ".join(evidence_var)))
    subprocess.check_call(shlex.split("./simulate.bash RANDOM-HARD %s %s %s 32 %s CONDITIONAL  %s 0" % (hmm_sim_fname, training_dataset_fname, training_size, query_var, evidence_filename)))
    subprocess.check_call(shlex.split("./simulate.bash RANDOM-HARD %s %s %s 32 %s CONDITIONAL  %s 1" % (hmm_sim_fname, testing_dataset_fname, testing_size, query_var, evidence_filename)))

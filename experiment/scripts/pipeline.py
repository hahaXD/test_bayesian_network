import subprocess, shlex
import generate_hmm
import json
import learn.data
import learn.tac
import simulate_hmm
import merge_hmm_parameters
import math
import logging
import hmm
import re
import argparse

#import validation

gamma_min     = 8
gamma_default = 8
gamma_max     = 8.0001
thresh_min    = 0
thresh_max    = 1
ac_learning_rate  = 0.1
tac_learning_rate = 0.01

def TrainCircuit(fname_prefix, evidence_vars, training_examples, training_labels, testing_examples, test_labels, learning_rate, config):
    learn.tac.gamma_min = gamma_min
    learn.tac.gamma_max = gamma_max
    learn.tac.gamma_default = gamma_default
    learn.tac.thresh_min = thresh_min
    learn.tac.thresh_max = thresh_max
    logging.info("Using gamma_min %s, gamma_default %s, gamma_max %s" % (gamma_min, gamma_default, gamma_max))
    node_ac = learn.tac.read_tac(fname_prefix+".tac",fname_prefix+".lmap")
    # preprocessing data
    training_inputs = learn.data.binary_dataset_to_tac_inputs(evidence_vars,training_examples,node_ac)
    testing_inputs = learn.data.binary_dataset_to_tac_inputs(evidence_vars,testing_examples,node_ac)
    predictions,weights  = learn.tac.learn(node_ac,training_inputs,training_labels,testing_inputs, rate=learning_rate,thresh=1e-5,iterations=10000, grd_thresh=1e-5)
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
    return error_mae, error_mse, predictions, node_ac, weights


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

def get_hmm_parameter_generator(config):
    parameter_mode = config.setdefault("parameter_mode", "peak")
    hidden_state_size = config["hidden_state_size"]
    window_length = config["window_length"] # window length for simulation
    emission_error = config.setdefault("emission_error", 0.2)
    if parameter_mode == "peak":
        logging.info("Parameter generation mode peak is used.")
        return hmm.HmmParameterGenerateorWithPeak(hidden_state_size, window_length, emission_error);
    elif parameter_mode == "det_transition":
        logging.info("Parameter generation mode det_transition is used.")
        return hmm.HmmParameterGeneratorDetTransition(hidden_state_size, window_length, emission_error)
    else:
        logging.error("Parameter generation mode {0} cannot be recognized, using peak instead.".format(parameter_mode))
        return hmm.HmmParameterGenerateorWithPeak(hidden_state_size, window_length, emission_error);

def logging_learned_matrix(lmap_fname):
    initial_pattern = re.compile(r"[0-9]+ p ([0-9.e-]+) H0=([0-9]+) \| *")
    transition_pattern = re.compile(r"[0-9]+ p ([0-9.e-]+) Hi=([0-9]+) \| Hj=([0-9]+) *")
    emission_pattern = re.compile(r"[0-9]+ p ([0-9.e-]+) Ei=([0-9]+) \| Hi=([0-9]+) *")
    thres_pattern = re.compile(r"[0-9]+ p ([0-9.e-]+) Hi \| Hj=([0-9]+) Thres")
    gamma_pattern = re.compile(r"[0-9]+ p ([0-9.e-]+) Hi \| Hj=([0-9]+) Gamma")
    initial_pr = {}
    positive_transition = {}
    negative_transition = {}
    transition = {}
    emission = {}
    thres = {}
    gamma = {}
    with open(lmap_fname, "r") as fp:
        for line in fp:
            match = initial_pattern.match(line)
            if match:
                initial_pr[match.group(2)] = float(match.group(1))
                continue
            match = transition_pattern.match(line)
            if match:
                key = (match.group(2), match.group(3))
                pr = float(match.group(1))
                if " +" in line:
                    # positive parameter
                    positive_transition[key] = pr
                elif " -" in line:
                    # negative parameter
                    negative_transition[key] = pr
                else:
                    # regular parameter
                    transition[key] = pr
                continue
            match = emission_pattern.match(line)
            if match:
                emission[(match.group(2), match.group(3))] = float(match.group(1))
                continue
            match = gamma_pattern.match(line)
            if match:
                gamma[match.group(2)] = match.group(1)
                continue
            match = thres_pattern.match(line)
            if match:
                thres[match.group(2)] = match.group(1)
                continue
    result = {}
    if len(initial_pr) != 0:
        result["initial_pr"] = initial_pr
    if len(transition) != 0:
        result["transition"] = transition
    if len(positive_transition) != 0:
        result["positive_transition"] = positive_transition
    if len(negative_transition) != 0:
        result["negative_transition"] = negative_transition
    if len(emission) != 0:
        result["emission"] = emission
    if len(gamma) != 0:
        result["gamma"] = gamma
    if len(thres) != 0:
        result["thres"] = thres
    if "initial_pr" in result:
        logging.info("Initial Pr:")
        for i in range(0, len(initial_pr)):
            logging.info("Pr(H0=%s)=%s"%(i, initial_pr[str(i)]))
    if "transition" in result:
        logging.info("Transition Pr:")
        num_state = int(math.sqrt(len(transition)))
        for i in range(0, num_state):
            for j in range(0, num_state):
                logging.info("Pr(Hi=%s|Hj=%s) = %s" % (i,j, transition[(str(i),str(j))]))
    if "positive_transition" in result:
        logging.info("Positive transition Pr:")
        num_state = int(math.sqrt(len(positive_transition)))
        for i in range(0, num_state):
            for j in range(0, num_state):
                logging.info("Pr(Hi=%s|Hj=%s) = %s" % (i,j, positive_transition[(str(i),str(j))]))
    if "negative_transition" in result:
        logging.info("Negative transition Pr:")
        num_state = int(math.sqrt(len(negative_transition)))
        for i in range(0, num_state):
            for j in range(0, num_state):
                logging.info("Pr(Hi=%s|Hj=%s) = %s" % (i,j, negative_transition[(str(i),str(j))]))
    if "thres" in result:
        logging.info("Thres :")
        for i in range(0, len(thres)):
            logging.info("Thres(Hj=%s) = %s" % (i, thres[str(i)]))
    if "gamma" in result:
        logging.info("Gamma :")
        for i in range(0, len(gamma)):
            logging.info("Gamma(Hj=%s) = %s" % (i, gamma[str(i)]))
    return result


def logging_config(config):
    for key in config:
        logging.info("%s: %s" % (key, config[key]))

if __name__ == "__main__":
    # Load Config
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="running configuration.")
    parser.add_argument("seed", type=int, help="seed used for sample parameters and training data.")
    args = parser.parse_args()
    config = args.config
    seed = args.seed
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.info("config: %s" % config)
    logging.info("seed: %s" % seed)
    with open(config, "r") as fp:
        config = json.load(fp)
    chain_length = config["chain_length"]
    hidden_state_size = config["hidden_state_size"]
    compiler_path = config["tac_compiler"]
    working_dir = config["working_dir"]
    emission_error = config.setdefault("emission_error", 0.2)
    missing_mode = config.setdefault("missing_mode", "MCAR")
    missing_pr = config.setdefault("missing_pr", 0)
    gamma_config = config.setdefault("gamma_config", {"min": -8, "max":8, "default":0})
    gamma_min = gamma_config["min"]
    gamma_max = gamma_config["max"]
    gamma_default = gamma_config["default"]
    hmm_net_fname = "%s/hmm.net" % working_dir
    ac_fname_prefix = "%s/hmm" % working_dir
    thmm_net_fname = "%s/thmm.net" % working_dir
    tac_fname_prefix = "%s/thmm" % working_dir
    logging_config(config)
    # simulator config
    window_length = config["window_length"] # window length for simulation
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
    merge_hmm_parameters.MergeParameters(tac_fname_prefix+".tac", tac_fname_prefix+".lmap", tac_fname_prefix+".tac", tac_fname_prefix+".lmap")
    # Simulate data
    parameter_generator = get_hmm_parameter_generator(config)
    true_model = hmm.Hmm(chain_length, window_length, hidden_state_size, parameter_generator)
    generated_examples = true_model.generate_dataset(training_size + testing_size, missing_pr, missing_mode, seed);
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
    ac_mae,ac_mse, ac_prediction, ac_node, ac_weight = TrainCircuit(ac_fname_prefix, ["E%s"% i for i in range(0, chain_length)], training_examples, training_labels, testing_examples, testing_labels, ac_learning_rate,config)
    tac_mae, tac_mse, tac_prediction, tac_node, tac_weight = TrainCircuit(tac_fname_prefix, ["E%s"% i for i in range(0, chain_length)], training_examples, training_labels, testing_examples, testing_labels, tac_learning_rate,config)
    print ("MSE : TAC, %s, AC, %s, Ratio, %s" % (tac_mse, ac_mse, tac_mse / ac_mse))
    print ("MAE : TAC, %s, AC, %s, Ratio, %s" % (tac_mae, ac_mae, tac_mae / ac_mae))
    ac_weight_lmap_fname = "%s/trained_ac.lmap" % working_dir
    tac_weight_lmap_fname = "%s/trained_tac.lmap" % working_dir
    ac_network_fname = "%s.tac" % ac_fname_prefix
    tac_network_fname = "%s.tac" % tac_fname_prefix
    prediction_fname = "%s/prediction.txt" % working_dir
    learn.tac.Literal.lmap_write(ac_node.lmap, ac_weight, ac_weight_lmap_fname)
    learn.tac.Literal.lmap_write(tac_node.lmap, tac_weight, tac_weight_lmap_fname)
    with open(prediction_fname,'w') as f:
	    f.write("evidence BN  AC  TAC\n")
	    for z in zip(testing_examples,testing_labels,ac_prediction,tac_prediction):
		    f.write("%-20s\t%.6f\t%.6f\t%.6f\n" % z)
    logging.info("Logging learned parameters for AC:")
    logging_learned_matrix(ac_weight_lmap_fname)
    logging.info("Logging learned parameters for TAC:")
    logging_learned_matrix(tac_weight_lmap_fname)
    logging.info("Running verification")
    #validation.validate(ac_network_fname, ac_weight_lmap_fname, tac_network_fname, tac_weight_lmap_fname, prediction_fname)

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

import random
import math
import logging

def simulate_from_distribution(dist):
    cdf = [dist[0]]
    for i in range(1, len(dist)):
        cdf.append(cdf[-1] + dist[i])
    cur_rand = random.random()
    for j in range(0, len(dist)):
        if cur_rand < cdf[j]:
            return j
    # should not happen
    return len(dist) -1;


# last item in state_conf represents the child's state
def index_from_conf(state_conf, state_size):
    index = 0;
    index = state_conf[0]
    for i in range(1, len(state_conf)):
        index *= state_size
        index += state_conf[i]
    return index

def decode_index(index, state_size, conf_size):
    conf = []
    for i in range(0, conf_size):
        conf.append(index % state_size)
        index = index//state_size
    return list(reversed(conf))




def generate_distribution(num_states,alpha=0.5,beta=1.0, cpt_selection="gamma"):
    if cpt_selection == "random":
        pr = [random.random() for i in range(num_states)]
    else:
        pr = [ random.gammavariate(alpha,beta) for i in range(num_states) ]
    pr_sum = sum(pr)
    return [ p/pr_sum for p in pr ]

def GeneratePriorDistribution(config, cpt_selection = "gamma", alpha=0.5, beta=1.0):
    cpt_selection = config["cpt_selection"]
    hidden_state_size = config["hidden_state_size"]
    window_length = config["window_length"]
    factor_size = int(math.pow(hidden_state_size, window_length))
    logging.info("Generating prior using %s " % cpt_selection)
    pr = generate_distribution(factor_size, cpt_selection=cpt_selection)
    pr_string = "\n".join(["%s : %s "% (decode_index(i, hidden_state_size, window_length), pr[i]) for i in range(0, factor_size)])
    logging.info("Prior Distribution :\n %s" % pr_string)
    return pr

def GenerateTransitionDistribution(config):
    cpt_selection = config["cpt_selection"]
    hidden_state_size = config["hidden_state_size"]
    window_length = config["window_length"]
    factor_size = int(math.pow(hidden_state_size, window_length+1))
    parent_conf_size = factor_size // hidden_state_size
    conditional_cpt = []
    logging.info("Generating transition using %s " % cpt_selection)
    for i in range(0, parent_conf_size):
        start_index = i * hidden_state_size
        end_index = (i+1) * hidden_state_size
        local_distribution = generate_distribution(hidden_state_size, cpt_selection=cpt_selection)
        conditional_cpt += local_distribution
    conditional_cpt_string = "\n".join(["%s"%decode_index(i, hidden_state_size, window_length) + " ".join([str(tok) for tok in conditional_cpt[i*hidden_state_size:(i+1)*hidden_state_size]]) for i in range(0, parent_conf_size)])
    logging.info("Conditional Distribution :\n %s" % conditional_cpt_string)
    return conditional_cpt

def GenerateEmissionDistribution(config):
    cpt_selection = config["cpt_selection"]
    hidden_state_size = config["hidden_state_size"]
    emission_cpt = [None] * hidden_state_size
    logging.info("Generating emission using %s " % cpt_selection)
    for h_state in range(0, hidden_state_size):
        emission_cpt[h_state] = generate_distribution(2, cpt_selection=cpt_selection)
    logging.info("Emission Distribution : %s" % emission_cpt)
    return emission_cpt

def SimulateHmm(config, num_examples, seed):
    window_length = config["window_length"]
    chain_length = config["chain_length"]
    hidden_state_size = config["hidden_state_size"]
    random.seed(seed)
    # simulate parameters
    prior_hidden_cpt = GeneratePriorDistribution(config)
    conditional_cpt = GenerateTransitionDistribution(config)
    emission_cpt = GenerateEmissionDistribution(config)
    # simulate examples
    random.seed(seed)
    examples = []
    for i in range(0, num_examples):
        # simulate Hs
        simulated_h_start = simulate_from_distribution(prior_hidden_cpt)
        h_start_conf = decode_index(simulated_h_start, hidden_state_size, window_length)
        simulated_e = []
        simulated_h = list(h_start_conf)
        for j in range(window_length, chain_length):
            parent_conf = simulated_h[-window_length :]
            parent_conf_index = index_from_conf(parent_conf,hidden_state_size)
            local_cpt = conditional_cpt[parent_conf_index* hidden_state_size : (parent_conf_index+1) * hidden_state_size]
            cur_simulated_j = simulate_from_distribution(local_cpt)
            simulated_h.append(cur_simulated_j)
        # simulate emissions
        for j in range(0, chain_length):
            local_cpt = emission_cpt[simulated_h[j]]
            cur_simulated_e = simulate_from_distribution(local_cpt)
            simulated_e.append(cur_simulated_e)
        examples.append({"H":simulated_h, "E":simulated_e})
    return prior_hidden_cpt, conditional_cpt, emission_cpt, examples

def inference(prior_cpt, conditional_cpt, emission_cpt, observation, hidden_state_size, chain_length, window_length):
    # generate soft evidence on hidden states
    h_softs = []
    for obs in observation:
        h_softs.append([a[0]* obs + a[1] * (1-obs) for a in emission_cpt])
    # Adding priors to prior hs
    message = list(prior_cpt)
    for i in range(0, len(prior_cpt)):
        cur_conf = decode_index(i, hidden_state_size, window_length)
        for j in range(0, window_length):
            message[i] *= h_softs[j][cur_conf[j]]
    # sum out h_s
    message_size = len(message)
    for i in range(window_length, chain_length):
        new_message = []
        size_after_multiply = message_size * hidden_state_size
        for j in range(0, size_after_multiply):
            cur_soft_evid = h_softs[i][j%hidden_state_size]
            new_message.append(message[j // hidden_state_size] * conditional_cpt[j] * cur_soft_evid)
        #sumout h_{i-window_length}
        for j in range(0, message_size):
            message[j] = 0
            for k in range(0, hidden_state_size):
                message[j] += new_message[j + k * message_size]
    query_result = [0] * hidden_state_size
    for i in range(0, message_size):
        cur_state = i % hidden_state_size
        query_result[cur_state] += message[i]
    return query_result


def GenerateRandomHardDatasetWithMCAR(config, num_examples, seed):
    hidden_state_size = config["hidden_state_size"]
    chain_length = config["chain_length"]
    simulate_seed = seed
    window_length = config["window_length"]
    missing_pr = config["missing_pr"] if config["missing_pr"] >= 0 else random.random()
    logging.info("Missing Pr : %s" % missing_pr)
    prior_cpt, conditional_cpt, emission_cpt, examples = SimulateHmm(config, num_examples, seed)
    missing_result = []
    random.seed(simulate_seed)
    for cur_data in examples:
        cur_record = {}
        evidence = []
        for i in range(0, chain_length):
            if random.random() < missing_pr:
                # missing
                evidence.append(0.5)
            else:
                evidence.append(0.0 if cur_data["E"][i] == 1 else 1.0)
        pr = inference(prior_cpt, conditional_cpt, emission_cpt, evidence, hidden_state_size, chain_length, window_length)
        pr_evid = sum(pr)
        cur_record["input"] = evidence
        cur_record["label"] = cur_data["H"][-1]
        cur_record["value"] = [ a / pr_evid for a in pr]
        missing_result.append(cur_record)
    return missing_result, prior_cpt, conditional_cpt, emission_cpt

"""
def GenerateRandomHardDatasetByMissingFromHidden(hmm_config):
    num_examples = hmm_config ["num_examples"]
    hidden_state_size = hmm_config["hidden_state_size"]
    chain_length = hmm_config["chain_length"]
    simulate_seed = hmm_config["simulate_seed"]
    window_length = hmm_config["window_length"]
    missing_cpt = [None]*hidden_state_size
    picked_state = simulate_from_distribution(generate_distribution(hidden_state_size))
    for i in range(0, hidden_state_size):
        if i == picked_state:
            missing_cpt[i] = [0.8, 0.2]
        else:
            missing_cpt[i] = [0.2, 0.8]
    prior_cpt, conditional_cpt, emission_cpt, examples = SimulateHmm(hmm_config, num_examples)
    missing_result = []
    random.seed(simulate_seed)
    for cur_data in examples:
        cur_record = {}
        evidence = []
        for i in range(0, chain_length):
            cur_missing_cpt = missing_cpt[cur_data["H"][i]]
            if random.random() < cur_missing_cpt[0]:
                # missing
                evidence.append(0.5)
            else:
                evidence.append(0.0 if cur_data["E"][i] == 1 else 1.0)
        pr = inference(prior_cpt, conditional_cpt, emission_cpt, evidence, hidden_state_size, chain_length, window_length)
        pr_evid = sum(pr)
        cur_record["input"] = evidence
        cur_record["label"] = cur_data["H"][-1]
        cur_record["value"] = [ a / pr_evid for a in pr]
        missing_result.append(cur_record)
    return missing_result, prior_cpt, conditional_cpt, emission_cpt

def GenerateRandomHardDatasetByMissing(hmm_config):
    num_examples = hmm_config ["num_examples"]
    hidden_state_size = hmm_config["hidden_state_size"]
    chain_length = hmm_config["chain_length"]
    simulate_seed = hmm_config["simulate_seed"]
    window_length = hmm_config["window_length"]
    missing_cpt = [None]*2
    missing_cpt[0] = generate_distribution(2)
    missing_cpt[1] = generate_distribution(2)
    prior_cpt, conditional_cpt, emission_cpt, examples = SimulateHmm(hmm_config, num_examples)
    missing_result = []
    random.seed(simulate_seed)
    for cur_data in examples:
        cur_record = {}
        evidence = []
        for i in range(0, chain_length):
            cur_missing_cpt = missing_cpt[cur_data["E"][i]]
            if random.random() < cur_missing_cpt[0]:
                # missing
                evidence.append(0.5)
            else:
                evidence.append(0.0 if cur_data["E"][i] == 1 else 1.0)
        pr = inference(prior_cpt, conditional_cpt, emission_cpt, evidence, hidden_state_size, chain_length, window_length)
        pr_evid = sum(pr)
        cur_record["input"] = evidence
        cur_record["label"] = cur_data["H"][-1]
        cur_record["value"] = [ a / pr_evid for a in pr]
        missing_result.append(cur_record)
    return missing_result, prior_cpt, conditional_cpt, emission_cpt

hmm_config = {}
hmm_config["window_length"] = 2
hmm_config["chain_length"] = 10
hmm_config["parameter_seed"] = 0
hmm_config["hidden_state_size"] = 2
hmm_config["simulate_seed"] = 0
hmm_config["num_examples"] = 1000
#missing_result, _, _,_ = GenerateRandomHardDatasetByMissing(hmm_config)
#print missing_result


prior, conditional, emission, simulated_result = SimulateHmm(hmm_config, 100000)

sampled_result = [0]*hmm_config["hidden_state_size"]
total = 0
for i in range(0, 100000):
    if simulated_result[i]["E"][-1] == 0:
        total += 1
        sampled_result[simulated_result[i]["H"][-1]] += 1

sampled_result = [float(a)/total for a in sampled_result]
print sampled_result
observation = [0.5] * hmm_config["chain_length"]
observation[-1] = 1
inf_result = inference(prior, conditional, emission, observation, hmm_config["hidden_state_size"], hmm_config["chain_length"], hmm_config["window_length"])
partition = sum(inf_result)
print [a/partition for a in inf_result]

"""

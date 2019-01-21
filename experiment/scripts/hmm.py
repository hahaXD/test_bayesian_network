import random
import numpy as np
import math
import random
import logging

def sample_from_distribution(distribution, sim_gen):
    acc = 0
    cur_rand = sim_gen.random()
    for i in range(0, len(distribution)):
        acc += distribution[i]
        if (cur_rand < acc):
            return i
    return len(dist) - 1;

class HmmParameterGeneratorRandomTransition:
    def __init__(self,  cardinality, window, seed):
        self.random_generator = random.Random(seed)
        self.cardinality = cardinality
        self.window_size = window

    def generate_from_dirchlet (self, exponents):
        pr = [ self.random_generator.gammavariate(cur_exponent, 1.0) for cur_exponent in exponents]
        pr_sum = sum(pr)
        return [ p/pr_sum for p in pr ]

    def generate_initial_cpt(self):
        parameter_size = int(math.pow(self.cardinality, self.window_size))
        return np.array(self.generate_from_dirchlet([1 for i in range(0, parameter_size)])).reshape([self.cardinality]*self.window_size)

    def generate_transition_cpt(self):
        num_parent_states = int(math.pow(self.cardinality, self.window_size))
        cpt = np.zeros([self.cardinality]*(self.window_size + 1)).reshape([num_parent_states, self.cardinality])
        for i in range(0, num_parent_states):
            cur_parent_config = np.unravel_index(i, [self.cardinality] * self.window_size)
            cur_state = (cur_parent_config[0] + 1) % self.cardinality
            cur_dist =  self.generate_from_dirchlet([1 for i in range(0, self.cardinality)])
            cpt[i] = cur_dist
        return cpt.reshape([self.cardinality] * (self.window_size+1))

    def generate_emission_cpt(self):
        emission_fp = 1.0 / (1.0 + math.exp(self.random_generator.gauss(4, 2)))
        emission_fp = min(emission_fp, 1 - emission_fp)
        emission_fn = 1.0 / (1.0 + math.exp(self.random_generator.gauss(4, 2)))
        emission_fn = min(emission_fn, 1 - emission_fn)
        cpt = np.zeros([self.cardinality, 2])
        for i in range(0, self.cardinality):
            # sensing whether the hidden state id is even or odd
            if i % 2 == 0:
                cpt[i][0] = 1 - emission_fp
                cpt[i][1] = emission_fp
            else:
                cpt[i][0] = emission_fn
                cpt[i][1] = 1 - emission_fn
        return cpt



class HmmParameterGeneratorDetTransition:
    def __init__(self, cardinality, window_size, emission_error):
        self.cardinality = cardinality
        self.window_size = window_size
        self.emission_error = emission_error

    def generate_initial_cpt(self):
        parameter_size = int(math.pow(self.cardinality, self.window_size))
        return np.array([1.0/(parameter_size) for i in range(0, parameter_size)]).reshape([self.cardinality]*self.window_size) # uniform

    def generate_transition_cpt(self):
        num_parent_states = int(math.pow(self.cardinality, self.window_size))
        cpt = np.zeros([self.cardinality]*(self.window_size + 1)).reshape([num_parent_states, self.cardinality])
        for i in range(0, num_parent_states):
            cur_parent_config = np.unravel_index(i, [self.cardinality] * self.window_size)
            cur_state = (cur_parent_config[0] + 1) % self.cardinality
            cur_dist = [0 if i != cur_state else 1 for i in range(0, self.cardinality)]
            cpt[i] = cur_dist
        return cpt.reshape([self.cardinality] * (self.window_size+1))

    def generate_emission_cpt(self):
        cpt = np.zeros([self.cardinality, 2])
        for i in range(0, self.cardinality):
            # sensing whether the hidden state id is even or odd
            if i % 2 == 0:
                cpt[i][0] = 1 - self.emission_error
                cpt[i][1] = self.emission_error
            else:
                cpt[i][0] = self.emission_error
                cpt[i][1] = 1 - self.emission_error
        return cpt

class HmmParameterGenerateorWithPeak:
    def __init__(self, cardinality, window_size, emission_error):
        self.cardinality = cardinality
        self.window_size = window_size
        self.emission_error = emission_error

    def generate_initial_cpt(self):
        parameter_size = int(math.pow(self.cardinality, self.window_size))
        return np.array([1.0/(parameter_size) for i in range(0, parameter_size)]).reshape([self.cardinality]*self.window_size) # uniform

    def generate_peak_distribution(self, peak):
        dist = [float(1-peak)/(self.cardinality-1)] * self.cardinality
        dist[0] = peak
        return np.array(dist)

    def generate_transition_cpt(self):
        num_parent_states = int(math.pow(self.cardinality, self.window_size))
        cpt = np.zeros([self.cardinality]*(self.window_size + 1)).reshape([num_parent_states, self.cardinality])
        for i in range(0, num_parent_states):
            cur_peak_dist = self.generate_peak_distribution(float(i)/(num_parent_states-1))
            cpt[i] = cur_peak_dist
        return cpt.reshape([self.cardinality] * (self.window_size+1))

    def generate_emission_cpt(self):
        cpt = np.zeros([self.cardinality, 2])
        for i in range(0, self.cardinality):
            # sensing whether the hidden state id is even or odd
            if i % 2 == 0:
                cpt[i][0] = 1 - self.emission_error
                cpt[i][1] = self.emission_error
            else:
                cpt[i][0] = self.emission_error
                cpt[i][1] = 1 - self.emission_error
        return cpt


class Hmm:
    def __init__(self, chain_size, window_size, cardinality, parameter_generator):
        assert window_size == parameter_generator.window_size
        assert cardinality == parameter_generator.cardinality
        self.initial_pt = parameter_generator.generate_initial_cpt()
        self.transition_cpt = parameter_generator.generate_transition_cpt()
        self.emission_cpt = parameter_generator.generate_emission_cpt()
        self.chain_size = chain_size
        self.window_size = window_size
        self.cardinality = cardinality
        logging.info("hmm's initial pt: \n{0}".format(self.initial_pt))
        logging.info("hmm's transition cpt: \n{0}".format(self.transition_cpt))
        logging.info("hmm's emission cpt: \n{0}".format(self.emission_cpt))


    def sample(self, sim_rand_gen):
        # simulate Hs
        simulated_h_start = sample_from_distribution(self.initial_pt.reshape([int(math.pow(self.cardinality, self.window_size))]), sim_rand_gen)
        h_sampled = [int (a) for a in np.unravel_index(simulated_h_start, [self.cardinality] * self.window_size)]
        for j in range(self.window_size, self.chain_size):
            parent_conf = tuple(h_sampled[-self.window_size : ])
            local_cpt = self.transition_cpt[parent_conf]
            h_sampled.append(sample_from_distribution(local_cpt, sim_rand_gen))
        # simulate emissions
        e_sampled = []
        for j in range(0, self.chain_size):
            local_cpt = self.emission_cpt[h_sampled[j]]
            e_sampled.append(sample_from_distribution(local_cpt, sim_rand_gen))
        return h_sampled, e_sampled

    # Evidence 0 stands for 0% in state0 and 1 stands for 100% in state0
    def inference(self, evidence):
        # generate soft evidence on hidden states
        h_softs = []
        for obs in evidence:
            h_softs.append([a[0]* obs + a[1] * (1-obs) for a in self.emission_cpt])
        # Adding priors to prior hs
        message = self.initial_pt.copy()
        for i in range(0, message.size):
            cur_conf = np.unravel_index(i, [self.cardinality] * self.window_size)
            for j in range(0, self.window_size):
                message[cur_conf] *= h_softs[j][cur_conf[j]]
        # sum out h_s
        size_after_multiply = message.size * self.cardinality
        for i in range(self.window_size, self.chain_size):
            new_message = np.zeros(int(size_after_multiply))
            for j in range(0, size_after_multiply):
                cur_conf = np.unravel_index(j, [self.cardinality] * (self.window_size + 1))
                cur_soft_evid = h_softs[i][cur_conf[-1]]
                new_message[j] = message[cur_conf[:-1]] * self.transition_cpt[cur_conf] * cur_soft_evid
            new_message = new_message.reshape([self.cardinality] * (self.window_size + 1))
            message = np.sum(new_message, axis=0)
        for i in range(0, self.window_size-1):
            message = np.sum(message, axis = 0)
        query_result = message.reshape(self.cardinality)
        query_partition = np.sum(query_result)
        return [a/query_partition for a in query_result]

    def generate_dataset(self, num_examples, missing_pr,missing_mode, seed):
        r_generator = random.Random(seed)
        records = []
        for i in range(0, num_examples):
            cur_h, cur_e = self.sample(r_generator)
            cur_evid = []
            for j in range(0, self.chain_size):
                cur_rand = r_generator.random()
                if missing_mode == "MCAR":
                    if (cur_rand < missing_pr):
                        # missing
                        cur_evid.append(0.5)
                    else:
                        cur_evid.append(float(1-cur_e[j]))
                else:
                    # recent
                    if j >= (1-missing_pr) * self.chain_size:
                        cur_evid.append(0.5)
                    else:
                        cur_evid.append(float(1-cur_e[j]))
            pr = self.inference(cur_evid)
            cur_record = {}
            cur_record["input"] = cur_evid
            cur_record["label"] = cur_h[-1]
            cur_record["value"] = pr
            records.append(cur_record)
        return records

hmm_generator = HmmParameterGenerateorWithPeak(3,2,0.2)

"""
def generate_random_distribution(num_state):
    dist = []
    for i in range(0, num_state):
        dist.append(random.random())
    dist_partition = sum(dist)
    return [a/dist_partition for a in dist]

def simulate_state(distribution):
    acc = 0
    cur_rand = random.random()
    for i in range(0, len(distribution)):
        acc += distribution[i]
        if cur_rand < acc:
            return i
    return len(distribution) -1;

class JumpingHmm:
    def __init__(self, config):
        self.config = config
        self.prior_cpt = None
        self.emission_cpt = None
        self.chain_length = config["chain_length"]

    def Inference(self, evidence):
        factor_network = FactorGraph();
        for i in range(0, self.chain_length):
            factor_network.add_node("H%s"%i)
            factor_network.add_node("E%s"%i)
        # add cpt
        for i in range(0,self.chain_length):
            cur_factor = DiscreteFactor(["H%s"%i, "E%s"%i], cardinality=[2,2], values= self.emission_cpt[0]+self.emission_cpt[1])
            factor_network.add_factors(cur_factor)
            factor_network.add_edge("H%s"%i, cur_factor)
        cur_factor = DiscreteFactor(["H0", "H1"], cardinality=[2,2], values= self.prior_cpt)
        factor_network.add_factors(cur_factor)
        factor_network.add_edge("H0", cur_factor)
        factor_network.add_edge("H1", cur_factor)
        for i in range(2, self.chain_length):
            cur_factor = DiscreteFactor(["H%s"%(i-2), "H%s"%(i-1), "H%s"%i], cardinality=[2,2,2], values= [0,1,0,1,1,0,1,0])
            factor_network.add_factors(cur_factor)
            factor_network.add_edge("H%s"%(i-2), cur_factor)
            factor_network.add_edge("H%s"%(i-1), cur_factor)
            factor_network.add_edge("H%s"%i, cur_factor)
        for i in range(0, self.chain_length):
            cur_factor = DiscreteFactor(["E%s" % i], cardinality=[2], values= [evidence[i], 1-evidence[i]])
            factor_network.add_factors(cur_factor)
            factor_network.add_edge("E%s" % i, cur_factor)
        inference = VariableElimination(factor_network)
        q_variable = "E%s" % (self.chain_length-1)
        res = inference.query([q_variable])
        print (res.value)
        return None
    def SampleParameters(self, seed):
        random.seed(seed)
        self.prior_cpt = generate_random_distribution(4)
        self.emission_cpt = [generate_random_distribution(2), generate_random_distribution(2)]

    def SampleExample(self, number_examples, seed):
        random.seed(seed)
        examples = []
        for i in range(0, number_examples):
            hs = []
            es = []
            state_id = simulate_state(self.prior_cpt)
            hs.append(state_id >> 1)
            hs.append(state_id % 2)
            for i in range(2, self.chain_length):
                if hs[-1] == hs[-2]:
                    hs.append(1 - hs[-1]);
                else:
                    hs.append(hs[-1])
            for i in range(0, self.chain_length):
                cur_emission_distribution = self.emission_cpt[hs[i]]
                es.append(simulate_state(cur_emission_distribution))
            examples.append({"H":hs, "E":es})
        return examples

    def GenerateDatasetForEval(self, number_examples, seed):
        missing_pr = self.config["missing_pr"]
        sampled_example = self.SampleExample(number_examples, seed)
        random.seed(seed) # seed for missing
        for i in range(0, number_examples):
            cur_example = sampled_example[i]
            cur_evidence = cur_example["E"]
            for j in range(0, self.chain_length):
                if random.random() < missing_pr:
                    # missing
                    cur_evidence[j] = 0.5
                else:
                    cur_evidence[j] = float(1 - cur_evidence[j])
            # Inference

"""

from TBNCompiler.ArithmeticCircuit import *
import numpy as np
from TBNCompiler.TestingBayesianNetwork import *
from TBNCompiler.utils import *
from TBNCompiler.compile_ve import compileTAC_ve
from hmm import *

def file2tac(fname, trained_fname, useCompiler='jason'):
    flmap = open(trained_fname, mode='r')
    ftac = open(fname, mode='r')

    nodes = {}
    literals = {}
    inputs = []
    # initialize literals from lmap file
    for line in flmap:
        line = line.rstrip().split(' ', maxsplit=3)
        if line[1] == 'i':
            assert(len(line) == 3)
            id = int(line[0])
            literals[id] = IndicatorNode(label=line[-1].split('=')[0], inst=line[-1].split('=')[1])
            inputs.append(literals[id])
        elif line[1] == 'p':
            assert(len(line) == 4)
            id = int(line[0])
            literals[id] = ParameterNode(value=float(line[2]), label=line[-1])
        else:
            raise RunTimeError('Invalid literals.')
    # initialize tac in topo order
    id = -1
    for i, line in enumerate(ftac):
        line = line.rstrip().split(' ')
        id = int(line[0])
        # literals
        if line[1] == 'L':
            assert(len(line) == 3)
            nodes[id] = literals[int(line[-1])]
        # Add, Multiply, Normalize Nodes
        elif line[1] == '+' or line[1] == '*' or line[1] == 'Z':
            assert(int(line[2]) == len(line[3:]))
            children = [nodes[int(child)] for child in line[3:]]
            if line[1] == '+':
                nodes[id] = AddNode(children=children)
            elif line [1] == '*':
                nodes[id] = MultiplyNode(children=children)
            elif line[1] == 'Z':
                nodes[id] = NormalizeNode(children=children)
        # Testing Nodes
        elif line[1] == '?':
            assert(int(line[2]) == len(line[3:]))
            children = [nodes[int(child)] for child in line[3:]]
            if len(children) == 5: # if gamma is not given
                children.append(None)
            if useCompiler == 'jason':
                fields = ['x', 'norm', 'thetaP', 'thetaN', 'thres', 'gamma']
            else:
                fields = ['x', 'norm', 'thres', 'thetaN', 'thetaP', 'gamma']
            args = {key:value for (key, value) in zip(fields, children)}
            nodes[id] = TestingNode(**args)
        else:
            raise RunTimeError('Invalid type of tac nodes.')
    
    for (key, node) in nodes.items():
        node.id = key
    # build the tac
    root = nodes[id]
    #print(root)
    input_dict = {}
    for ind in inputs:
        if ind.label not in input_dict:
            input_dict[ind.label] = [ind]
        else:
            input_dict[ind.label].append(ind)

    # sort inputs of the same variable according to the state number
    for (key, value) in input_dict.items():
        input_dict[key] = sorted(value, key=lambda x: int(x.inst), reverse=False)

    inputs = []
    for l in input_dict.values():
        inputs += l

    flmap.close()
    ftac.close()
    return ArithmeticCircuit.ArithmeticCircuit(root=root, inputs=inputs)

def validate(ac_fname, trained_ac_fname,tac_fname, trained_tac_fname, pred_fname, useCompiler='jason'):
    print('Start validation...')
    hmm_ac = file2tac(ac_fname, trained_ac_fname, useCompiler=useCompiler)
    thmm_tac = file2tac(tac_fname, trained_tac_fname, useCompiler=useCompiler)

    # read in the prediction data
    pred = open(pred_fname, mode='r')
    report=[]
    for (i, line) in enumerate(pred):
        if i == 0: 
            continue #skip the first column
        line = line.rstrip()
        evidence = line[line.find('[')+1: line.find(']')].split(', ')
        args = {}
        for (j, ev) in enumerate(evidence):
            args['E' + str(j)] = [float(ev), 1 - float(ev)]
        query = line[line.find(']')+1:].split()
        #print(i, query)
        assert(len(query) == 3)
        bn_pred = float(query[0])
        hmm_pred = float(query[1])
        thmm_pred = float(query[2])
        # predict the query again by hmm and thmm
        hmm_pred_re = hmm_ac.forward(**args)
        thmm_pred_re = thmm_tac.forward(**args)
        if not np.isclose(hmm_pred, hmm_pred_re, atol=1e-6, rtol=0):
            msg = f'dubious sample {i}: bn={bn_pred} ac is {hmm_pred} but should be {hmm_pred_re} in validation.'
            report.append(msg)
        if not np.isclose(thmm_pred, thmm_pred_re, atol=1E-6, rtol=0):
            msg = f'dubious sample {i}: bn={bn_pred} tac is {thmm_pred} but should be {thmm_pred_re} in validation.'
            report.append(msg)
    
    print('Find %d problems in validation' %len(report))
    for msg in report:
        print(msg)

def gen_BN_from_HMM(true_model):
    chain_size = true_model.chain_size
    win_size = true_model.window_size
    num_hidden_states = true_model.cardinality
    # creat a table for the intial pt
    initial_CPTable = CPTable()
    initial_CPTable.varName = ['H' + str(i) for i in range(win_size)]
    initial_CPTable.varNum = [num_hidden_states] * win_size
    initial_CPTable.table = true_model.initial_pt
    nodes = {}
    # create the roots
    for i in range(win_size):
        name = 'H' + str(i)
        node = TBNNode()
        node.name = name
        node.states = ['state' + str(j) for j in range(num_hidden_states)]
        node.parents = ['H' + str(j) for j in range(i)]
        node.children = ['H' + str(j) for j in range(i + 1, max(i + win_size + 1, chain_size))]
        joint_prob = factorProject(initial_CPTable, node.parents + [name]) # compute P(E0, E1, ..., Ei)
        if i == 0:
            cond_prob = joint_prob
        else:
            norm_prob = factorProject(initial_CPTable, node.parents) #compute P(E0, E1,..., Ei-2, Ei-1)
            cond_prob = CPTable()
            cond_prob.varName = [name] + node.parents
            cond_prob.varNum = [num_hidden_states] * len(cond_prob.varName)
            # find position of norm_prob variable in joint probability table
            varMap = {n: joint_prob.varName.index(n) for n in norm_prob.varName}
            for (pos, theta) in np.ndenumerate(joint_prob.table):
                pos_norm = []
                try:
                    for n in norm_prob.varName:
                        pos_norm.append(pos[varMap[n]])
                except IndexError:
                    print(f'pos: {pos}')
                    print('num: %d' % varMap[n])
                    print('prob: %s' % norm_prob.varName)
                    print('joint: %s' % joint_prob.varName)
                    exit(0)
                joint_prob.table[pos] = theta / norm_prob.table[tuple(pos_norm)]
                child_idx = joint_prob.varName.index(name)
                cond_prob.table = np.moveaxis(joint_prob.table, child_idx, 0)
                # obtain the conditional probability from the joint probability
                
        node.table = cond_prob
        nodes[name] = node
    # create the hidden nodes
    for i in range(win_size, chain_size):
        name = 'H' + str(i)
        node = TBNNode()
        node.name = name
        node.states = ['state' + str(j) for j in range(num_hidden_states)]
        node.parents = ['H' + str(j) for j in range(i - win_size, i)]
        node.children = ['H' + str(j) for j in range(i + 1, max(i + win_size + 1, chain_size))]
        node.table = CPTable()
        node.table.varName = [name] + node.parents
        node.table.varNum = [num_hidden_states] * (win_size + 1)
        node.table.table = np.moveaxis(true_model.transition_cpt, -1, 0)
        nodes[name] = node

    # create the evidence
    for j in range(chain_size):
        name = 'E' + str(j)
        node = TBNNode()
        node.name = name
        node.states = ['state0', 'state1']
        node.parents = ['H' + str(j)]
        node.chilren = []
        node.table = CPTable()
        node.table.varName = [name] + node.parents
        node.table.varNum = [2, num_hidden_states]
        node.table.table = np.moveaxis(true_model.emission_cpt, -1, 0)
        nodes[name] = node

    bn = TestingBayesianNetwork()
    bn.nodes = nodes
    #printTBN(bn)
    #query = 'H' + str(true_model.chain_size - 1)
    #evidence = ['E' + str(i) for i in range(true_model.chain_size)]
    #sim_ac = compileTAC_ve(bn, query, evidence, inst='state0', normalized=True)
    #query_value = sim_ac.forward(E0=[1.0,0.0],E1=[1.0,0.0],E2=[0.5,0.5],E3=[0.5,0.5],E4=[0.0,1.0],E5=[0.5,0.5],E6=[0.0,1.0],E7=[0.0,1.0])
    #print(query_value)
    return bn
    
    
def validateHMM(ac_fname, trained_ac_fname, tac_fname, trained_tac_fname, pred_fname, true_model, useCompiler='haiying'):
    print('Start validation...')
    hmm_ac = file2tac(ac_fname, trained_ac_fname, useCompiler=useCompiler)
    thmm_tac = file2tac(tac_fname, trained_tac_fname, useCompiler=useCompiler)
    sim_bn = gen_BN_from_HMM(true_model)
    query = 'H' + str(true_model.chain_size - 1)
    evidence = ['E' + str(i) for i in range(true_model.chain_size)]
    sim_ac = compileTAC_ve(sim_bn, query, evidence, normalized=True)
    # read in the prediction data
    pred = open(pred_fname, mode='r')
    report=[]
    for (i, line) in enumerate(pred):
        if i == 0: 
            continue #skip the first column
        line = line.rstrip()
        evidence = line[line.find('[')+1: line.find(']')].split(', ')
        args = {}
        for (j, ev) in enumerate(evidence):
            args['E' + str(j)] = [float(ev), 1 - float(ev)]
        query = line[line.find(']')+1:].split()
        #print(i, query)
        assert(len(query) == 3)
        bn_pred = float(query[0])
        hmm_pred = float(query[1])
        thmm_pred = float(query[2])
        # predict the query again by hmm and thmm
        sim_pred_re = sim_ac.forward(**args)
        hmm_pred_re = hmm_ac.forward(**args)
        thmm_pred_re = thmm_tac.forward(**args)
        if not np.isclose(bn_pred, sim_pred_re, atol=1e-6, rtol=0):
            msg = f'dubious sample {i}: bn is {bn_pred} but should be {sim_pred_re} in validation'
            report.append(msg)
        if not np.isclose(hmm_pred, hmm_pred_re, atol=1e-6, rtol=0):
            msg = f'dubious sample {i}: bn={bn_pred} ac is {hmm_pred} but should be {hmm_pred_re} in validation.'
            #report.append(msg)
        if not np.isclose(thmm_pred, thmm_pred_re, atol=1e-6, rtol=0):
            msg = f'dubious sample {i}: bn={bn_pred} tac is {thmm_pred} but should be {thmm_pred_re} in validation.'
            #report.append(msg)
    
    print('Find %d problems in validation' %len(report))
    for msg in report:
        print(msg)
    























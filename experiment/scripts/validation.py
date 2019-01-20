from ArithmeticCircuit import *
import numpy as np

def file2tac(fname, trained_fname):
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
            fields = ['x', 'norm', 'thetaP', 'thetaN', 'thres', 'gamma']
            args = {key:value for (key, value) in zip(fields, children)}
            nodes[id] = TestingNode(**args)
        else:
            raise RunTimeError('Invalid type of tac nodes.')
    
    for (key, node) in nodes.items():
        node.id = key
    # build the tac
    root = nodes[id]
    #print(root)
    flmap.close()
    ftac.close()
    return ArithmeticCircuit(root=root, inputs=inputs)

def validate(ac_fname, trained_ac_fname,tac_fname, trained_tac_fname, pred_fname):
    print('Start validation...')
    hmm_ac = file2tac(ac_fname, trained_ac_fname)
    thmm_tac = file2tac(tac_fname, trained_tac_fname)
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
            args['E' + str(j)] = [1 - float(ev), float(ev)]
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
            msg = f'dubious sample {i}: bn={bn_pred} tac is {hmm_pred} but should be {hmm_pred_re} in validation.'
            report.append(msg)
    
    print('Find %d problems in validation' %len(report))
    for msg in report:
        print(msg)










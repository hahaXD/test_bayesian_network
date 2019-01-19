#!/usr/bin/env python

import numpy as np

def generate_grid(dim):
    eps = 1./dim
    X = np.arange(0, 1+eps, eps)
    Y = np.arange(0, 1+eps, eps)
    X,Y = np.meshgrid(X,Y)
    n = dim+1
    flat_X = X.reshape((n*n,))
    flat_Y = Y.reshape((n*n,))
    return X,Y,list(zip(flat_X,flat_Y))

def project_dataset(header,examples,evidence_variables):
    indices = [ header.index(var) for var in evidence_variables ]
    evidence = [ [example[i] for i in indices] for example in examples ]
    return evidence

def binary_dataset_to_tac_inputs(header,data,tac):
    """header is a list of evidence variables (strings of variable names).
    data is a list of tuple's, where each value of the tuple is a soft
    evidence value p of a binary variable.  The specified p is assumed
    to be asserted on the positive value of a variable (state index
    1), and the value 1-p is asserted on the negative value of a
    variable (state inddex 0).

    The output is a tensorflow input vector.
    """
    lmap,imap = tac.lmap,tac.imap
    indicator_count = sum(len(imap[var]) for var in imap)
    parameter_count = len(lmap)-indicator_count

    inst_to_index = {}
    for var in imap:
        for val in imap[var]:
            lit = imap[var][val]
            inst_to_index[lit.st] = lit.index

    tac_inputs = []
    for example in data:
        tac_input = [None] * indicator_count
        for var,val in zip(header,example):
            lit_name = "%s=0" % var
            index = inst_to_index[lit_name]
            tac_input[index] = val

            lit_name = "%s=1" % var
            index = inst_to_index[lit_name]
            tac_input[index] = 1.0-val
        tac_inputs.append(tac_input)
    return tac_inputs

def read_csv(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    examples = []
    labels = []
    for line in lines[1:]:
        line = line.strip().split(',')
        examples.append([ float(x) for x in line[:-1] ])
        label = float(line[-1])
        labels.append(label)
    return header,examples,labels

def save_csv(header,examples,labels,filename):
    with open(filename,'w') as f:
        f.write(",".join(header))
        f.write("\n")
        for example,label in zip(examples,labels):
            example = [ "%.8g" % x for x in example ]
            f.write("%s,%.8g\n" % (",".join(example),label))

def parse_hugin_states(filename):
    import re
    with open(filename,'r') as f:
        lines = f.readlines()
    domain = dict()
    current_variable = None
    for line in lines:
        line = line.strip()
        if line.startswith("node"):
            current_variable = line.split(' ')[-1]
        elif line.startswith("states"):
            line = re.sub(r'\s+',' ',line)
            _,line = line.split('=')
            line = line.strip(" ();")
            states = line.split(' ')
            states = [ state.strip('"') for state in states ]
            domain[current_variable] = states
    return domain

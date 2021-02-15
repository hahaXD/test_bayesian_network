#!/usr/bin/env python

import math
import random
import numpy as np

cname = "H" # prefix of hidden nodes
ename = "E" # prefix of evidence nodes
mname = "M" # prefix of map nodes

def cnode(index):
    return "%s%s" % (cname,index)

def enode(index):
    return "%s%s" % (ename,index)

def mnode(index):
    return "%s%s" % (mname,index)

def state(index):
    return "s%s" % index

def random_parameter_set(k,alpha=0.5,beta=1.0):
    """Draws a parameter set from a Dirichlet distribution.
    k is the number of states
    psi is the Dirichlet meta-parameter"""
    pr = [ random.gammavariate(alpha,beta) for i in range(k) ]
    pr_sum = sum(pr)
    return [ p/pr_sum for p in pr ]

# cardinality is number of states for each node
def write_nodes(nodes,cardinality,f):
    for node in nodes:
        f.write('<VARIABLE TYPE="nature">\n')
        f.write("  <NAME>%s</NAME>\n" % node)
        for i in range(cardinality):
            f.write("  <OUTCOME>%s</OUTCOME>\n" % state(i))
        f.write("</VARIABLE>\n\n")

# each map var is a root
def write_map_cpts(n,cardinality,map_vars,f):
    for var in map_vars:
        f.write("<DEFINITION>\n")
        f.write("  <FOR>%s</FOR>\n" % var)
        f.write("  <TABLE>")
        for p in random_parameter_set(cardinality): # distribution for map var
            f.write("%.6f " % p)
        f.write("</TABLE>\n")
        f.write("</DEFINITION>\n\n")

# each evidence var has a single hidden var as parent
def write_evidence_cpts(n,cardinality,hidden_vars,evidence_vars,f):
    cpt = [] # 1D array, unique emissions cpt
    for i in range(cardinality):
        cpt.append(random_parameter_set(2)) # evidence vars are binary
    for i in range(n):
        f.write("<DEFINITION>\n")
        f.write("  <FOR>%s</FOR>\n" % evidence_vars[i])
        f.write("    <GIVEN>%s</GIVEN>\n" % hidden_vars[i])
        f.write("  <TABLE>")
        for dist in cpt: # distribution for parent state
            f.write("%.6f %.6f " % (dist[0], dist[1]))
        f.write("</TABLE>\n")
        f.write("</DEFINITION>\n\n")

def write_hidden_cpts(n,cardinality,window,hidden_vars,map_vars,f):
    map = len(map_vars) > 0
    if map == True:
        m = 1
    else:
        m = 0 
    cpts = [] # 2D array, transition CPTs
    for i in range(window+1+m): # i is number of parents
        cpt = []
        for j in range(cardinality**i): # j is state of parents
            cpt.extend(random_parameter_set(cardinality))
        cpts.append(cpt)
    for i in range(n):
        f.write("<DEFINITION>\n")
        f.write("  <FOR>%s</FOR>\n" % hidden_vars[i])
        if map == True:
            f.write("    <GIVEN>%s</GIVEN>\n" % map_vars[i])
        hidden_parent_count = min(i,window) # without map variable
        for j in range(hidden_parent_count): # write parents
            k = i-j-1 # parent index
            assert k >= 0
            f.write("    <GIVEN>%s</GIVEN>\n" % hidden_vars[k])
        f.write("  <TABLE>")
        assert hidden_parent_count >=0 and hidden_parent_count <= window
        for p in cpts[hidden_parent_count+m]:
            f.write("%.6f " % p)
        f.write("</TABLE>\n")
        f.write("</DEFINITION>\n\n")

def write_file(type,n,cardinality,window,hidden_vars,evidence_vars,map_vars,f):
    # write header
    f.write('<BIF VERSION="0.3">\n')
    f.write("<NETWORK>\n")
    f.write("  <NAME>%s</NAME>\n\n" % type)
    # write nodes
    write_nodes(hidden_vars,cardinality,f)
    write_nodes(evidence_vars,2,f)
    write_nodes(map_vars,cardinality,f)
    # write CPTs
    write_map_cpts(n,cardinality,map_vars,f)
    write_evidence_cpts(n,cardinality,hidden_vars,evidence_vars,f)
    write_hidden_cpts(n,cardinality,window,hidden_vars,map_vars,f)
    # write footer
    f.write("</NETWORK>\n")
    f.write("</BIF>\n")

# n: length of hmm
# cardinality: number of states for hidden variables (evidence is binary)
# window: number of parents per node (window=1 is traditional hmm) 
# use_map_variable: filename to store the map_vars
# hmm_fname: filename to store the result hmm
def generate_hmm(n, cardinality, window, use_map_variable, hmm_fname):
    assert n >= 2
    assert cardinality >= 2
    assert window >= 1
    # variables
    hidden_vars   = []
    evidence_vars = []
    map_vars      = []
    for i in range(n):
        hidden_vars.append(cnode(i))
        evidence_vars.append(enode(i))
        if use_map_variable:
            map_vars.append(mnode(i))
    # write hmm and thmm to files
    with open(hmm_fname,'w') as f:
        write_file("hmm", n, cardinality, window, hidden_vars, evidence_vars, map_vars, f)
    # return
    query_var     = cnode(n-1)
    testing_vars  = hidden_vars[1:]
    return hmm_fname, evidence_vars, query_var, testing_vars, map_vars

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 5:
        print ("Usage: <n> <cardinality> <window> <use_map_variable> <hmm_fname>")
        sys.exit(1)
    n = int(sys.argv[1])
    cardinality = int(sys.argv[2])
    window = int(sys.argv[3])
    use_map_variable = bool(sys.argv[4])
    hmm_fname = sys.argv[5]
    res = generate_hmm(n, cardinality, window, use_map_variable, hmm_fname)
    print ("query_var : %s" % res[2])

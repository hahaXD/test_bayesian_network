from .TestingBayesianNetwork import *
from .ArithmeticCircuit import *
from graphviz import Digraph
from typing import List, Set, Dict, Tuple, Optional
from .utils import printTBN
from . import config

'''
import sys
import il2.inf.structure.EliminationOrders as EO
import il2.model.Index as Index
import il2.model.Domain as Domain
import il2.util.IntSet as IntSet
'''

# set CPT selection settings

# config.useSigmoid = True
# config.useGlobalGamma = False
# config.useTotalOrdering = True
# config.gamma = 8
gammaNode = None
# config.testing_order = None

# path to the java samiam library
from os import path
dir_path = path.relpath(path.dirname(path.abspath(__file__)))
inflib_path = path.normpath(path.join(dir_path, '..','lib', 'inflib.jar'))
tmp_path = path.normpath(path.join(dir_path, '..', 'tmp'))
pathToLibrary = dir_path + ':' + inflib_path
#print(pathToLibrary)

from queue import PriorityQueue

# Return a topological sort ordering of all variables in the network
def topoSort(tbn, sorted=False):
    InDeg = {n:len(nNode.parents) for (n, nNode) in tbn.nodes.items()}
    topo = []
    if sorted:
        # visit nodes parents before children
        # pop nodes with most evidence first
        # import pdb; pdb.set_trace()
        roots = PriorityQueue(maxsize=-1)
        for n in tbn.nodes:
            if InDeg[n] == 0:
                roots.put(tbn.nodes[n])
        while not roots.empty():
            nNode = roots.get()
            topo.append(nNode.name)
            for c in nNode.children:
                if c in tbn.nodes:
                # Be careful that after pruning some children might not exist
                    InDeg[c] -= 1
                    if InDeg[c] == 0:
                        roots.put(tbn.nodes[c])

    else:
        roots = [n for n in tbn.nodes if InDeg[n] == 0]
        # visit nodes parents before children
        while roots:
            n = roots.pop()
            topo.append(n)
            for c in tbn.nodes[n].children:
                if c in tbn.nodes: # Be careful that after pruning some children might not exist
                    InDeg[c] -= 1
                    if InDeg[c] == 0:
                        roots = [c] + roots
    return topo

# Keep removing leaf nodes that does not satisfy the condition
def pruneTBN(tbn, targets):
    OutDeg = {n:len(nNode.children) for (n, nNode) in tbn.nodes.items()}
    marked = set()
    seq = topoSort(tbn)[::-1] 
    # visit nodes bottom-up
    # if a leaf is not included in targets, remove it and update the OutDegree of its parents
    for n in seq:
        if OutDeg[n] == 0 and n not in targets:
            marked.add(n) # delete n later
            for p in tbn.nodes[n].parents:
                OutDeg[p] -= 1
    # remove all marked nodes from original tbn
    tbn_pruned = TestingBayesianNetwork()
    tbn_pruned.nodes = {n:copy(nNode) for (n, nNode) in tbn.nodes.items() if n not in marked}
    tbn_pruned.regularize()
    return tbn_pruned

# traverse the tbn and set for each node the evidences at or below it
def setEvidenceBelow(tbn, evidence):
    seq = topoSort(tbn)[::-1]
    # visit nodes bottom up
    for n in seq:
        nNode = tbn.nodes[n]
        l = set()
        if n in evidence:
            l.add(n)
        l = l.union(*[tbn.nodes[c].evidence for c in nNode.children])
        # collect the evidence from children
        nNode.evidence = l



# a helper function that assigns label to a parameter node
def getLabel(nNode, pos):
    l = ''
    for (i, k) in enumerate(pos):
        var = nNode.table.varName[i]
        l = l + var + '=' + str(k) + ' '
        if i == 0 and len(pos) > 1:
            l += '\| '
    return l

# a helper function that assigns label to a testing node
def getTestingLabel(nNode, pos):
    l = ''
    for (i, k) in enumerate(pos):
        var = nNode.table.varName[i]
        if i == 0:
            l = l + var + '\| '
        else:
            l = l + var + '=' + str(k) + ' '
    return l

def initTBN(tbn, query, evidence, hashTable=None):
    global gamma, gammaNode
    inputs = []
    for e in evidence:
        for k in range(len(tbn.nodes[e].states)):
            ind = uniqueNodeFactory(hashTable, IndicatorNode, label=e, inst=str(k), value=k) ##ACACACC
            inputs.append(ind)

    for (n, nNode) in tbn.nodes.items():
        symbolTable = np.zeros(nNode.table.table.shape, dtype=GeneralNode) 
        # construct a CPT factor
        if not nNode.testing:
            for (pos, theta) in np.ndenumerate(nNode.table.table):
                label = getLabel(nNode, pos)
                p = uniqueNodeFactory(hashTable, ParameterNode, value=theta, label=label)
                if n in evidence:
                    # find the corresponding indicator
                    ind = uniqueNodeLookup(hashTable, IndicatorNode, label=n, inst=str(pos[0]))
                    # multiply the indicator into parameter
                    symbolTable[pos] = uniqueNodeFactory(hashTable, MultiplyNode, children = [p, ind])
                else:
                    symbolTable[pos] = p
        # for testing node
        else:
            thresDict = {}
            gammaDict = {}
            for (pos, thetaP) in np.ndenumerate(nNode.tableP.table):
                label = getLabel(nNode, pos)
                labelT = getTestingLabel(nNode, pos)
                # initialize a threshold for each parent state
                if pos[1:] not in thresDict:
                    thresDict[pos[1:]] = uniqueNodeFactory(hashTable, ParameterNode, value=nNode.thres[pos[1:]], label=labelT + 'thres')
                t = thresDict[pos[1:]]
                # if we use sigmoid instead of comparison, initialize a gamma for each testing node
                if config.useSigmoid:
                    if not config.useGlobalGamma:
                        if pos[1:] not in gammaDict:
                            gammaDict[pos[1:]] = uniqueNodeFactory(hashTable, ParameterNode, value=config.gamma, label=labelT + 'gamma')
                        g = gammaDict[pos[1:]]
                    else:
                        g = gammaNode
                pthetaP = uniqueNodeFactory(hashTable, ParameterNode, value=thetaP, label=label + '+')
                pthetaN = uniqueNodeFactory(hashTable, ParameterNode, value=nNode.tableN.table[pos], label=label + '-')
                # multiplying evidence
                if n in evidence:
                    # find the corresponding indicator
                    ind = uniqueNodeLookup(hashTable, IndicatorNode, label=n, inst=str(pos[0]))
                    pthetaP = uniqueNodeFactory(hashTable, MultiplyNode, children=[ind, pthetaP])
                    pthetaN = uniqueNodeFactory(hashTable, MultiplyNode, children=[ind, pthetaN])
                # Create a testing Node for this row. We will flatten it later
                symbolTable[pos] = TestingNode(x=None, norm=None, thetaP=pthetaP, thetaN=pthetaN, thres=t, gamma=g if config.useSigmoid else None, alive=False, label=label)
        nNode.table.table = symbolTable # replace the CPT table with symbolic factor
    return inputs

def ignoreEvidence(tbn, X, evidence, seq=None):
    queue = [p for p in tbn.nodes[X].parents]
    # Becareful that we need to copy the parent list 
    if not config.useTotalOrdering:
        visited = set()
        # first pass: visit ancestors of X, color testing nodes blue
        blue = set() 
        while queue:
            n = queue.pop()
            visited.add(n)
            if tbn.nodes[n].testing:
                blue.add(n)
            for p in tbn.nodes[n].parents:
                if p not in visited:
                    queue = [p] + queue
        # second pass: visit all nodes top-down, color testing nodes red if not colored blue
        # children of a red node are also colored red
    else:
        pos = seq.index(X)
        blue = set(seq[:pos])

    red = set()
    for n in topoSort(tbn): 
        flag = False
        if tbn.nodes[n].testing and n not in blue:
            flag = True
        for p in tbn.nodes[n].parents:
            if p in red:
                flag = True
        if flag:
            red.add(n)
    # ignore evidence on red nodes
    evidence_X = [ev for ev in evidence if ev not in red]
    '''
    with open('evidence.txt', mode='a+') as f:
        f.write(X + ': ' + ', '.join(evidence_X) + '\n')
    '''
    #print(f'node: {X}', f'evidence: {evidence_X}')
    return evidence_X


# get the domain and adjacency relationship of this tbn and
# write them to domain.txt and variables.txt
def writeSubdomains(tbn):
    global tmp_path
    factors = [nNode.table for nNode in tbn.nodes.values()]
    domain_path = path.join(tmp_path, 'domain.txt')
    with open(domain_path, mode='w') as f_domain:
        for (n, nNode) in tbn.nodes.items():
            #print(type(n))
            f_domain.write(f'{n},{len(nNode.states)}\n')
    variables_path = path.join(tmp_path, 'variables.txt')
    with open(variables_path, mode='w') as f_var:
        for f in factors:
            f_var.write(','.join(f.varName[::-1]) + '\n')

import subprocess
# find a minfill elimination order of tbn with queries at the last
def minfillSort(tbn, Q):
    global pathToLibrary
    writeSubdomains(tbn)
    # write the subdomains of this tbn to files
    subprocess.check_call(["java", "-classpath", pathToLibrary, "MinFillOrder"] + Q)
    # call a java function that outputs an eliminiation order for this tbn
    order_path = path.join(tmp_path, 'order.txt')
    f = open(order_path, mode='r')
    line = f.readline()
    if line == "":
        raise RunTimeError("No elimination order.")
    mf_order = line.rstrip().split(',')
    f.close()
    return mf_order

    # return mf_order
def variableElimination(tbn, Q, hashTable=None):
    factors = [nNode.table for nNode in tbn.nodes.values()]
    buckets = {}
    mf_order = minfillSort(tbn, Q)
    # print(f'variable order: {mf_order}')
    for n in mf_order:
        buckets[n] = []
    for f in factors:
        for n in mf_order:
            if n in f.varName:
                buckets[n].append(f)
                break
    for (i, var) in enumerate(mf_order):
        # eliminate all variables except the query
        if var in Q:
            break
        f_product = factorMultiply(buckets[var], hashTable)
        f_sum = factorSum(f_product, [var], hashTable)
        for n in mf_order[i+1:]:
            if n in f_sum.varName:
                buckets[n].append(f_sum)
                break
    f_priors = []
    # the remained factors represent the marginal P(Q)
    for query in Q:
        f_priors += buckets[query]
    return factorMultiply(f_priors, hashTable)

# Select a regular CPT for testing Node X
def select(tbn, X, evidence, seq=None, hashTable=None):
    Xnode = tbn.nodes[X]
    evidence_X = ignoreEvidence(tbn, X, evidence, seq=seq)
    tbn_X = pruneTBN(tbn, evidence_X + Xnode.parents)
    '''
    if X == 'C':
        print("the sub graph for c")
        printTBN(tbn_X)
    '''
    Prf = variableElimination(tbn_X, Xnode.parents, hashTable)
    # compile the sub tbn for marginal on parents of X
    PrNorm = factorProject(Prf, "", hashTable).table[0]
    # l = [node.item() for node in np.nditer(Prf.table, flags=['refs_ok'])]
    # PrNorm = uniqueNodeFactory(hashTable, AddNode, children=l)
    # create a node that represents the sum of all parent state values
    symbolTable = Xnode.table.table
    for (pos, tNode) in np.ndenumerate(symbolTable):
        pos_u = np.zeros(len(pos) - 1, dtype=int) 
        # find the index of parent state in Prf
        for (i, n) in enumerate(Xnode.table.varName):
            if i == 0: 
                continue # skip X itself
            try:
                k = Prf.varName.index(n)
                pos_u[k] = pos[i]
            except ValueError:
                raise RunTimeError("Inconsistent parent marginal")
        # select the node that represents the corresponding parent state
        PrU = Prf.table[tuple(pos_u)]
        # create a flattened testing Node for each state
        symbolTable[pos] = uniqueNodeFactory(hashTable, TestingNode, x=PrU, norm=PrNorm, 
            thres=tNode.thres, thetaP=tNode.thetaP, thetaN=tNode.thetaN, gamma=tNode.gamma, alive=False if not evidence_X else True, label=tNode.label)
    Xnode.table.table = symbolTable
    # replace the CPT for node X

def compileTAC_ve(tbn, query, evidence, inst=None, normalized=False):
    global gamma, gammaNode
    # first, prune leaves that is not query or evidence
    print('Compiler settings:')
    print(f'useSigmoid={config.useSigmoid} useGlobalGamma={config.useGlobalGamma} gamma={config.gamma} useTotalOrdering={config.useTotalOrdering}')
    print('Start compilation...')
    tbn = pruneTBN(tbn, [query] + evidence)
    # Second, initialize indicators, parameters, thresholds, and entering evidence
    setEvidenceBelow(tbn, evidence)
    # for each node, collect the evidence at or below it
    #printTBN(tbn)
    hashTable = {}
    if config.useSigmoid and config.useGlobalGamma:
        gammaNode = uniqueNodeFactory(hashTable, ParameterNode, value=config.gamma, label='global gamma')
    inputs = initTBN(tbn, query, evidence, hashTable)
    # Third, visit each testing nodes top down, compute the parent marginals, and select a regular CPT for it
    if not config.useTotalOrdering:
        # use arbitrary topo ordering
        seq = [n for n in topoSort(tbn) if tbn.nodes[n].testing]
    elif not config.testing_order:
        # use topo ordering that selects node of most evidence first
        seq = [n for n in topoSort(tbn, sorted=True) if tbn.nodes[n].testing]
    else:
        # use the given ordering
        seq = config.testing_order

    print('Testing Nodes Order: ', seq)
    for n in seq:
        select(tbn, n, evidence, seq=seq, hashTable=hashTable)
    # Now that we have obtained a regular bn, run variable elimination for query
    f_query = variableElimination(tbn, [query], hashTable) # do not forget the hashtable
    # in default, we query the first value assignment of query
    if inst is None:
        inst = 0
    elif type(inst) == int:
        pass
    elif type(inst) == str:
        inst = tbn.nodes[query].states.index(inst)
    root = f_query.getProbByInst(**{query:inst})
    # find the corresponding prob of query state
    if normalized:
        l = [node.item() for node in np.nditer(f_query.table, flags=['refs_ok']) if node.item() is not root]
        root = uniqueNodeFactory(hashTable, NormalizeNode, children=[root] + l)
        # Becareful that the query state must be the first children of Normalize node
    #genGraphNodes(hashTable)
    print('Compilation complete.')
    print(f'Nodes in hashtable: {len(hashTable)}')
    print('Start building tac...')
    ac = ArithmeticCircuit(root=root, inputs=inputs)
    print('Building tac complete.')
    ac.pruneNode()
    return ac


def genGraphNodes(hashTable):
    # visualize the arithmetic circuit 
    dot = Digraph(comment='Test')
    dot.attr(rankdir='BT')
    for node in hashTable.values():
        if type(node) == GeneralNode:
            raise NotImplementedError()
        elif type(node) == IndicatorNode:
            dot.node(str(node.id), str(node.id) + ' I: ' + node.label + '=' + str(node.inst), shape='record')
        elif type(node) == ParameterNode:
            dot.node(str(node.id), str(node.id) + ' P: ' + node.label, shape='record', color='lightgrey', style='filled')
        elif type(node) == MultiplyNode:
            dot.node(str(node.id), str(node.id) + ' *')
            for c in node.children:
                dot.edge(str(c.id), str(node.id))
        elif type(node) == AddNode:
            dot.node(str(node.id), str(node.id) + ' +')
            for c in node.children:
                dot.edge(str(c.id), str(node.id))
        elif type(node) == TestingNode:
            dot.node(str(node.id), str(node.id) + ' S: ' + node.label, shape='diamond', color='blue', style='filled')
            if not node.x is None:
                dot.edge(str(node.x.id), str(node.id))
            if not node.norm is None:
                dot.edge(str(node.norm.id), str(node.id))
            if not node.gamma is None:
                dot.edge(str(node.gamma.id), str(node.id))
            dot.edge(str(node.thres.id), str(node.id))
            dot.edge(str(node.thetaP.id), str(node.id))
            dot.edge(str(node.thetaN.id), str(node.id))
        elif type(node) == NormalizeNode:
            dot.node(str(node.id), str(node.id) + ' Normal', shape='record')
            for (i, c) in enumerate(node.children):
                if i == 0:
                    dot.edge(str(c.id), str(node.id), label='query')
                else:
                    dot.edge(str(c.id), str(node.id))

    dot.view('tmp/my test1')

    












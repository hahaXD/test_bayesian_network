# Generate Various TBN test cases
from .readTBNXML import readTBNXML
from .utils import *
from .TestingBayesianNetwork import *
from typing import List
from copy import deepcopy
from itertools import product
import random

# generate table of the corresponding shape with entries' values all between 0 and 1
def genProb(shape):
    random.seed()
    t = np.zeros(shape)
    for x in np.nditer(t, op_flags=['readwrite']):
        x[...] = random.random()
    return t

# generate a CPT table with specific column number
def genTable(columnNum,positive = False,odd = False):
    assert(columnNum>0)
    if columnNum == 1:
        x = random.random()
        return np.array([x,1-x])
    t = np.zeros([2]*columnNum)

    t[0] = np.ones(t.shape[1:])
    # if positive == (not odd):
    if positive:
    # if positive == (not odd):
    # if True:
        t[0] = t[0]*0.2
    #     t = np.array([[0.8,0.2],[0.2,0.8]])
    else:
        # t[0] = t[0]*0.1
        t[0] = t[0]*0.8
        # t = np.array([[0.2,0.8],[0.8,0.2]])

    t[0] += (genProb(t.shape[1:])*0.4 - 0.2)

    # t[0] = genProb(t.shape[1:])

    t[t>=1] = 0.99
    t[t<=0] = 0.01
    t[1] = 1 - t[0]
    # print(t)
    return t

# generate a CPT table of the given shape
def genTableOfShape(shape):
    assert(len(shape) > 1)
    t = np.zeros(shape)
    for i in range(shape[0]):
        t[i] = np.random.random(shape[1:]) 
    t_sum = np.sum(t, axis=0, keepdims=True)
    t = t / t_sum
    return t
    # the probability must be summed to one along the first axis



# convert a bn to tbn by makin specifed nodes to testing nodes,
# to make good comparison, complement corresponding entry in positive
# and negative table, uniform threshold for the time being
def bn2Tbn(bn:TestingBayesianNetwork, testingNodes:List[str], thres, randomTable=False):
    tbn = deepcopy(bn)
    for node in tbn.nodes.values():
        if node.name in testingNodes:
            node.testing = True
            node.tableP = node.table
            node.tableN = deepcopy(node.table)
            if randomTable:
                tableShape = node.tableP.table.shape
                node.tableP.table = genTableOfShape(tableShape)
                node.tableN.table = genTableOfShape(tableShape)
                #print(f'node {node.name}', f'table size: {node.tableP.table.shape}')
            else:
                # Just reversi the entry in the table
                for v in product(*([[0,1]]*len(node.table.varNum))):
                    if v[0] == 0:
                        tmp = node.tableN.table[tuple(v)]
                        node.tableN.table[tuple(v)] = 1 - tmp
                    else:
                        v2 = list(v)
                        v2[0] = 0
                        node.tableN.table[tuple(v)] = 1 - node.tableN.table[tuple(v2)]
            node.thres = np.zeros(node.tableP.table.shape[1:])
            node.thres.fill(thres)
            # node.thres.fill(thres)
    return tbn

# Add a child to the corresponding node, table is  a 2*2 numpy arrayj
def addChild(tbn,nodeName,table,parentNodeName):
    tbn = deepcopy(tbn)
    node = TBNNode()
    node.name = nodeName
    node.parents = [parentNodeName]
    tbn.nodes[nodeName] = node
    cpt = CPTable()
    cpt.varName = [nodeName,parentNodeName]
    cpt.varNum = [2,2]
    cpt.table = table
    node.table = cpt
    return tbn


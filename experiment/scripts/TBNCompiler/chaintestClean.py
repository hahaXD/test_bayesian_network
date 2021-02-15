from TestingBayesianNetwork import *
from readTBNXML import *
from readBNXML import *
from utils import genDiagram,file2tac,tac2file,findNode
from ArithmeticCircuit import *
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random



# generate table of the corresponding shape with entries' values all between 0 and 1
def genProb(shape):
    random.seed()
    t = np.zeros(shape)
    for x in np.nditer(t, op_flags=['readwrite']):
        x[...] = random.random()
    return t

# generate a CPT table with specific column numbers
def genTable(columnNum,positive = False,odd = False):
    assert(columnNum>0)
    if columnNum == 1:
        x = random.random()
        return np.array([x,1-x])
    t = np.zeros([2]*columnNum)

    t[0] = np.ones(t.shape[1:])
    # if positive == (not odd):
    # if positive:
    if positive == (not odd):
    # if True:
    #     t[0] = t[0]*0.1
        t = np.array([[0.8,0.2],[0.2,0.8]])
    else:
        # t[0] = t[0]*0.1
        # t[0] = t[0]*0.9
        t = np.array([[0.2,0.8],[0.8,0.2]])

    t[0] += (genProb(t.shape[1:])*0.4 - 0.2)

    # t[0] = genProb(t.shape[1:])

    t[t>1] = 1
    t[t<0] = 0
    t[1] = 1 - t[0]
    # print(t)
    return t

# generate TBN: X->T1->T2->..->T_n->Y where T_i is testing node
def genChain(testNum,test=True,thres=-1):
    tbn = readTBNXML('samples/chain.xml')

    tNode = deepcopy(tbn.nodes['T'])
    xNode = deepcopy(tbn.nodes['X'])
    yNode = deepcopy(tbn.nodes['Y'])

    tNode.tableP = deepcopy(tNode.table)
    tNode.tableN = deepcopy(tNode.table)
    tNode.thres = np.zeros(tNode.table.table.shape[1:])
    tNode.thres.fill(0.1)

    prevTNode = tNode
    nodes = {}
    nodes['X'] = xNode
    nodes['Y'] = yNode
    prevTNode.name = 'T0'
    prevTNode.testing = True
    prevTNode.parents = ['X']
    prevTNode.table.varName = ['T0','X']
    prevTNode.table.varNum = [2,2]


    prevTNode.tableP = prevTNode.table
    prevTNode.tableN = deepcopy(prevTNode.table)

    curTNode = deepcopy(tNode)
    nodes['T0'] = tNode

    thresShape = prevTNode.thres.shape
    tableShape = prevTNode.tableP.table.shape
    prevTNode.thres = genProb(thresShape)
    prevTNode.thres[1] = prevTNode.thres[0]
    #
    # prevTNode.thres[0] = 0.75
    # prevTNode.thres[1] = 0.75

    prevTNode.tableP.table = genTable(len(tableShape),positive=True,odd = False)
    prevTNode.tableN.table = genTable(len(tableShape),positive=False,odd = False)
    # prevTNode.tableP.table = genProb(tableShape)
    # prevTNode.tableN.table = genProb(tableShape)

    for i in range(1,testNum):
        curTNode.table = curTNode.tableP
        curTNode.parents = [prevTNode.name]

        curTNode.tableP.varName = ['T'+str(i),'T'+str(i-1)]
        curTNode.tableN.varName = ['T'+str(i),'T'+str(i-1)]


        curTNode.name = 'T'+str(i)
        nodes['T'+str(i)] = curTNode

        curTNode.thres= genProb(thresShape)

        # curTNode.thres[0] = 0.7
        curTNode.thres[1] = curTNode.thres[0]
        print(curTNode.thres)

        # if thres>=0:
        #     curTNode.thres[0] = thres
        #     curTNode.thres[1] = curTNode.thres[0]
            # curTNode.thres[1] = 1 - curTNode.thres[0]

        curTNode.tableP.table = genTable(len(tableShape),positive=True, odd = i%2 == 1)
        curTNode.tableN.table = genTable(len(tableShape),positive=False,odd = i%2 == 1)

        # curTNode.tableP.table = genProb(tableShape)
        # curTNode.tableN.table = genProb(tableShape)


        prevTNode = curTNode
        curTNode = deepcopy(prevTNode)

    yNode.parents = [prevTNode.name]
    yNode.table.varName = ['Y','T'+str(testNum-1)]
    tbn.nodes = nodes

    if not test:
        for n in tbn.nodes.values():
            if n.testing:
                n.table = deepcopy(n.tableP)
                n.testing = False

    for n,v in tbn.nodes.items():
        print(v.name)
        print(v.parents)
        if v.testing:
            print(v.tableP,'tableP')
            print(v.tableN,'tableN')
            print(v.thres,'thres')
        else:
            print(v.table,'table')
        pass

    return tbn




def getData(tac,topo=False):
    pX = findNode(ParameterNode,'X',tac.root)
    pnX = findNode(ParameterNode,'!X',tac.root)
    ps = np.linspace(0,1.0,num=50)
    ys = []
    for p in ps:
        pX.value = p
        pnX.value = 1-p
        if topo:
            v = tac.forward([])
        else:
            v = tac.forward([])
        ys.append(v)
        # print(v)
    return ps,ys


chainNum = 4
tbn = genChain(chainNum,test=True)
tac = compileTAC(tbn,'Y','')
#will output output.tac and output.lmap
tac2file(tac,'output')

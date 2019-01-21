from TestingBayesianNetwork import *
from typing import List, Set, Dict, Tuple, Optional
from utils import printTBN


# Node for elimination tree structure
class Node:
    def __init__(self, id=0):
        self.id = id
        self.adjacents = []
        self.depth = 0
        self.table = None
        # node in adjacents that leads to root(parent)
        self.parent = -1
        self.message = None
        self.tbnParents = []
        self.tbnParentsCPT = []
        # Name of the corresponding TBN node
        self.tbnName = ''

    def __str__(self):
        return f'''
<Node: id:{self.id} 
   adjacents:{self.adjacents} 
   depth:{self.depth}
   table:{self.table}
   parent:{self.parent}
   message:{self.message}
   tbnParents:{self.tbnParents}
   tbnParentsCPT:{self.tbnParentsCPT}
   tbnName:{self.tbnName}
>
        '''


class ETree:
    def __init__(self, length):
        self.nodes = [Node(id=i) for i in range(length)]
        self.root = 0
        self.sep = {}  # separator

    def getNode(self, id):
        return self.nodes[id]

    def setAsRoot(self, id):
        self.getNode(id).depth = 0
        self.getNode(id).parent = -1
        # single node in the Elimination tree
        if not self.nodes[id].adjacents:
            return [id], [id]
        # do a BFS to set the parent for each node
        queue = [id]
        explored = set()
        leaves = []
        topo = []
        while queue:
            pid = queue.pop()
            explored.add(pid)
            topo.append(pid)
            n = self.getNode(pid)
            d = n.depth
            for c in n.adjacents:
                if c in explored:
                    continue
                cn = self.getNode(c)
                cn.parent = pid
                cn.depth = d + 1
                #print(f'parent: {cn.tbnName} child: {n.tbnName}')
                queue = [c] + queue
            if not n.adjacents:
                leaves.append(id)
        return leaves, topo[::-1]

    # Only when adjacent nodes except parent have called the function
    def collectMessage(self, id,hashTable):
        n = self.getNode(id)
        # assert( n.parent >= 0 )
        # print('Collect %d '% id)
        pid = n.parent

        toMerge = [n.table] + list(map(lambda x: self.getNode(x).message, filter(lambda x: x != pid, n.adjacents)))

        tmp = factorMultiply(toMerge,hashTable = hashTable)
        n.message = factorProject(tmp, self.sep[(id, pid)],hashTable = hashTable)
        return pid

    def factorElimination(self, rootId,hashTable = None):
        leaves, topo = self.setAsRoot(rootId)
        for id in topo:
            if id == rootId:
                continue
            self.collectMessage(id,hashTable = hashTable)

    def __str__(self):
        s = ''
        for n in self.nodes:
            s += str(n) + '\n'
        return s

    # return topological sort(of ids) according to TBN topology
    def topoTBN(self):
        inDeg = [len(n.tbnParents) for n in self.nodes]
        seq = []
        zeroDeg = [id for id in range(len(self.nodes)) if inDeg[id] == 0]
        while zeroDeg:
            id = zeroDeg.pop()
            n = self.getNode(id)
            seq.append(id)
            for a in n.adjacents:
                if n in n.tbnParents:
                    continue
                else:
                    inDeg[a] -= 1
                    if inDeg[a] == 0 and not a in seq:
                        zeroDeg.append(a)
        return seq

    # For every node, only the message from parents(as in the TBN corresponding ones) are collected
    def collectMessageTBN(self, id,hashTable):

        n = self.getNode(id)
        # assert( n.parent >= 0 )
        # print('Collect %d '% id)
        pids = n.tbnParents


        toMerge = list(map(lambda x: self.getNode(x).message, filter(lambda x: x in pids, n.adjacents)))
        n.tbnParentsCPT = toMerge
        toMerge = [n.table] + toMerge

        # print(n)
        # print(toMerge)
        tmp = factorMultiply(toMerge,hashTable = hashTable)
        n.message = factorProject(tmp, [n.tbnName],hashTable = hashTable)


# polytree algorithm for bn where all nodes are chances nodes, bn's structure should be polytree
def polyTreeAC(bn: TestingBayesianNetwork, query: str, evidence: List[str], inst: bool):
    # print(query)
    # print(evidence)
    bn = pruneTBN(bn, [query] + evidence)
    # build corresponding elimination Tree
    etree = ETree(len(bn.nodes))
    bnNodeDict = {}
    for num, k in enumerate(bn.nodes.keys()):
        bnNodeDict[k] = num

    inputs = [None] * (2 * len(evidence))
    inputsD = {}
    ct = 0
    for ev in evidence:
        inputs[2 * ct] = IndicatorNode(label=ev, inst=True)
        inputs[2 * ct + 1] = IndicatorNode(label=ev, inst=False)
        inputsD[(ev, True)] = inputs[2 * ct]
        inputsD[(ev, False)] = inputs[2 * ct + 1]
        ct += 1

    for name, bnNode in bn.nodes.items():
        id = bnNodeDict[name]

        symbolTable = np.zeros(bnNode.table.table.shape, dtype=GeneralNode)

        # labelling thing
        l = bnNode.table.varName[:]
        l = [[x + ' ', '!' + x + ' '] for x in l]
        if len(l) > 1:
            l[0][0] += '\| '
            l[0][1] += '\| '
        l = np.array(l)

        # print('name:',bnNode.name,'\ntable:',bnNode.table)
        for pos, value in np.ndenumerate(bnNode.table.table):
            label = ''
            for i, v in enumerate(pos):
                label += l[i][v]
            if bnNode.name in evidence:
                symbolTable[pos] = MultiplyNode(children=[ParameterNode(value=value, label=label),
                                                          inputsD[(bnNode.name, True if pos[0] == 0 else False)]])
            else:
                symbolTable[pos] = ParameterNode(value=value, label=label)

        etree.getNode(id).table = deepcopy(bnNode.table)
        etree.getNode(id).table.table = symbolTable
        for p in bnNode.parents:
            pid = bnNodeDict[p]
            etree.getNode(id).adjacents.append(pid)
            etree.getNode(pid).adjacents.append(id)
            etree.sep[(id, pid)] = [p]
            etree.sep[(pid, id)] = [p]

    # print(etree)

    rootId = bnNodeDict[query]
    etree.factorElimination(rootId=rootId)

    rootNode = etree.getNode(rootId)
    # print(rootNode)
    toMerge = [rootNode.table] + list(map(lambda x: etree.getNode(x).message, rootNode.adjacents))
    root = factorProject(factorMultiply(toMerge), query).getProb((0 if inst else 1,))

    return ArithmeticCircuit(root=root, inputs=inputs)


# Prune a tbn such that only nodes which have children nodes in leaves are preserved
def pruneTBN(tbn: TestingBayesianNetwork, leaves: List[str]):
    marks = {}
    # print('D',leaves)
    for leaf in leaves:
        queue = [leaf]
        while queue:
            nname = queue.pop()
            n = tbn.nodes[nname]
            marks[nname] = True
            for p in n.parents:
                queue = [p] + queue
    # print('pruning TBN, only %s are preserved'%str(list(marks.keys())))
    tbn2 = TestingBayesianNetwork()
    tbn2.nodes = {k: v for k, v in tbn.nodes.items() if k in marks}
    return tbn2



def polyTreeTAC(tbn: TestingBayesianNetwork, query: str, evidence: List[str], inst: bool = True, normalized = False):

    hashTable = {}

    tbn = pruneTBN(tbn, [query] + evidence)
    # import pdb;pdb.set_trace()
    # printTBN(tbn)
    # build corresponding elimination Tree
    etree = ETree(len(tbn.nodes))

    # 1st Pass

    tbnNodeDict = {}

    for num, k in enumerate(tbn.nodes.keys()):
        tbnNodeDict[k] = num
    tbnNodeDictRev = {v:k for k,v in tbnNodeDict.items()}

    inputs = [None] * (2 * len(evidence))
    inputsD = {}
    ct = 0
    for ev in evidence:
        inputs[2 * ct] = uniqueNodeFactory(hashTable,IndicatorNode,label=ev, inst=True)
        inputs[2 * ct + 1] = uniqueNodeFactory(hashTable,IndicatorNode,label=ev, inst=False)
        inputsD[(ev, True)] = inputs[2 * ct]
        inputsD[(ev, False)] = inputs[2 * ct + 1]
        ct += 1

    for name, tbnNode in tbn.nodes.items():
        id = tbnNodeDict[name]

        symbolTable = np.zeros(tbnNode.table.table.shape, dtype=GeneralNode)

        # labelling thing
        l = tbnNode.table.varName[:]
        l = [[x + ' ', '!' + x + ' '] for x in l]
        if len(l) > 1:
            l[0][0] += '\| '
            l[0][1] += '\| '
        l = np.array(l)

        thresCache = {}
        # print('name:',bnNode.name,'\ntable:',bnNode.table)

        if not tbnNode.testing:
            for pos, value in np.ndenumerate(tbnNode.table.table):
                label = ''
                for i, v in enumerate(pos):
                    label += l[i][v]
                if tbnNode.name in evidence:
                    p = uniqueNodeFactory(hashTable,ParameterNode,value=value,label=label)
                    symbolTable[pos] = uniqueNodeFactory(hashTable,MultiplyNode,children=[p,
                                                              inputsD[(tbnNode.name, True if pos[0] == 0 else False)]])
                else:
                    p = uniqueNodeFactory(hashTable,ParameterNode,value=value,label=label)
                    symbolTable[pos] = p

        # Testing node canont be evidence
        else:
            for pos, value in np.ndenumerate(tbnNode.tableP.table):
                label = ''
                for i, v in enumerate(pos):
                    label += l[i][v]

                labelT = ''
                for i, v in list(enumerate(pos))[1:]:
                    labelT += l[i][v]

                # import pdb;pdb.set_trace()
                # print(pos)
                # print(pos[1:])
                if not pos[1:] in thresCache:
                    thresCache[pos[1:]] = uniqueNodeFactory(hashTable,ParameterNode,value=tbnNode.thres[pos[1:]], label='*' + labelT + ' Thres')

                t = thresCache[pos[1:]]

                # !!! Since this testing node will be altered during construction, only add it
                # to hashTable once it is fixed
                # thres stores the thres hold value temporarily
                pThetaD = uniqueNodeFactory(hashTable,ParameterNode,value=tbnNode.tableN.table[pos],label=label + ' -')
                pThetaU = uniqueNodeFactory(hashTable,ParameterNode,value=value, label=label + ' +')
                if tbnNode.name in evidence:
                    ind = inputsD[(tbnNode.name, True if pos[0] == 0 else False)]
                    pThetaD =  uniqueNodeFactory(hashTable,MultiplyNode,children=[pThetaD,ind])
                    pThetaU =  uniqueNodeFactory(hashTable,MultiplyNode,children=[pThetaU,ind])
                symbolTable[pos] = TestingNode(x=None,
                                               thetaD=pThetaD,
                                               thetaU=pThetaU,
                                               thres=t,
                                               # label=f'{n.table.varName}')
                                               label=label)

        etree.getNode(id).table = deepcopy(tbnNode.table)
        etree.getNode(id).table.table = symbolTable
        etree.getNode(id).tbnName = tbnNode.name

        for p in tbnNode.parents:
            pid = tbnNodeDict[p]
            etree.getNode(id).adjacents.append(pid)
            etree.getNode(pid).adjacents.append(id)
            etree.getNode(id).tbnParents.append(pid)
            etree.sep[(id, pid)] = [p]
            etree.sep[(pid, id)] = [p]

    seq = etree.topoTBN()
    # print('seq',seq)

    # print(etree)
    for id in seq:
        n = tbn.nodes[tbnNodeDictRev[id]]
        table = etree.getNode(id).table

        pids = etree.getNode(id).tbnParents
        toMerge = list(map(lambda x: etree.getNode(x).message, filter(lambda x: x in pids, etree.getNode(id).adjacents)))
        etree.getNode(id).tbnParentsCPT = toMerge
        #Just a work around, testing node should be created before message is collected

        parentsName = list(map(lambda x:tbnNodeDictRev[x],etree.getNode(id).tbnParents))

        # print(etree)
        if n.testing:
            tableU = factorMultiply(list(filter(lambda x:x,etree.getNode(id).tbnParentsCPT)),hashTable=hashTable)
            prE = factorProject(tableU,'',hashTable=hashTable)
            prU = factorProject(tableU,parentsName,hashTable=hashTable)
            varMap = []

            for v in prU.varName:
                try:
                    ind = table.varName.index(v)
                    varMap.append(ind)
                except ValueError:
                    continue

            for x in product(*map(range,table.varNum)):
                ass = np.array(x)[varMap]
                # at this time mNode should makes up of a Testing Node
                # print(tuple(x))
                tNode = table.table[tuple(x)]
                tNode.x = prU.table[tuple(ass)]

            for x in np.nditer(table.table,flags=['refs_ok'],op_flags=['readwrite']):
                assert(issubclass(type(prE.table.flat[0]),GeneralNode))
                t = nodeMultiply(hashTable,x.flat[0].thres,prE.table[0])
                x.flat[0].thres = t
                tNode = x.flat[0]
                x.flat[0]  = uniqueNodeFactory(hashTable,TestingNode,
                                               x=tNode.x,thetaU = tNode.thetaU,thetaD = tNode.thetaD,
                                               thres = tNode.thres,label=tNode.label)
        etree.collectMessageTBN(id,hashTable = hashTable)


    # print(etree)
    # Second Pass
    rootId = tbnNodeDict[query]
    etree.factorElimination(rootId=rootId,hashTable = hashTable)

    rootNode = etree.getNode(rootId)
    # print(rootNode)

    # print(etree)

    toMerge = [rootNode.table] + list(map(lambda x: etree.getNode(x).message, rootNode.adjacents))
    t = factorProject(factorMultiply(toMerge,hashTable = hashTable), query,hashTable = hashTable)
    print(type(t))
    print("t is: %s" % t)

    if normalized:
        prob1 = t.getProb((0,))
        prob2 = t.getProb((1,))
        if not inst:
            prob1,prob2 = prob2,prob1
        # root = NormalizeNode(children=[prob1,prob2])
        root = uniqueNodeFactory(hashTable,NormalizeNode,children=[prob1,prob2])
        ac = ArithmeticCircuit(root=root, inputs=inputs)

    else:
        root = t.getProb((0 if inst else 1,))
        ac = ArithmeticCircuit(root=root, inputs=inputs)
    ac.pruneNode()
    return ac




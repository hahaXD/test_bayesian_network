from graphviz import Digraph

from .ArithmeticCircuit import *
import numpy as np
from copy import copy, deepcopy
# Conditional Probability Table
# Right now a naive implementation
# Each probability is stored in a dict with keys "state1_state2_state3"


CPTType = GeneralNode

class CPTable:
    def __init__(self,varName=None,varNum=None,table = np.array([],dtype=CPTType)):
        # Convert variable to id
        if not varName:
            self.varName = []
        else:
            self.varName = varName
        if not varNum:
            self.varNum = []
        else:
            self.varNum = varNum
        self.table = table

    # Get the probability given a value assignment
    def getProb(self, ind):
        print(self.table.shape, self.varName)
        return self.table[tuple(ind)]

    # Get the probability given an instantiation
    def getProbByInst(self, **inst):
        ind = []
        try:
            for n in self.varName:
                ind.append(inst[n])
        except KeyError:
            raise RuntimeError('Incomplete instantiation to get prob.')
        return self.table[tuple(ind)]

    def __deepcopy__(self, memodict={}):
        id_self= id(self)
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            _copy.varNum = self.varNum[:]
            _copy.varName = self.varName[:]
            _copy.table = self.table.copy()
            memodict[id_self] = _copy
        return _copy
    def __str__(self): 
        return 'varName:{}\n'.format(self.varName)+\
        'varNum:{}\n'.format(self.varNum) + str(self.table)

class TBNNode:
    def __init__(self):
        self.name = ''
        self.states = []
        self.table = None
        # Pointers to parent node
        self.parents = []
        # pointers to child node
        self.children = []
        self.testing = False
        self.tableP = None
        self.tableN = None
        self.thres = np.zeros([1])
        self.evidence = set()
    def __copy__(self): 
        # define a copy function that is used in pruning
        # the copied TBN node shares the same tables with the original node
        # but has its own parent and children lists
        _copy = type(self)()
        _copy.name = self.name
        _copy.table = self.table
        _copy.tableP = self.tableP
        _copy.tableN = self.tableN
        _copy.states = self.states[:]
        _copy.parents = copy(self.parents)
        _copy.children = copy(self.children)
        _copy.testing = self.testing
        _copy.thres = self.thres.copy()
        _copy.evidence = set(self.evidence)
        return _copy

    def __deepcopy__(self, memodict={}):
        id_self= id(self)
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            _copy.name = self.name
            _copy.states = self.states[:]
            _copy.table = deepcopy(self.table)
            _copy.parents = deepcopy(self.parents)
            _copy.children = deepcopy(self.children)
            _copy.testing = self.testing
            _copy.tableP = deepcopy(self.tableP)
            _copy.tableN = deepcopy(self.tableN)
            _copy.thres = self.thres.copy()
            _copy.evidence = set(self.evidence)
            memodict[id_self] = _copy


        return _copy
    # Override the cmp operator to sort TBNNodes based on the num of evidence at or below it
    # Notice that nodes with more evidence is of higher priority
    def __lt__(self, other):
        return len(self.evidence) > len(other.evidence)


class TestingBayesianNetwork:
    def __init__(self):
        self.nodes = {}
    def __deepcopy__(self, memodict={}):
        id_self = id(self)
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            _copy.nodes = deepcopy(self.nodes)
            memodict[id_self] = _copy
        return _copy

    def fromNetFile(self,fname):
        pass
    def query(self,*symbols):
        pass
    def toDot(self):
        dot = Digraph(comment='Test')
        dot.attr(rankdir='LR')
        for n in self.nodes.values():
            if n.testing:
                dot.node(n.name,n.name,shape='doublecircle')
            else:
                dot.node(n.name,n.name)
        for n in self.nodes.values():
            for p in n.parents:
                dot.edge(p,n.name)
        return dot
    # correct the parent and child relationship after pruning
    def regularize(self):
        for n in self.nodes.values():
            n.parents = []
            n.children = []
            # clear the parents and children attribute
        for n in self.nodes.values():
            for p in n.table.varName[1:]:
                n.parents.append(p)
                self.nodes[p].children.append(n.name)
    # input a set of variable assignments
    # the first one must be this node
    # the followings come from parents (in order)
    def getCondProb(self,**settings):
        # settings = list(settings.values())
        nv = list(settings.keys())[0]
        nn = self.nodes[nv]
        offset = 0
        i = 0
        settingsl = []
        ind = [nn.states.index(settings[nv])]
        for p in nn.parents:
            settingsl.append((self.nodes[p],settings[p]))

        # calculate the offset of CPT
        for (p,v) in settingsl:
            ind.append(p.states.index(v))
        return nn.table.getProb(ind)

    def priorMarginal2(self,variables):
        s = []
        for n in self.nodes.values():
            s.append(deepcopy(n.table))
        t = factorMultiply(s)
        v = list(set(self.nodes)-set(variables))
        t = factorSum(t,v)
        return t

    def priorMarginal(self,variables):

        #get elimination order, currently naive order, to be optimized
        order = list(set(map(lambda x:x.name,self.nodes.values()))-set(variables))
        order.sort()

        s = []
        #deepcopy all the CPTTables
        for n in self.nodes.values():
            s.append(deepcopy(n.table))

        for n in order:
            # print('To eliminate: ',n)
            sToMerge = []
            s2 = []
            for ct in s:
                if  n in ct.varName:
                    sToMerge.append(ct)
                else:
                    s2.append(ct)
            s = s2
            # print('fuck')
            # for st in sToMerge:
            #     print(st)
            #     print('')
            # print('To eliminate:',n)
            # print('fuck')
            t = factorMultiply(sToMerge)
            # print('Result1')
            # print(t)
            # print('Result1')
            t = factorSum(t,[n])
            # print('Result')
            # print(t)
            # print('Result')
            if t:
                s.append(t)
        # print(s)
        return factorMultiply(s)

            
#implement the MultiplyFactors method in the textbook
from itertools import product
def factorMultiply(tables,hashTable=None):
# def factorMultiply(tables,hashTable):
    if len(tables) == 1:
        # return deepcopy(tables[0])
        return tables[0]

    fres = CPTable()
    varName = []
    varNum = []

    # map variable of each table to the resulting table's variable
    varMap = [[] for _ in range(len(tables))]
    
    for k in range(len(tables)):
        t = tables[k]
        assert(t.varName)
        assert(t.varNum)
        for i in range(len(t.varName)):
            try:
                n = varName.index(t.varName[i])
                # variable already in the resulting table
                varMap[k].append(n)
                continue
            except ValueError:
                varMap[k].append(len(varName))
                varName.append(t.varName[i])
                varNum.append(t.varNum[i])

    fres.varName = varName
    fres.varNum = varNum
    fres.table = np.zeros(fres.varNum,dtype=CPTType)
    fres.table.fill(CPTType())
    # iterate over different instantiations
    for x in product(*map(range,fres.varNum)):
        for i in range(len(tables)):
            t = tables[i]
            # assignment of table[i]
            asses = np.array(x)[varMap[i]]
            if hashTable:
                node = nodeMultiply(hashTable,fres.table[tuple(x)],t.table[tuple(asses)])
                fres.table[tuple(x)] = node
            else:
                fres.table[tuple(x)] *= t.table[tuple(asses)]
            # print('**')
            # print(fres.table)
            # print('**')
    return fres


# implement the SumOutVars algorithm in the textbook
def factorSum(tOrig,zs,hashTable=None):
# def factorSum(tOrig,zs,hashTable):
    if not zs:
        # print('No sum indeed')
        # return deepcopy(tOrig)
        return tOrig
    # print('Sum:',zs)
    fres = CPTable();
    fres.varName = list(set(tOrig.varName)-set(zs))

    if not fres.varName:
        fres.table = np.zeros([1],dtype=CPTType)
        fres.table[0] = CPTType()
        fres.varName = []
        fres.varNum = []
        for x in product(*map(range,tOrig.varNum)):
            if hashTable:
                node = nodeAdd(hashTable,fres.table[0],tOrig.table[tuple(x)])
                fres.table[0] = node
            else:
                fres.table[0]+=tOrig.table[tuple(x)]
        return fres

    #map fres variable to the original table positions
    varMap = []
    for v in fres.varName:
        varMap.append(tOrig.varName.index(v))
    fres.varNum = []

    for v in varMap:
        fres.varNum.append(tOrig.varNum[v])

    fres.table = np.zeros(fres.varNum,dtype=CPTType)
    fres.table.fill(CPTType())
    for x in product(*map(range,tOrig.varNum)):
        # print(fres)
        # print(tOrig)
        # print(x)
        # print(varMap)
        # print(zs)
        # print(tOrig.table.shape)
        # print(fres.table.shape)
        if hashTable:
            node = nodeAdd(hashTable,fres.table[tuple(np.array(x)[varMap])],tOrig.table[tuple(x)])
            fres.table[tuple(np.array(x)[varMap])] = node
        else:
            fres.table[tuple(np.array(x)[varMap])] += tOrig.table[tuple(x)]
    return fres


def factorProject(tOrigs,zs,hashTable = None):
# def factorProject(tOrigs,zs,hashTable):

    return factorSum(tOrigs,list(set(tOrigs.varName)-set(zs)),hashTable = hashTable)

### Important: All the variables
# Preposessing of TBN before compiling, receive the input of TBN,query y, evidence e,
# output lists of nodes' with their corresponding evidence
# in a topological order linearizing the TBN

def compileAnalysis(tbn,y,e,inst):
    inDeg = {}
    queue  = [tbn.nodes[y]]

    while queue:
        n = queue.pop()
        n.evidence = set()
        if not n in inDeg:
            inDeg[n] = 0
        for p in n.parents:
            if tbn.nodes[p] in inDeg:
                inDeg[tbn.nodes[p]]+=1
            else:
                inDeg[tbn.nodes[p]]=1
                queue = [tbn.nodes[p]] + queue

    s = []
    for x, v in inDeg.items():
        if v == 0:
            s.append(x)
    # print(s)
    seq = []
    # print(inDeg.values())
    while s:
        n = s.pop()
        seq.append(deepcopy(n))
        for p in n.parents:
            inDeg[tbn.nodes[p]] -=1
            if inDeg[tbn.nodes[p]] ==0:
                s = [tbn.nodes[p]] + s

    nodes = seq
    inputs = [None]*(2*len(e))
    inputsD = {}


    ct = 0
    for ev in e:
        inputs[2*ct] = IndicatorNode(label=ev,inst=True)
        inputs[2*ct+1] = IndicatorNode(label=ev,inst=False)
        inputsD[(ev,True)] = inputs[2*ct]
        inputsD[(ev,False)] = inputs[2*ct+1]
        ct+=1

    for n in nodes[::-1]:
        if n.name in e:
            n.evidence.add(n.name)
        for p in n.parents:
            n.evidence = n.evidence.union(tbn.nodes[p].evidence)

        # !! for testing node, table equals to the positive table(tableP)
        symbolTable = np.zeros(n.table.table.shape, dtype=CPTType)
        l = n.table.varName[:]
        l = [[x + ' ', '!' + x + ' '] for x in l]
        if len(l) > 1:
            l[0][0] += '\| '
            l[0][1] += '\| '
        l = np.array(l)

        thresCache = {}
        if not n.testing:
            for pos,value in np.ndenumerate(n.table.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=l[i][v]

                if n.name in e:
                    symbolTable[pos] = MultiplyNode(children=[ParameterNode(value=value,label=label),
                                                              inputsD[(n.name,True if pos[0] == 0 else False)]])
                elif n.name == y:
                    symbolTable[pos] = ParameterNode(value=value if inst == (pos[0] == 0) else 0,label=label)
                else:
                    symbolTable[pos] = ParameterNode(value=value,label=label)

        # Testing node canont be evidence
        else:
            for pos,value in np.ndenumerate(n.tableP.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=l[i][v]

                labelT = ''
                for i,v in list(enumerate(pos))[1:]:
                    labelT+=l[i][v]
                if not pos[1:] in thresCache:
                    thresCache[pos[1:]] =  ParameterNode(value=n.thres[pos[1:]],label='*'+labelT+' Thres')

                t = thresCache[pos[1:]]

                symbolTable[pos] = TestingNode(x=None,
                                               thetaN=ParameterNode(value = n.tableN.table[pos],label=label+' -'),
                                               thetaP=ParameterNode(value = value,label=label+' +'),
                                               thres=MultiplyNode(children=[t]),
                                               # label=f'{n.table.varName}')
                                               label=label)
        n.table.table = symbolTable
    # build elimination tree
    nodes = nodes[::-1]
    t = list(map(lambda x:set(x.table.varName),nodes))

    # var1[i] = var[i,i+1]
    var1 = [None]*(len(t)-1)
    # var2[i] = var[i+1,i]
    var2 = [None]*(len(t)-1)

    # print(nodes)
    # print(var1)
    # print(t)
    var1[0] = set(t[0])
    for i in range(0,len(var1)-1):
        var1[i+1] = var1[i].union(t[i+1])

    t2 = t[::-1]
    var2[0] = set(t2[0])
    for i in range(0,len(var2)-1):
        var2[i+1] = var2[i].union(t2[i+1])
    var2 = var2[::-1]
    #separator
    s = [None] * (len(t)-1)

    # print(t)
    # print(var1)
    # print(var2)

    for i in range(0,len(t)-1):
        s[i] = var1[i].intersection(var2[i])

    return nodes,inputs,s

def preprocess_topo(tbn, y, evidences, inst=True):
    for n in tbn.nodes.values():
        n.evidence = set() # clear the evidence

    InDeg = {n.name : len(n.parents) for n in tbn.nodes.values()} # a dictionary that maps nodes to num_of_parents
    roots = [n.name for n in tbn.nodes.values() if InDeg[n.name] == 0] # roots are nodes with no parents

    seq = []
    # sort TBN nodes in topological order
    while roots:
        n = roots.pop()
        nNode = tbn.nodes[n]
        if n != y:
            seq.append(deepcopy(nNode))
        for c in nNode.children:
            InDeg[c] -= 1
            if InDeg[c] == 0:
                print(c)
                roots = [c] + roots

    seq.append(deepcopy(tbn.nodes[y])) # append query to the end of topo order

    # initialize the indicator nodes
    inputs = []
    inputsD = {}
    for e in evidences:
        eNode = tbn.nodes[e]
        for i,state in enumerate(eNode.states):
            inputs.append(IndicatorNode(label=e, inst=state))
            inputsD[(e, i)] = inputs[-1]

    # for each node, intialize the CPT table
    for n in seq: 
        if n.name in evidences:
            n.evidence.add(n.name)
        for p in n.parents:
            n.evidence.union(tbn.nodes[p].evidence)

        labels = []
        for v in n.table.varName[:]:
            vNode = tbn.nodes[v]
            if len(vNode.states) == 2:
                labels.append([ v + ' ', '!' + v + ' '])
            else:
                labels.append([ v + '=' + state + ' ' for state in vNode.states])

        if len(labels) > 1:
            for label in labels[0]:
                label += '\| '

        # !! for testing node, table equals to the positive table(tableP)
        symbolTable = np.zeros(n.table.table.shape, dtype=CPTType)

        thresCache = {}
        if not n.testing:
            for pos,prob in np.ndenumerate(n.table.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=labels[i][v]

                if n.name in e: # for evidence, multiply indicator into corresponding CPT entry
                    symbolTable[pos] = MultiplyNode(children=[ParameterNode(value=prob,label=label), inputsD[(n.name,pos[0])]]) 

                elif n.name == y: # for query, we need to pass in an instantiation
                    symbolTable[pos] = ParameterNode(value=prob if inst == n.states[pos[0]] else 0, label=label)
                else:
                    symbolTable[pos] = ParameterNode(value=prob, label=label)

        else: # for testing nodes, what if testing node is query?
            for pos,prob in np.ndenumerate(n.tableP.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=labels[i][v]

                labelT = ''
                for i,v in enumerate(pos):
                    if i > 0:
                        labelT+=labels[i][v]
                if not pos[1:] in thresCache:
                    thresCache[pos[1:]] =  ParameterNode(value=n.thres[pos[1:]],label='*'+labelT+' Thres')

                t = thresCache[pos[1:]]

                symbolTable[pos] = TestingNode(x=None,
                                               thetaN=ParameterNode(value = n.tableN.table[pos],label=label+' -'),
                                               thetaP=ParameterNode(value = prob,label=label+' +'),
                                               thres=MultiplyNode(children=[t]),
                                               # label=f'{n.table.varName}')
                                               label=label)
        n.table.table = symbolTable
        # update the CPT table
    print("Finished initializing symbol table.")

    # reverse the topo order
    '''
    yNode = seq[-1]
    seq = seq[:-1][::-1] + [yNode]\
    '''
    # build elimination tree
    t = list(map(lambda x:set(x.table.varName),seq))
    # find the separator between factors

     # var1[i] = var[i,i+1]
    var1 = [None]*(len(t)-1)
    # var2[i] = var[i+1,i]
    var2 = [None]*(len(t)-1)

    var1[0] = set(t[0])
    for i in range(0,len(var1)-1):
        var1[i+1] = var1[i].union(t[i+1])

    t2 = t[::-1]
    var2[0] = set(t2[0])
    for i in range(0,len(var2)-1):
        var2[i+1] = var2[i].union(t2[i+1])
    var2 = var2[::-1]
    #separator
    s = [None] * (len(t)-1)
    for i in range(0,len(t)-1):
        s[i] = var1[i].intersection(var2[i])
    return seq,inputs,s







# Compile TBN, receive the input of TBN,query y, evidence e,output TAC
# Alpha version without handling testing nodes, just for test
# inst specifies the instantiation of the query variable
def compileTAC(tbn,y,e,inst=True):
    nodes,inputs,sep = compileAnalysis(tbn,y,e,inst)
    tbls = list(map(lambda x:x.table,nodes))
    print("In total %d factors" % len(tbls))

    print("seq: "),
    for n in nodes:
        print(n.name),
    print()

    for i in range(1,len(tbls)):

        node = nodes[i]
        if node.testing:

            prE = factorProject(tbls[i-1],'')
            prU = factorProject(tbls[i-1],node.parents)
            # print('prE',prE)
            varMap = []
            for v in prU.varName:
                try:
                    ind = tbls[i].varName.index(v)
                    varMap.append(ind)
                except Value:
                    continue
            for x in product(*map(range,tbls[i].varNum)):
                # print('**')
                # print(node.name)
                # print(node.table)
                # print(x)
                # print(varMap)
                # print('**')
                ass = np.array(x)[varMap]
                # at this time mNode should makes up of a Testing Node
                # print(tuple(x))
                tNode = tbls[i].table[tuple(x)]
                tNode.x = prU.table[tuple(ass)]

            for x in np.nditer(tbls[i].table,flags=['refs_ok']):
                # print(tbls[i].table.shape)
                # print(x)
                # print(x.flat)
                # print(type(x.flat[0]))
                # print(dir(x))
                # print(x.shape)
                x.flat[0].thres.children.append(prE.table[0])

        n = factorProject(tbls[i-1],sep[i-1])
        # print('###')
        # print(i)
        # print(tbls[i-1])
        # print(sep[i-1])
        # print('###')
        assert(n)
        tbls[i] = factorMultiply([n,tbls[i]])
        print("finished %dth factor" %i)


    # print(tbls[-1])
    # print('y is',y)
    t = factorProject(tbls[-1],[y])
    # print('***',t,'***')
    root  = t.getProb((0 if inst else 1,))
    print("finish generating arithmetic circuit")
    return ArithmeticCircuit(root=root,inputs=inputs)

import ArithmeticCircuit
import numpy as np
from copy import deepcopy
# Conditional Probability Table
# Right now a naive implementation
# Each probability is stored in a dict with keys "state1_state2_state3"


CPTType = ArithmeticCircuit.GeneralNode

class CPTable:
    def __init__(self):
        # Convert variable to id
        self.varName = []
        self.varNum = []
        self.table = np.array([],dtype=CPTType)

    # Get the probability given a value assignment
    def getProb(self, ind):
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
        return f'varName:{self.varName}\n'+\
        f'varNum:{self.varNum}\n' + str(self.table)

class BNNode:
    def __init__(self):
        self.name = ''
        self.states = []
        self.table = None
        # Pointers to parent node
        self.parents = []


class BayesianNetwork:
    def __init__(self):
        self.nodes = {}
    def fromNetFile(self,fname):
        pass
    def query(self,*symbols):
        pass

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

    def compile(self, variables):

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
            print('fuck')
            for st in sToMerge:
                print(st)
                print('')
            print('To eliminate:',n)
            print('fuck')
            t = factorMultiply(sToMerge)
            print('Result1')
            print(t)
            print('Result1')
            t = factorSum(t,[n])
            print('Result')
            print(t)
            print('Result')
            if t:
                s.append(t)
        # print(s)
        return factorMultiply(s)

            
#implement the MultiplyFactors method in the textbook
from itertools import product
def factorMultiply(tables):
    if len(tables) == 1:
        return deepcopy(tables[0])

    fres = CPTable()
    varName = []
    varNum = []
    # map variable of each table to the resulting table's variable

    varMap = [[] for _ in range(len(tables))]
    
    for k in range(len(tables)):
        t = tables[k]
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
            fres.table[tuple(x)] *= t.table[tuple(asses)]
            # print('**')
            # print(fres.table)
            # print('**')
    return fres


#implement the SumOutVars algorithm in the textbook
def factorSum(tOrig,zs):
    fres = CPTable();
    fres.varName = list(set(tOrig.varName)-set(zs))

    if not fres.varName:
        return None
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
        fres.table[tuple(np.array(x)[varMap])] += tOrig.table[tuple(x)]
    return fres



            

        


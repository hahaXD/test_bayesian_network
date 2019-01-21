import abc
import numpy as np
#from . import config
useSigmoid=True

# CPT selection settings
nodeDict = {}

def nodeDictReset():
    global nodeDict
    nodeDict = {}

# Inputs are the indicator nodes, one by one, true first
class ArithmeticCircuit:
    def __init__(self,root,inputs):
        self.root = root
        self.inputs = inputs
        l = []
        for ind in inputs:
            if ind.label not in l:
                l.append(ind.label)
        self.inputsVar = l
        self.seq = []
        self.topo()
        self.printTACStats()

    def checkEvidence(self, **evidence):
        return set(self.inputsVar) == set(evidence.keys())

    def forward(self, **evidence):
        return self.forwardSoft(**evidence)
        # self.checkEvidence(evidence)
        # # self.root.reset()
        # for i in range(len(evidence)):
        #     if evidence[i] == True:
        #         self.inputs[2*i].value = 1
        #         self.inputs[2*i+1].value = 0
        #     else:
        #         self.inputs[2*i].value = 0
        #         self.inputs[2*i+1].value = 1
        # global ct
        # return self.root.forward()

    def forwardHard(self, **evidence):
        if not self.checkEvidence(**evidence):
            raise RuntimeError('Inconsistent hard evidence.')
        for ind in self.inputs:
            if ind.inst == str(evidence[ind.label]):
                ind.value = 1
            else:
                ind.value = 0
        for n in self.seq:
            '''
            if n.id == 24 or n.id == 25:
                import pdb
                pdb.set_trace()
            '''
            if type(n) == ParameterNode or type(n) == IndicatorNode:
                continue
            else:
                n.forwardTopo()
        return self.seq[-1].value

    def forwardSoft(self, **evidence):
        if not self.checkEvidence(**evidence):
            raise RuntimeError('Inconsistent soft evidence.')
        values = []
        for ev in self.inputsVar:
            values += evidence[ev]
        if len(values) != len(self.inputs):
            raise RuntimeError('Inconsistent soft evidence.')
        for i in range(len(self.inputs)):
            self.inputs[i].value = values[i]
        #import pdb; pdb.set_trace()
        for n in self.seq:
            if type(n) == ParameterNode or type(n) == IndicatorNode:
                continue
            else:
                n.forwardTopo()
        return self.seq[-1].value

    def topo(self):
        inDeg = {}
        queue = [self.root]
        ids = set()
        while queue:
            n = queue.pop()
            if n.id not in ids:
                ids.add(n.id)
            else:
                print('loop node: %d' %n.id)
                raise RuntimeError('Find loops.')
            if not n in inDeg:
                inDeg[n] = 0
            if type(n) == AddNode or type(n) == MultiplyNode or type(n) == NormalizeNode:
                for c in n.children:
                    if not (issubclass(type(c),GeneralNode)):
                        # print(n.label)
                        for c in n.children:
                            # print(c.label)
                            pass
                        assert(False)
                    if not c in inDeg:
                        queue = [c] + queue
                        inDeg[c] = 1
                    else:
                        inDeg[c]+=1

            elif type(n) == TestingNode:
                l = [n.x,n.norm,n.thetaN,n.thetaP,n.thres]
                if n.gamma:
                    l.append(n.gamma)
                for x in l:
                    if x is None:
                        raise RuntimeError('Unflattened testing node')
                    if not x in inDeg:
                        queue = [x] + queue
                        inDeg[x] = 1
                    else:
                        inDeg[x] +=1
            else:
                if not (issubclass(type(n),GeneralNode)):
                    print(type(n))
                    print(n)
                    print(n[0].label)
                    raise RuntimeError('')
        s = []
        for x,v in inDeg.items():
            if v == 0:
                s.append(x)
        # print(s)
        seq = []
        #print('Nodes in ac: %d' %len(inDeg))
        while s:
            n = s.pop()
            seq.append(n)
            if type(n) == AddNode or type(n) == MultiplyNode or type(n) == NormalizeNode:
                for c in n.children:
                    inDeg[c]-=1
                    if inDeg[c]==0:
                        s = [c]+s
                        # inDeg.pop(c)
            elif type(n) == TestingNode:
                l = [n.x,n.norm,n.thetaN,n.thetaP,n.thres,n.gamma]
                for x in l:
                    if x is n.gamma and x is None:
                        continue
                    inDeg[x]-=1
                    if inDeg[x]==0:
                        s = [x]+s
                        # inDeg.pop(x)

        if np.array(list(inDeg.values())).sum()!=0:
            print(np.array(list(inDeg.values())).sum())
            print(inDeg.values())
            raise RuntimeError('Loop in graph!')

        self.seq = seq[::-1]
        # print('seq len:',len(self.seq))

    #use hashing to prune duplicate node in TAC
    def pruneNode(self):
        oldLen = len(self.seq)
        newSeq = []
        idTable = {}
        nodes = []
        hashTable = {}
        replaceTable = {}
        for i in range(len(self.seq)):
            n = self.seq[i]
            if type(n) == MultiplyNode or type(n) == AddNode or type(n) == NormalizeNode:
                for j in range(len(n.children)):
                    if n.children[j] in replaceTable:
                        n.children[j] = replaceTable[n.children[j]]
            elif type(n) == TestingNode:
                if n.thetaP in replaceTable:
                    n.thetaP = replaceTable[n.thetaP]
                elif n.thetaN in replaceTable:
                    n.thetaN = replaceTable[n.thetaN]
                elif n.thres in replaceTable:
                    n.thres = replaceTable[n.thres]
                elif n.x in replaceTable:
                    n.x = replaceTable[n.x]

            h = getHashKey(n,idTable,hashTable)
            if h in idTable:
                replaceTable[n] = nodes[idTable[h]]
                self.seq[i] = nodes[idTable[h]]
            else:
                idTable[h] = len(nodes)
                nodes.append(n)

        seen = set()
        newSeq = [n for n in self.seq if not(n in seen or seen.add(n))]
        print('Old size:%d new size: %d Pruned %d nodes'%(oldLen,len(newSeq),len(newSeq)-oldLen))
        self.seq = newSeq


    def printTACStats(self):
        num_nodes_total = len(self.seq)
        num_add_nodes = 0
        num_multiply_nodes = 0
        num_test_nodes = 0
        num_alive_test_nodes = 0
        num_ind_nodes = 0
        num_param_nodes = 0
        num_thres_nodes = 0
        num_gamma_nodes = 0
        num_normal_nodes = 0
        num_edges = 0
        for node in self.seq:
            if type(node) == GeneralNode:
                raise NotImplementedError()
            elif type(node) == IndicatorNode:
                num_ind_nodes += 1
            elif type(node) == ParameterNode:
                if "Thres" in node.label:
                    num_thres_nodes += 1
                elif "Gamma" in node.label:
                    num_gamma_nodes += 1
                else:
                    num_param_nodes += 1
            elif type(node) == AddNode:
                num_add_nodes += 1
                num_edges += len(node.children)
            elif type(node) == MultiplyNode:
                num_multiply_nodes += 1
                num_edges += len(node.children)
            elif type(node) == TestingNode:
                num_test_nodes += 1
                if node.alive:
                    num_alive_test_nodes += 1
                l = [node.x, node.norm, node.thetaP, node.thetaN, node.thres, node.gamma]
                num_edges += len(list(filter(lambda x: x is not None, l)))
            elif type(node) == NormalizeNode:
                num_normal_nodes += 1
                num_edges += len(node.children)
        print('TAC summary: %s nodes, %s edges' % (num_nodes_total, num_edges))
        print('%s AddNodes, %s MultiplyNodes, %s of %s TestingNodes are alive'% (num_add_nodes, num_multiply_nodes, num_alive_test_nodes, num_test_nodes))
        print('%s indicators, %s parameters, %s thresholds, %s gammas, and %s NormalizeNodes.'% (num_ind_nodes, num_param_nodes, num_thres_nodes, num_gamma_nodes, num_normal_nodes))

# save the tac to .tac & .lmap file
def tac2file(tac,filename):
    global visits
    global lVisits
    visits = {}
    lVisits = {}
    # print('fuck!!!',visits)
    ftac = open(filename+'.tac','w')
    flmap = open(filename+'.lmap','w')
    tac2fileRec(tac.root,ftac,flmap)
    ftac.close()
    flmap.close()

def tac2fileRec(node, ftac,flmap):
    global visits
    global indicatorV
    global lVisits
    if node in visits:
        return visits[node]
    if type(node) == AddNode or type(node) == MultiplyNode or type(node) == NormalizeNode:
        ids = []
        for x in node.children:
            id = tac2fileRec(x,ftac,flmap)
            ids.append(str(id))
        t = '+' if type(node) == AddNode else ('*' if type(node) == MultiplyNode else 'Z')
        visits[node] = len(visits)+1
        ftac.write('%s %s %s '%(visits[node], t, len(ids))+' '.join(ids)+'\n')
        node.id = visits[node]
        return visits[node]
    elif type(node) == TestingNode:
        l = [node.x, node.norm, node.thetaP, node.thetaN, node.thres, node.gamma]
        ids = [str(tac2fileRec(c, ftac, flmap)) for c in l if c is not None]
        visits[node] = len(visits)+1
        if len(ids) < 5:
            raise RunTimeError('Incomplete testing node.')
        ftac.write('%s ? %s '%(visits[node],len(ids)) + ' '.join(ids) + '\n')
        node.id = visits[node]
        return visits[node]
    elif type(node) == ParameterNode:
        visits[node] = len(visits)+1
        lVisits[node] = len(lVisits) + 1
        ftac.write('%s L %s'%(visits[node], lVisits[node])+ '\n')
        flmap.write('%s p %s %s'%(lVisits[node], node.value, node.label)+'\n')
        node.id = visits[node]
        return visits[node]
    else:
        assert(type(node) == IndicatorNode)
        visits[node] = len(visits)+1
        lVisits[node] = len(lVisits)+1
        flmap.write('%s i %s=%s'%(lVisits[node], node.label, node.inst) + '\n')
        ftac.write('%s L %s'%(visits[node], lVisits[node]) + '\n')
        node.id = visits[node]
        return visits[node]

# Binary Node
class GeneralNode:
    def __init__(self):
        self.cache = 0
        self.isCached = False
        self.id = 0
        self.value = 0
        pass
    @abc.abstractmethod
    def forward(self):
        pass
    def forwardTopo(self):
        pass
    def reset(self):
        pass
    # TODO possible optimization: when parameter is 0
    def __add__(self, other):
        # print('***This should not be called now***')
        # raise RuntimeError('')
        if type(self) == GeneralNode:
            return other
        elif type(other) == GeneralNode:
            return self
        elif type(self) == AddNode and type(other) == AddNode:
            return AddNode(self.children + other.children)
        elif type(self) == AddNode:
            children = self.children[:]
            children.append(other)
            return AddNode(children=children)
        elif type(other) == AddNode:
            children = other.children[:]
            children.append(self)
            return AddNode(children=children)
        else:
            return AddNode([self,other])

    # TODO possible optimization: when parameter is 0 or 1
    def __mul__(self, other):
        # print('***This should not be called now***')
        # raise RuntimeError('')
        if type(self) == GeneralNode:
            return other
        elif type(other) == GeneralNode:
            return self
        elif type(self) == MultiplyNode and type(other) == MultiplyNode:
            return MultiplyNode(self.children + other.children)
        elif type(self) == MultiplyNode:
            children = self.children[:]
            children.append(other)
            return MultiplyNode(children=children)
        elif type(other) == MultiplyNode:
            children = other.children[:]
            children.append(self)
            return MultiplyNode(children=children)
        else:
            return MultiplyNode([self,other])


class MultiplyNode(GeneralNode):
    def __init__(self,children = None):
        super().__init__()
        if children == None:
            self.children = []
        else:
            self.children = children
    def forward(self):
        if self.isCached:
            return self.cache
        else:
            global ct
            ct+=1
            # print(ct)
            res = 1
            for c in self.children:
                try:
                    res*=c.forward()
                except AttributeError:
                # except ValueError:
                    print('Something is wrong:',type(c))
            self.cache = res
            self.isCached = True
            return self.cache
    def forwardTopo(self):
        res = 1
        for c in self.children:
            try:
                res*=c.value
            except TypeError:
                print('hehe')
                print('the id of multiply node is: %d' %self.id)
                print('the id of parameter node is: %d' %c.id)
                print('the id of parameter value is: %d' %c.value.id)
                print(type(res))
                print(type(c))
                print(type(c.value))
                exit(0)
                print('**',c.thetaP.label,'**')
                print(c.thetaP.value)
                print(c.thetaN.value)
                print(c.thres.value)
                print(c.x.value)
                if not (type(c) == MultiplyNode or type(c) == AddNode):
                    print(c.label)
                print('hehe')
                assert(False)
        self.value = res

    def reset(self):
        self.isCached = False
        for c in self.children:
            c.reset()


ct = 0
class AddNode(GeneralNode):
    def __init__(self,children = None):
        super().__init__()
        if children == None:
            self.children = []
        else:
            self.children = children
    def forward(self):
        if self.isCached:
            return self.cache
        else:
            # global ct
            # ct+=1
            # print(ct)
            res = 0
            for c in self.children:
                res+=c.forward()
            self.cache = res
            self.isCached = True
            return self.cache
    def forwardTopo(self):
        res = 0
        for c in self.children:
            res+=c.value
        self.value = res
    def reset(self):
        self.isCached = False
        for c in self.children:
            c.reset()

class IndicatorNode(GeneralNode):
    def __init__(self,value = 1,label = '',inst=True):
        super().__init__()
        self.value = value
        self.label = label
        self.inst = inst
    def forward(self):
        return self.value

class ParameterNode(GeneralNode):
    def __init__(self,value = 0,label=''):
        super().__init__()
        self.value = value
        self.label = label
    def forward(self):
        return self.value

class TestingNode(GeneralNode):
        def __init__(self, x=None, norm=None, thetaP=None, thetaN=None, thres=None, gamma=None, alive=False, label=''):
            super().__init__()
            self.thetaP = thetaP
            self.thetaN = thetaN
            self.thres = thres
            self.gamma = gamma
            self.x = x
            self.norm = norm
            self.alive = alive
            self.label = label
        def forward(self):
            if self.isCached:
                return self.cache
            else:
                self.cache = self.thetaP.forward() if self.x.forward() / self.norm.forward() >= self.thres.forward() else self.thetaN.forward()
                self.isCached = True
                return self.cache
        def reset(self):
            self.isCached = False
            self.thetaP.reset()
            self.thetaN.reset()
            self.thres.reset()
            self.gamma.reset()
            self.x.reset()
            self.norm.reset()
            self.alive = False
        def forwardTopo(self):
            def sigmoid(a, x):
                return 1 / (1 + np.exp(-1*a*x))
            if useSigmoid:
                #print('with sigmoid.')
                try:
                    tau = sigmoid(self.gamma.value, self.x.value / self.norm.value - self.thres.value)
                except ZeroDivisionError:
                    print('divide by 0!')
                    print('id=%s x=%s n=%s'% (self.id,self.x.value,self.norm.value))
                    exit(0)
                self.value = tau * self.thetaP.value + (1 - tau) * self.thetaN.value
            else:
                self.value =  self.thetaP.value if self.x.value / self.norm.value >= self.thres.value else self.thetaN.value


class NormalizeNode(GeneralNode):
    def __init__(self,children = None):
        super().__init__()
        if children == None:
            self.children = []
        else:
            self.children = children
    def forward(self):
        res = 0
        for c in self.children:
            res += c.forward()
        return self.children[0].forward()/res
    def forwardTopo(self):
        res = 0
        for c in self.children:
            res+=c.value
        self.value  =  self.children[0].value/res

#return a multiply node of theta and indicator
def circuitFactorEntry(prob,label,inst,indicator=None):
    if not indicator:
        return MultiplyNode([ParameterNode(value = prob),IndicatorNode(label = label,inst = inst)])
    else:
        return MultiplyNode([ParameterNode(value = prob),indicator])

def testingFactorEntry(probP,probN,label,inst,indicator=None):
    pass
    # if not indicator:
    #     return MultiplyNode(ParameterNode(value = prob),IndicatorNode(label = label,inst = inst))
    # else:
    #     return MultiplyNode(ParameterNode(value = prob),indicator)


# get hash key of a node , for node with children, hash value is concatenation of type+sorted children id
def getHashKey(node:GeneralNode,idTable,hashTable):
    if node in hashTable:
        return hashTable[node]
    h = None
    if type(node) == ParameterNode:
        h =  (ParameterNode,node.label)
    elif type(node) == IndicatorNode:
        h =  (IndicatorNode,node.label,node.inst)
    elif type(node) == MultiplyNode or type(node) == AddNode or type(node) == NormalizeNode:
        cs = []
        for c in node.children:
            cs.append(idTable[getHashKey(c,idTable,hashTable)])
        cs.sort()
        h =  (type(node),)+tuple(cs)
    elif type(node) == TestingNode:
        cs = []
        cs.append(idTable[getHashKey(node.thetaP,idTable,hashTable)])
        cs.append(idTable[getHashKey(node.thetaN,idTable,hashTable)])
        cs.append(idTable[getHashKey(node.thres,idTable,hashTable)])
        cs.append(idTable[getHashKey(node.x,idTable,hashTable)])
        h =  (type(node),)+tuple(cs)
    else:
        raise RuntimeError('')

    hashTable[node] = h
    return h


def uniqueNodeHash(nodeType,kwargs):
    if nodeType == ParameterNode:
        h =  (ParameterNode,kwargs['label'])
    elif nodeType == IndicatorNode:
        h =  (IndicatorNode,kwargs['label'],kwargs['inst'])
    elif nodeType == MultiplyNode or nodeType == AddNode or nodeType == NormalizeNode:
        cs = []
        for c in kwargs['children']:
            cs.append(c.id)
        cs.sort()
        h = (nodeType,)+tuple(cs)
    elif nodeType == TestingNode:
        cs = []
        cs.append(kwargs['thetaP'].id)
        cs.append(kwargs['thetaN'].id)
        cs.append(kwargs['thres'].id)
        cs.append(kwargs['x'].id)
        h =  (nodeType,)+tuple(cs)
    else:
        raise RuntimeError('')
    return h

def uniqueNodeFactory(hashTable,nodeType,**kwargs):
    h = uniqueNodeHash(nodeType,kwargs)
    if h in hashTable:
        return hashTable[h]
    else:
        id = len(hashTable)
        node = nodeType(**kwargs)
        node.id = id
        hashTable[h] = node
        return node

def uniqueNodeLookup(hashTable,nodeType,**kwargs):
    h = uniqueNodeHash(nodeType,kwargs)
    return hashTable[h]


def nodeAdd(hashTable,one, other):
    if type(one) == IndicatorNode:
        print("adding two evidence")
        exit(0)
    if type(one) == GeneralNode:
        return other
    elif type(one) == GeneralNode:
        return one
    elif type(one) == AddNode and type(other) == AddNode:
        return uniqueNodeFactory(hashTable,AddNode,children=one.children + other.children)
    elif type(one) == AddNode:
        children = one.children[:]
        children.append(other)
        return uniqueNodeFactory(hashTable,AddNode,children=children)
    elif type(other) == AddNode:
        children = other.children[:]
        children.append(one)
        return uniqueNodeFactory(hashTable,AddNode,children=children)
    else:
        return uniqueNodeFactory(hashTable,AddNode,children=[one,other])

# TODO possible optimization: when parameter is 0 or 1
def nodeMultiply(hashTable,one, other):
    if type(one) == GeneralNode:
        return other
    elif type(other) == GeneralNode:
        return one
    elif type(one) == MultiplyNode and type(other) == MultiplyNode:
        return uniqueNodeFactory(hashTable,MultiplyNode,children=one.children + other.children)
    elif type(one) == MultiplyNode:
        children = one.children[:]
        children.append(other)
        return uniqueNodeFactory(hashTable,MultiplyNode,children=children)
    elif type(other) == MultiplyNode:
        children = other.children[:]
        children.append(one)
        return uniqueNodeFactory(hashTable,MultiplyNode,children=children)
    else:
        return uniqueNodeFactory(hashTable,MultiplyNode,children=[one,other])

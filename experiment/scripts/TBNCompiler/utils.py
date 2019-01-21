from . import ArithmeticCircuit
from graphviz import Digraph


dot = Digraph(comment='Test')
dot.attr(rankdir='BT')



ct = 0

# prevent multiple graph nodes for single ac nodes
d = {}
def genDiagram(ac,showID = False,showValue = True):
    global dot
    global d
    global ct
    d = {}
    ct = 0
    dot = Digraph(comment='Test')
    # dot.attr(rankdir='BT')
    dot.attr(rankdir='LR')
    genDiagramRec(ac,showID,showValue=showValue)
    return dot

def genDiagramRec(node,showID = False,showValue = True):
    global ct
    global dot
    global d
    if node in d:
        return d[node]
    else:
        id = str(ct)
        d[node] = id
    ct+=1
    if type(node) == ArithmeticCircuit.GeneralNode:
        raise NotImplementedError()
    elif type(node) == ArithmeticCircuit.IndicatorNode:
        s = '' if node.inst else '!'
        dot.node(id,'I '+s+node.label+' '+(str(node.value) if showValue else '')+(f' id:{node.id}' if showID else ''),shape='record')
    elif type(node) == ArithmeticCircuit.ParameterNode:
        dot.node(id,'P '+node.label+'   '+(str(node.value) if showValue else '')+(f' id:{node.id}' if showID else ''),shape='record',color='lightgrey',style='filled')
    elif type(node) == ArithmeticCircuit.MultiplyNode:
        dot.node(id,'*'+(f' id:{node.id}' if showID else ''))
        for c in node.children:
            id2 = genDiagramRec(c,showID,showValue)
            dot.edge(id2,id)
    elif type(node) == ArithmeticCircuit.AddNode:
        # print('fuckAdd')
        dot.node(id,'+'+ (f'id:{node.id}' if showID else ''))
        for c in node.children:
            id2 = genDiagramRec(c,showID,showValue)
            dot.edge(id2,id)
    elif type(node) == ArithmeticCircuit.NormalizeNode:
        # print('fuckAdd')
        dot.node(id,'Z '+ (f'id:{node.id}' if showID else ''))
        for c in node.children:
            id2 = genDiagramRec(c,showID,showValue)
            if c == node.children[0]:
                dot.edge(id2,id,label='Output')
            else:
                dot.edge(id2,id)
    elif type(node) == ArithmeticCircuit.TestingNode:
        # print('fuckTest')
        dot.node(id,f'? {node.label}' + (f'id:{node.id}' if showID else ''),shape='diamond',color='grey',style='filled')
        id2 = genDiagramRec(node.x,showID,showValue)
        dot.edge(id2,id,label='x')
        id2 = genDiagramRec(node.thetaP,showID,showValue)
        dot.edge(id2,id,label='Positive')
        id2 = genDiagramRec(node.thetaN,showID,showValue)
        dot.edge(id2,id,label='Negative')
        id2 = genDiagramRec(node.thres,showID,showValue)
        dot.edge(id2,id,label='Thres')
    return id


# save the id of visited nodes and literals
visits  = {}
lVisits = {}

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
    if type(node) == ArithmeticCircuit.AddNode or type(node) == ArithmeticCircuit.MultiplyNode or type(node) == ArithmeticCircuit.NormalizeNode:
        ids = []
        for x in node.children:
            id = tac2fileRec(x,ftac,flmap)
            ids.append(str(id))
        t = '+' if type(node) == ArithmeticCircuit.AddNode else ('*' if type(node) == ArithmeticCircuit.MultiplyNode else 'Z')
        visits[node] = len(visits)+1
        ftac.write(f'{visits[node]} {t} {len(ids)} '+' '.join(ids)+'\n')
        node.id = visits[node]
        return visits[node]
    elif type(node) == ArithmeticCircuit.TestingNode:
        l = [node.x, node.norm, node.thetaP, node.thetaN, node.thres, node.gamma]
        ids = [str(tac2fileRec(c, ftac, flmap)) for c in l if c is not None]
        visits[node] = len(visits)+1
        if len(ids) < 5:
            raise RunTimeError('Incomplete testing node.')
        ftac.write(f'{visits[node]} ? {len(ids)} ' + ' '.join(ids) + '\n')
        node.id = visits[node]
        return visits[node]
    elif type(node) == ArithmeticCircuit.ParameterNode:
        visits[node] = len(visits)+1
        lVisits[node] = len(lVisits) + 1
        ftac.write(f'{visits[node]} L {lVisits[node]}' + '\n')
        flmap.write(f'{lVisits[node]} p {node.value} {node.label}'+'\n')
        node.id = visits[node]
        return visits[node]
    else:
        assert(type(node) == ArithmeticCircuit.IndicatorNode)
        visits[node] = len(visits)+1
        lVisits[node] = len(lVisits)+1
        flmap.write(f'{lVisits[node]} i {node.label}={node.inst}' + '\n')
        ftac.write(f'{visits[node]} L {lVisits[node]}' + '\n')
        node.id = visits[node]
        return visits[node]

def file2tac(filename):
    ftac = open(filename+'.tac','r')
    flmap = open(filename+'.lmap','r')
    literals = {}
    nodes = {}
    inputs = []
    ct = 0
    for l in flmap:
        if not l:
            continue
        ct+=1
        s = l.split(' ',maxsplit=3)
        if s[1] == 'p':
            literals[ct]=ArithmeticCircuit.ParameterNode(value=float(s[2]), label=s[3])
        elif s[1] == 'i':
            s = l.split(' ',maxsplit=2)
            literals[ct] = ArithmeticCircuit.IndicatorNode(label=s[2])
            inputs.append(literals[ct])
            literals[-1*ct] = ArithmeticCircuit.IndicatorNode(label=s[2], inst=False)
            inputs.append(literals[-1*ct])
    ct = 0
    for l in ftac:
        if not l.strip():
            continue
        ct+=1
        sp = l.split(' ')
        if sp[1] == 'L':
            nodes[ct] = literals[int(sp[2])]
        elif sp[1] == '*' or sp[1] == '+' or sp[1] == 'Z':
            nodes[ct] = ArithmeticCircuit.MultiplyNode() if sp[1] == '*' else (ArithmeticCircuit.AddNode() if sp[1] == '+' else ArithmeticCircuit.NormalizeNode())
            for c in sp[3:]:
                nodes[ct].children.append(nodes[int(c)])
        elif sp[1] == '?':
            nodes[ct] = ArithmeticCircuit.TestingNode(x=nodes[int(sp[3])], thres=nodes[int(sp[4])]
                                                      , thetaN=nodes[int(sp[5])], thetaP=nodes[int(sp[6])])
        nodes[ct].id = int(sp[0])
    # print(len(literals))
    # print(len(nodes))
    for n in nodes.values():
        # print(f'node:{n.id} children:')
        if type(n) == ArithmeticCircuit.ParameterNode or type(n) == ArithmeticCircuit.IndicatorNode or type(n) == ArithmeticCircuit.TestingNode:
            continue
        for c in n.children:
            # print('\t',c.id)
            pass
    return ArithmeticCircuit.ArithmeticCircuit(root=nodes[ct], inputs=inputs)

#find node from TAC with specified type and name
def findNode(typeN,label,node):
    if type(node) == typeN and label == node.label.strip():
        return node
    if type(node) == ArithmeticCircuit.MultiplyNode or type(node) == ArithmeticCircuit.AddNode:
        for c in node.children:
            n = findNode(typeN,label,c)
            if n:
                return n

    return None

def printTBN(tbn):
    for k, v in tbn.nodes.items():
        print('***')
        print('name:', v.name)
        print('parents:', v.parents)
        if v.testing:
            print('tableP:',v.tableP,'\n')
            print('tableN:',v.tableN,'\n')
            print('thres:',v.thres,'\n')
        else:
            print('table',v.table)
        print('***')


def tacToPdf(tac,filename,showID=False,showValue = True):
    d = genDiagram(tac.root,showID=showID,showValue = showValue)
    d.save(filename)

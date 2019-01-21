from BayesianNetwork import *
from lxml import etree

# Read from BIF XML Bayesian Data File
def readBNXML(fileName):
    t = etree.parse(fileName).getroot()
    vs = t[0].findall('VARIABLE')
    ds = t[0].findall('DEFINITION')
    bn = BayesianNetwork()
    for v in vs:
        node = BNNode()
        node.name = v.find('NAME').text
        node.states = list(map(lambda x:x.text,v.findall('OUTCOME')))
        bn.nodes[node.name] = node
    for d in ds:
        n = d.find('FOR').text
        tmp = d.findall('GIVEN')
        ps = list(map(lambda x:x.text,tmp))
        bn.nodes[n].parents = ps
        table = CPTable()
        table.varName = [n]+ps[:]
        table.varNum = []
        for v in table.varName:
            table.varNum.append(len(bn.nodes[v].states))
        probs = list(map(float,d.find('TABLE').text.strip().split(' ')))
        table.table = np.zeros(table.varNum,dtype=CPTType)
        table.table.fill(CPTType())
        for x in product(*map(range,table.varNum)):
            offset = 0
            for k in range(1,len(table.varNum)):
                offset+=x[k]
                offset*=table.varNum[k]
            offset+=x[0]
            # print(f'offset:{offset} x:{x}')
            #TODO
            label = 'no'
            table.table[tuple(x)] = circuitFactorEntry(probs[offset],label,inst= x[0]==0)

        bn.nodes[n].table = table
    return bn



            
    
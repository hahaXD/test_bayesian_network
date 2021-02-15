from .TestingBayesianNetwork import *
from lxml import etree

# Read from BIF XML Bayesian Data File
def readTBNXML(fileName):
    t = etree.parse(fileName).getroot()
    vs = t[0].findall('VARIABLE')
    ds = t[0].findall('DEFINITION')
    tbn = TestingBayesianNetwork()
    for v in vs:
        node = TBNNode()
        node.name = v.find('NAME').text
        node.states = list(map(lambda x:x.text,v.findall('OUTCOME')))
        tbn.nodes[node.name] = node
    for d in ds:
        n = d.find('FOR').text
        tmp = d.findall('GIVEN')
        ps = list(map(lambda x:x.text,tmp))
        tbn.nodes[n].parents = ps
        for p in ps:
            tbn.nodes[p].children.append(n)
        table = CPTable()
        table.varName = [n]+ps[:]
        table.varNum = []
        for v in table.varName:
            table.varNum.append(len(tbn.nodes[v].states))
        probs = list(map(float,d.find('TABLE').text.strip().split(' ')))
        #print(f'length of probability: {len(probs)}')
        table.table = np.zeros(table.varNum,dtype='float64')
        # table.table.fill(CPTType())
        for x in product(*map(range,table.varNum)):
            offset = 0
            for k in range(1,len(table.varNum)):
                offset+=x[k]
                if k + 1 < len(table.varNum):
                    offset *= table.varNum[k + 1]
                else:
                    offset *= table.varNum[0]
            offset+=x[0]
            #print(f'offset:{offset} x:{x}')
            label = table.varName[0]
            table.table[tuple(x)] = probs[offset]

        tbn.nodes[n].table = table
    return tbn



            
    

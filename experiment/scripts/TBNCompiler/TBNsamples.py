from TestingBayesianNetwork import *
from ArithmeticCircuit import *

cptH = CPTable(varName=['H'],varNum=[2])
cptH.table = {
             True:circuitFactorEntry(0.2,'H',True),
             False:circuitFactorEntry(0.8,'H',False)
}

iX1t = IndicatorNode(value = 1,label='X1',inst=True)
iX1f = IndicatorNode(value = 1,label='X1',inst=False)
iX2t = IndicatorNode(value = 1,label='X2',inst=True)
iX2f = IndicatorNode(value = 1,label='X2',inst=False)
iX3t = IndicatorNode(value = 1,label='X3',inst=True)
iX3f = IndicatorNode(value = 1,label='X3',inst=False)
iYt = IndicatorNode(value = 1,label='Y',inst=True)
iYf = IndicatorNode(value = 1,label='Y',inst=False)

cptX1 = CPTable(varName=['X1'],varNum=[2,2])
cptX1.table = {
    (True,True):circuitFactorEntry(prob=0.1,indicator=iX1t),
    (True,False):circuitFactorEntry(prob=0.9,indicator=iX1t),
    (False,True):circuitFactorEntry(prob=0.9,indicator=iX1f),
    (False,False):circuitFactorEntry(prob=0.1,indicator=iX1f),
}

cptX2 = CPTable(varName=['X2'],varNum=[2,2])
cptX2.table = {
    (True,True):circuitFactorEntry(prob=0.2,indicator=iX2t),
    (True,False):circuitFactorEntry(prob=0.8,indicator=iX2t),
    (False,True):circuitFactorEntry(prob=0.8,indicator=iX2f),
    (False,False):circuitFactorEntry(prob=0.2,indicator=iX2f),
}

cptX3 = CPTable(varName=['X3'],varNum=[2,2])
cptX3.table = {
    (True,True):circuitFactorEntry(prob=0.3,indicator=iX3t),
    (True,False):circuitFactorEntry(prob=0.7,indicator=iX3t),
    (False,True):circuitFactorEntry(prob=0.7,indicator=iX3f),
    (False,False):circuitFactorEntry(prob=0.3,indicator=iX3f),
}


cptX3 = CPTable(varName=['Y'],varNum=[2,2])
cptX3.table = {
    (True,True):circuitFactorEntry(prob=0.3,indicator=iYt),
    (True,False):circuitFactorEntry(prob=0.7,indicator=iYt),
    (False,True):circuitFactorEntry(prob=0.7,indicator=iYf),
    (False,False):circuitFactorEntry(prob=0.3,indicator=iYf),
}

from TestingBayesianNetwork import *
from ArithmeticCircuit import *
from graphviz import Digraph
from readTBNXML import *
from utils import genDiagram,file2tac,tac2file
from typing import List
from testcases import *
from compile import polyTreeAC,polyTreeTAC
from compile_ve import *
from chaintest import *
import matplotlib.pyplot as plt

# test selection
print('test selction...')
bn = readTBNXML('samples/cancer.xml')
tbn = bn2Tbn(bn, ['B', 'C', 'D', 'E'], 0.7, randomTable=True)
tac = compileTAC_ve(tbn, 'A', ['B', 'C', 'D', 'E'], normalized=True)
tac2file(tac, 'samples/simple1')
print('Finish writing tac to files')


query = tac.forwardSoft(B=[1.0, 0.0], C=[1.0, 0.0], D=[1.0, 0.0], E=[1.0, 0.0])
print('query value %.5f' %query)

# test rectangle
print('test rectangle...')
bn = readTBNXML('samples/rectangleModelBinary.xmlbif')
roots = ["Height", "Width", "X_ref", "Y_ref"]
evidence = []
NUM_ROWS = 5
NUM_COLS = 5
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        evidence.append('Out_' + str(i) + '_' + str(j))
tests = [n for n in bn.nodes.keys() if n not in roots and n not in evidence]
tbn = bn2Tbn(bn, tests, -0.5, randomTable=True)
tbn_graph = tbn.toDot()
#tbn_graph.render('samples/rectangle.pdf', view=True)

ac = compileTAC_ve(tbn, 'Height', evidence, normalized=True)
tac2file(ac, 'samples/rectangle')
exit(0)

exit(0)

# test rectangle
print('test rectangle...')
bn = readTBNXML('samples/rectangleModel10by10.xml')
tbn = bn2Tbn(bn, bn.nodes.keys(), 0.7, randomTable=True)
tbn_graph = tbn.toDot()
#tbn_graph.render('samples/rectangle.pdf', view=True)
evidences = []
NUM_ROWS = 5
NUM_COLS = 5
for i in range(NUM_ROWS):
	for j in range(NUM_COLS):
		evidences.append('Out_' + str(i) + '_' + str(j))
ac = compileTAC_ve(tbn, 'Height', evidences, normalized=True)
tac2file(ac, 'samples/cancerTesting')
exit(0)

# read in the original bn
bn = readTBNXML('samples/cancer.xml')
bn_dot = bn.toDot()
#bn_dot.view('cancer')


# inject the testing nodes
print('inject testing nodes...')
tbn = bn2Tbn(bn, ['C'], 0.7, randomTable=True)
tbn_dot = tbn.toDot()
#tbn_dot.view('cancer_testing')
printTBN(tbn)

# test pruning
print('test pruning...')
tbn2 = pruneTBN(tbn, ['A', 'C'])
printTBN(tbn2)

# test initialization
print('test initializing...')
hashtable = {}
initTBN(tbn2, 'A', ['C'], hashtable)
#for label in hashtable:
	#print(label)
#genGraphNodes(hashtable)

# test minfill function
print('test minfill...')
order = minfillSort(tbn2, ["A"])
print(order)

# test variable elimination
print('test variable elimination')
prior = variableElimination(tbn2, ['A'], hashtable)
#genGraphNodes(hashtable)


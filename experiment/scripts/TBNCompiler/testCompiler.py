from TestingBayesianNetwork import *
from readTBNXML import *
from utils import genDiagram,file2tac,tac2file
from typing import List
from testcases import *
from compile import polyTreeAC,polyTreeTAC
from compile_ve import *
from chaintest import *
import matplotlib.pyplot as plt
import graphviz

bn = readTBNXML('samples/cancer.xml')
d = bn.toDot()
d.view('cancer')
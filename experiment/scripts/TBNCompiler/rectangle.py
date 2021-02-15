from readTBNXML import *
from TestingBayesianNetwork import *
from utils import tac2file
import sys

# parameters of the rectangle net
NUM_ROWS = 5
NUM_COLS = 5

rectangle_filename = "samples/rectangleModelBinary.xmlbif"
rectangle_tbn = readTBNXML("samples/rectangleModelBinary.xmlbif")

query = "Height"
evidences = []
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        cell = "Out" + "_" + str(i) + "_" + str(j)
        evidences.append(cell)

arguments = sys.argv[1:]
if len(arguments) != 0 and arguments[0] == "test":
	test_tbn = readTBNXML("samples/cancer.xml")
	test_tac = compileTAC(test_tbn, "A", ["D", "E"], inst=False)
	tac2file(test_tac,'samples/cancerchange')
else:
	rectangle_tac = compileTAC(rectangle_tbn, query, evidences, inst=False)
	tac2file(rectangle_tac,'samples/rectangle')

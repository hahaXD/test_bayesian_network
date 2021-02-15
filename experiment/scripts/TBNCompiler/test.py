from TestingBayesianNetwork import *
from readTBNXML import *
from readBNXML import *
from utils import genDiagram,file2tac,tac2file
from typing import List
from testcases import *
from compile import polyTreeAC,polyTreeTAC
from compile_ve import *
from chaintest import *
import matplotlib.pyplot as plt
import graphviz
import sys

# fs = 18 # fontsize
# plt.rcParams.update({'xtick.labelsize': fs, 'ytick.labelsize': fs,
#                        'figure.autolayout': True})
# font = {'family' : 'sans-serif',
#         'weight' : 'normal',
#         'size'   : 15}
# plt.rc('font', **font)
# plt.rcParams.update({'pdf.use14corefonts' : True,'text.usetex' : True})

def testCompile(filename:str,query:str,evidence:List[str],compileF):
    tbn = readTBNXML(filename)
    tacT = compileF(tbn,query,evidence,inst=True)
    tacF = compileF(tbn,query,evidence,inst=False)
    if not evidence:
        print(f'Pr(y) var {query}:',tacT.forward([]))
        print(f'Pr(!y) var {query}:',tacF.forward([]))
        return
    evidence0 = evidence[0]
    evidence1 = evidence[:]
    evidence1.remove(evidence0)
    tacET = compileF(tbn,evidence[0],evidence1,inst=True)
    tacEF = compileF(tbn,evidence[0],evidence1,inst=False)
    print('evidence var:',evidence)
    print('query var:',query)
    for e in product(*([[True,False]]*len(evidence))):
        if e[0]:
            prE = tacET.forward(e[1:])
        else:
            prE = tacEF.forward(e[1:])
        pr = {}
        pr[True] = tacT.forward(e)
        pr[False] = tacF.forward(e)
        print(f'evidence {evidence}:',e)
        print('Pr(e):',prE)
        for y in [True,False]:
            print(f'y {query}:',y)
            print('Pr(y,e)',pr[y])
            print('Pr(y|e)',pr[y]/prE)
        print('*')
    print('***')

def testCompileNormalized(filename:str,query:str,evidence:List[str],compileF):
    tbn = readTBNXML(filename)
    tacT = compileF(tbn,query,evidence,inst=True,normalized=True)
    tacF = compileF(tbn,query,evidence,inst=False,normalized=True)
    if not evidence:
        print(f'Pr(y) var {query}:',tacT.forward([]))
        print(f'Pr(!y) var {query}:',tacF.forward([]))
        return

    print('evidence var:',evidence)
    print('query var:',query)
    for e in product(*([[True,False]]*len(evidence))):
        pr = {}
        pr[True] = tacT.forward(e)
        pr[False] = tacF.forward(e)
        print(f'evidence {evidence}:',e)

        for y in [True,False]:
            print(f'y {query}:',y)
            print('Pr(y|e)',pr[y])
        print('*')
    print('***')

# now only handles when input size is 1 or 2
def genDataTAC(tac:ArithmeticCircuit):
    steps1 = 100
    steps2 = 100
    n = tac.inputsNum
    if n == 1:
        xs = np.linspace(0,1,num=steps1)
        ys = []
        for x in xs:
            y = tac.forwardSoft([x])
            ys.append(y)
        return xs,np.array(ys)

    elif n == 2:
        xs = np.linspace(0,1,num=steps2)
        ys = np.linspace(0,1,num=steps2)
        X, Y = np.meshgrid(xs, ys)
        ys = np.zeros(X.shape)
        for i in range(steps2):
            for j in range(steps2):
                p = X[i, j]
                p2 = Y[i, j]
                ys[i, j] = tac.forwardSoft([p,p2])

        return X,Y,ys

def draw2D(xs,ys,fileName='',title=''):

    fig, ax = plt.subplots(1, 1, sharey=True)

    # ax.plot(xs, ys, 'ro-', label='rec')
    plt.plot(xs, ys, 'ro-', label='rec')

    # ax.set(xlabel=r'indicator X', ylabel='Pr(Y)',
    ax.set(xlabel=r'$\lambda$', ylabel='$f(\lambda)$',
           title=title)
    if fileName:
        plt.savefig(fileName)
        plt.show()
    else:
        plt.show()


def draw3D(X,Y,Z,fileName='',title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe.
    # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    ax.plot_surface(X, Y, Z, cmap='coolwarm',linewidth=0,antialiased=False)
    ax.set(xlabel=r'$\lambda_1$', ylabel='$\lambda_2$',zlabel='Pr*(B)',
           title='')
    # ax.contourf(X, Y, Z, cmap=cm.coolwarm)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)

    # pylab.savefig("plot-orig-%d.png" % index)
    # pylab.savefig("plot-orig-%d.pdf" % index)

    # plt.show()
    if fileName:
        fig.savefig(fileName)
        plt.show()
    else:
        plt.show()
'''
bn = readTBNXML('samples/abc.xml')
# bn.nodes['A'].table = genTable(2)
bn.nodes['B'].table.table = genTable(2)
bn.nodes['C'].table.table = genTable(2)

tbn = bn2Tbn(bn,['B'],0.5)
tac = polyTreeTAC(tbn,'B',['A','C'],normalized=True)

X,Y,Z = genDataTAC(tac)
# draw3D(X,Y,Z,'abc.pdf')
draw3D(X,Y,Z)
exit(0)
'''
# for y in 'ABCDEF':
#     testCompile('samples/polytree.xml',y,['C','F',],polyTreeAC)
    # testCompileNormalized('samples/polytree.xml',y,['C','F',],polyTreeTAC)
    # print()

'''
for i in [1,2,4,8,16,32]:
# for i in [4]:
    # tbn = gen2DChain(i)
    tbn = genChain2D(i,0)
    # d = tbn.toDot()
    # d.view('2in')
    for n in tbn.nodes.values():
        # n.testing = False
        pass
    tac = polyTreeTAC(tbn,'Y',['X','X2'],normalized=True)
    tac2file(tac,'testcases/2inchain'+str(i))
exit(0)
# tac = file2tac('chains/chain2')
# tacToPdf(tac,'chains/chain2',showValue=False,showID=False)
'''
# test for the children attribute
'''
tbn = readTBNXML("samples/cancer.xml")
d=tbn.toDot()
d.render('tbn',view=True)
exit(0)
'''
'''
# test for compile analysis
tbn = readTBNXML("samples/cancer.xml")
for n in tbn.nodes:
    print(n, tbn.nodes[n].children)
preprocess_topo(tbn, "A", ["C", "D"])
exit(0)
'''

# parameters of the rectangle net
NUM_ROWS = 3
NUM_COLS = 3

rectangle_filename = "samples/rectangleModelSmall.xmlbif"
rectangle_tbn = readTBNXML(rectangle_filename)

query = "Height"
evidences = []
for i in range(NUM_ROWS):
    for j in range(NUM_COLS):
        cell = "Out" + "_" + str(i) + "_" + str(j)
        evidences.append(cell)

#query = "Out_1_1"
#evidences = ["Height"]

arguments = sys.argv[1:]
if len(arguments) != 0 and arguments[0] == "test":
    test_tbn = readTBNXML("samples/cancer.xml")
    test_tac = compileTAC_ve(test_tbn, "A",["B"] ) #normalized p true
    print("I am here.")
    tac2file(test_tac,'samples/cancertopo')
else:
    rectangle_tac = compileTAC_ve(rectangle_tbn, query, evidences) #normalized=True
    tac2file(rectangle_tac,'samples/rectangleSmall')
    print("Finished rectangle with evidence at root")

exit(0)


bn = readTBNXML('samples/abc.xml')
tbn = bn2Tbn(bn, ['B'], 0.7, randomTable=True)
tbn.nodes['A'].table.table = genTable(1)
tbn.nodes['C'].table.table = genTable(2)
# d = tbn.toDot()
# d.view('abc')
tac = polyTreeTAC(tbn,'A',['D','C'],normalized=True)
tac2file(tac,'testcases/abc')
# tacToPdf(tac,'testcases/abc',showID=False,showValue=False)
X,Y,Z = genDataTAC(tac)
draw3D(X,Y,Z)
exit(0)


'''
tbn = readTBNXML('samples/polytree.xml')
tac = polyTreeTAC(tbn,'D',['C','F'],inst=True,normalized=False)
tacToPdf(tac,'tac3.pdf',showID=True)
exit(0)
# bn = readTBNXML('samples/test2.xml')
# tbn = bn
# tbn = bn2Tbn(bn,['Y'],0.5)
tbn = genChain(2)
tbn = addChild(tbn,'X2',np.array([[0.1,0.9],[0.9,0.1]]),'T0')

'''
tbn = genChain(1)
d=tbn.toDot()
d.render('tbn',view=True)
exit(0)
tbn = readTBNXML('samples/poster.xml')
# tbn.nodes['Y'].testing = True
# table = tbn.nodes['Y'].table
# tbn.nodes['Y'].tableP = deepcopy(table)
# tbn.nodes['Y'].tableN = deepcopy(table)
# tbn.nodes['Y'].thres = np.array([0.5,0.5])
tac = polyTreeTAC(tbn,'Y',[],inst=True,normalized=True)
tacToPdf(tac,'poster',showValue=False)
exit(0)
tbn = genChain2D(5,2)
tac = polyTreeTAC(tbn,'Y',['X','X2'],inst=True,normalized=True)
tac2file(tac,'samples/chain52in')
exit(0)
# d = tbn.toDot()
# d.render('C:\\Users\\Rel\\Desktop\\AAAI19\\Graphs\\tbn1')
# tbn = adHocChain()
# tbn = bn
tbn = genChain(2)
tbn = addChild(tbn,'X2',np.array([[0.1,0.9],[0.9,0.1]]),'T0')
d = tbn.toDot()
d.render('C:\\Users\\Rel\\Desktop\\AAAI19\\Graphs\\tbn3')

for n in tbn.nodes.values():
    # n.testing = False
    pass

# tac = polyTreeTAC(tbn,'Y',['X'],inst=True,normalized=True)
tac = polyTreeTAC(tbn,'Y',['X','X2'],inst=True,normalized=True)
for n in tbn.nodes.values():
    n.testing = False
    pass
# tac2 = polyTreeTAC(tbn,'Y',['X'],inst=True,normalized=True)
tac2 = polyTreeTAC(tbn,'Y',['X','X2'],inst=True,normalized=True)

# d = tbn.toDot()
# d.render('C:\\Users\\Rel\\Desktop\\AAAI19\\Graphs\\tbn2')
X,Y,Z = genDataTAC(tac)
X2,Y2,Z2 = genDataTAC(tac2)
# X,Y = genDataTAC(tac)
# X2,Y2 = genDataTAC(tac2)

fileName = 'C:\\Users\\Rel\\Desktop\\AAAI19\\Graphs\\2dtac4.pdf'
fileName2 = 'C:\\Users\\Rel\\Desktop\\AAAI19\\Graphs\\2dac4.pdf'
# fileName = ''

draw3D(X,Y,Z,fileName=fileName,title='TAC')
draw3D(X2,Y2,Z2,fileName=fileName2,title='AC')

# draw2D(X,Y,fileName=fileName,title='TAC')
# draw2D(X2,Y2,fileName=fileName2,title='AC')


exit(0)
if False:
# if True:
# if tac.inputsNum == 1:
     xs,ys = genDataTAC(tac)
     # xs,ys = getData(tac)
     # ys = [ys[i]/(ys2[i]+ys[i]) for i in range(len(ys))]
     draw2D(xs,ys)
else:
     X,Y,Z = genDataTAC(tac)
     # import pdb;pdb.set_trace()
     X2,Y2,Z2 = genDataTAC(tac2)
     # X,Y,Z = genData2D(tac)
     # X2,Y2,Z2 = genData2D(tac2)
     Z = Z/(Z+Z2)
     draw3D(X,Y,Z)

# tacToPdf(tac,'tac.pdf',showID=True)




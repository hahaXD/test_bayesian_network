from TestingBayesianNetwork import *
from readTBNXML import *
from readBNXML import *
from utils import genDiagram,file2tac,tac2file,findNode,printTBN
from ArithmeticCircuit import *
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import random
from compile import polyTreeTAC
from testcases import *

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d



# generate TBN: X->T1->T2->..->T_n->Y where T_i is testing node
def genChain(testNum,test=True,thres=-1):
    tbn = readTBNXML('samples/chain.xml')

    tNode = deepcopy(tbn.nodes['T'])
    xNode = deepcopy(tbn.nodes['X'])
    yNode = deepcopy(tbn.nodes['Y'])

    tNode.tableP = deepcopy(tNode.table)
    tNode.tableN = deepcopy(tNode.table)
    tNode.thres = np.zeros(tNode.table.table.shape[1:])
    tNode.thres.fill(0.1)

    prevTNode = tNode
    nodes = {}
    nodes['X'] = xNode
    nodes['Y'] = yNode
    prevTNode.name = 'T0'
    prevTNode.testing = True
    prevTNode.parents = ['X']
    prevTNode.table.varName = ['T0','X']
    prevTNode.table.varNum = [2,2]


    prevTNode.tableP = prevTNode.table
    prevTNode.tableN = deepcopy(prevTNode.table)

    curTNode = deepcopy(tNode)
    nodes['T0'] = tNode

    thresShape = prevTNode.thres.shape
    tableShape = prevTNode.tableP.table.shape
    prevTNode.thres = genProb(thresShape)
    prevTNode.thres[1] = prevTNode.thres[0]
    #
    # prevTNode.thres[0] = 0.75
    # prevTNode.thres[1] = 0.75

    prevTNode.tableP.table = genTable(len(tableShape),positive=True,odd = False)
    prevTNode.tableN.table = genTable(len(tableShape),positive=False,odd = False)
    # prevTNode.tableP.table = genProb(tableShape)
    # prevTNode.tableN.table = genProb(tableShape)

    for i in range(1,testNum):
        curTNode.table = curTNode.tableP
        curTNode.parents = [prevTNode.name]

        curTNode.tableP.varName = ['T'+str(i),'T'+str(i-1)]
        curTNode.tableN.varName = ['T'+str(i),'T'+str(i-1)]


        curTNode.name = 'T'+str(i)
        nodes['T'+str(i)] = curTNode

        curTNode.thres= genProb(thresShape)

        # curTNode.thres[0] = 0.7
        curTNode.thres[1] = curTNode.thres[0]
        # print(curTNode.thres)

        # if thres>=0:
        #     curTNode.thres[0] = thres
        #     curTNode.thres[1] = curTNode.thres[0]
            # curTNode.thres[1] = 1 - curTNode.thres[0]

        curTNode.tableP.table = genTable(len(tableShape),positive=True, odd = i%2 == 1)
        curTNode.tableN.table = genTable(len(tableShape),positive=False,odd = i%2 == 1)

        # curTNode.tableP.table = genProb(tableShape)
        # curTNode.tableN.table = genProb(tableShape)


        prevTNode = curTNode
        curTNode = deepcopy(prevTNode)

    yNode.parents = [prevTNode.name]
    yNode.table.varName = ['Y','T'+str(testNum-1)]
    tbn.nodes = nodes

    if not test:
        for n in tbn.nodes.values():
            if n.testing:
                n.table = deepcopy(n.tableP)
                n.testing = False
    '''
    for n,v in tbn.nodes.items():
        print(v.name)
        print(v.parents)
        if v.testing:
            print(v.tableP,'tableP')
            print(v.tableN,'tableN')
            print(v.thres,'thres')
        else:
            print(v.table,'table')
        pass
    '''
    return tbn




def getData(tac,topo=False):
    pX = findNode(ParameterNode,'X',tac.root)
    pnX = findNode(ParameterNode,'!X',tac.root)
    ps = np.linspace(0,1.0,num=50)
    ys = []
    for p in ps:
        pX.value = p
        pnX.value = 1-p
        if topo:
            v = tac.forward([])
        else:
            v = tac.forward([])
        ys.append(v)
        # print(v)
    return ps,ys

def adHocChain():
    tbn = readTBNXML('samples/adhocChain.xml')
    n  = tbn.nodes['Y']
    n.thres = np.array([0.75,0.75])
    n.testing = True

    n.tableP = deepcopy(n.table)
    n.tableN = deepcopy(n.table)

    n.tableN.table = np.array([[0.9,0.1],[0.1,0.9]])
    n.tableP.table = np.array([[0.1,0.9],[0.9,0.1]])

    n  = tbn.nodes['Q']
    n.table.table = np.array([[1,0],[0,1]])

    # for n in tbn.nodes.values():
    #     if not n.testing:
    #         print(n.table)
    #     else:
    #         print('!')
    #         print(n.tableP)
    #         print(n.tableN)
    #         print('!')
    return tbn


# chainNum = 4

# midNode = 'T0'
# tbn = genChain(chainNum,test=True)

# tbn = genChain(chainNum,test=False)

# tbn2 = adHocChain()
# tac = compileTAC(tbn2,'Q','',inst=True)
# tac2 = compileTAC(tbn2,'Q','',inst=False)

# tac = compileTAC(tbn,'Y','',inst=True)

# tac2 = compileTAC(tbn,'Y','',inst=False)

# tacP = compileTAC(tbn,midNode,'',inst=True)
# tacT = compileTAC(tbn,midNode,'',inst=False)

# tbn.nodes[midNode].testing = False
# tacP2 = compileTAC(tbn,midNode,'',inst=True)

# print(tac.root)
# print('tac forward',tac.forward([]))
# ps,ys = getData(tac)
# ps2,ys2 = getData(tacP)
# ps3,ys3 = getData(tacT)


# for y1,y2 in zip(ys,ys2):
#     print(y1+y2)
# d = genDiagram(tacP2.root,showID=False)
# d.render('chain2.pdf',view=True)

# fig, (ax1,ax2) = plt.subplots(1,1,sharey=True)
# fig, ax = plt.subplots(1,1,sharey=True)
#
# ax.plot(ps,ys,'ro-',label='Y')
#
# ax.set(xlabel=r'value of \theta_x', ylabel='output of y',
#        title=f'chainNum:{chainNum}')
#
# ax.plot(ps2,ys2,'go-',label=midNode)
#
# ax.plot(ps3,ys3,'bo-',label='!'+midNode)
#
# plt.legend()
# plt.show()

def testDup(node,s):
    if node in s and type(node) != ParameterNode:
        # print(type(node),node.label)
        print(type(node),'nodeid:',node.id)
        return True
    else:
        s.add(node)
        if type(node) == MultiplyNode or type(node) == AddNode:
            for c in node.children:
                if testDup(c,s):
                    return True
        return False

def thresTest():
    chainNum = 8

    for x in np.linspace(0,1.0,num=10):
        midNode = 'T0'
        tbn = genChain(chainNum,test=True,thres = x)
        # tbn.nodes['T1'].tableP.table = np.array([[0.9,0.1],[0.1,0.9]])

        # tbn = genChain(chainNum,test=False)

        # tbn2 = adHocChain()
        # tac = compileTAC(tbn2,'Q','',inst=True)
        # tac2 = compileTAC(tbn2,'Q','',inst=False)

        tac = compileTAC(tbn,'Y','',inst=True)
        # tacP = compileTAC(tbn,'Y','',inst=False)
        tac2file(tac,'samples\chain2')
        tac = file2tac('samples\chain2')
        d = genDiagram(tac.root,showID=True)
        d.render('chain3.pdf',view=True)
        # testDup(tac.root,set())

        # tac2 = compileTAC(tbn,'Y','',inst=False)

        # tacP = compileTAC(tbn,midNode,'',inst=True)
        # tacT = compileTAC(tbn,midNode,'',inst=False)

        # print(tac.root)
        # print('tac forward',tac.forward([]))
        ps,ys = getData(tac,topo=True)
        # ps2,ys2 = getData(tacP)
        # ps2,ys2 = getData(tac,topo=True)
        # ps3,ys3 = getData(tacT)

        # sum = 0
        # for y1,y2 in zip(ys,ys2):
        #     sum+=(1-y1-y2)**2
        # print('sum:',sum)
        # for i in range(len(ys)):
        #     if ys[i] != ys2[i]:
        #         print('unmatched:',ps2[i])
        # fig, (ax1,ax2) = plt.subplots(1,1,sharey=True)
        fig, ax = plt.subplots(1,1,sharey=True)

        ax.plot(ps,ys,'ro-',label='rec')

        ax.set(xlabel=r'value of \theta_x', ylabel='output of y',
               title=f'chainNum:{chainNum}')

        # ax.plot(ps2,ys2,'go-',label='topo')

        # ax.plot(ps3,ys3,'bo-',label='!'+midNode)

        plt.legend()
        plt.show()

# thresTest()


# print(testDup(tac.root,set()))

def adHocChain2():
    tbn = readTBNXML('samples/adhocChain2.xml')
    n  = tbn.nodes['T']
    n.thres = np.array([0.75,0.75])
    n.testing = True

    n.tableP = deepcopy(n.table)
    n.tableN = deepcopy(n.table)

    n.tableN.table = np.array([[0.9,0.1],[0.1,0.9]])
    n.tableP.table = np.array([[0.1,0.9],[0.9,0.1]])

    # n  = tbn.nodes['Q']
    # n.table.table = np.array([[1,0],[0,1]])

    # for n in tbn.nodes.values():
    #     if not n.testing:
    #         print(n.table)
    #     else:
    #         print('!')
    #         print(n.tableP)
    #         print(n.tableN)
    #         print('!')
    return tbn


# tbn = genChain(1)
# tbn = adHocChain2()
# tac = compileTAC(tbn,'Y','',inst=True)
# tacP = compileTAC(tbn,'Y','',inst=False)
# tac2file(tac,'samples\chain2')
# tac = file2tac('samples\chain2')
# d = genDiagram(tac.root)
# d.render(f'chain2.pdf',view=True)


# Fixing random state for reproducibility
# np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


# x2Num n: X2 will be added as the parent of Tn
def genChain2D(chainNum,x2Num):
    assert(x2Num >= 0 and x2Num <= chainNum-1)
    tbn = genChain(chainNum)
    tNodeName = 'T'+str(x2Num)
    tNode = tbn.nodes[tNodeName]
    x2Node = deepcopy(tbn.nodes['X'])
    x2Node.name = 'X2'
    tbn.nodes['X2'] = x2Node
    x2Node.table.varName = ['X2']

    tNode.parents.append('X2')
    tNode.tableP.varName = [tNodeName,('T'+str(x2Num-1)) if x2Num > 0 else 'X','X2']
    tNode.tableP.varNum = [2]*3
    tNode.tableP.table = genTable(3)

    tNode.table = tNode.tableP

    tNode.tableN.varName = [tNodeName,('T'+str(x2Num-1)) if x2Num > 0 else 'X','X2']
    tNode.tableN.varNum = [2]*3
    tNode.tableN.table = genTable(3)

    tNode.thres = genProb([2,2])
    tNode.thres[1] = tNode.thres[0]

    return tbn

def genData2D(tac):
    pX = findNode(ParameterNode, 'X', tac.root)
    pnX = findNode(ParameterNode, '!X', tac.root)
    pX2 = findNode(ParameterNode, 'X2', tac.root)
    pnX2 = findNode(ParameterNode, '!X2', tac.root)

    # ps = np.linspace(0, 1.0, num=50)
    # ps2 = np.linspace(0, 1.0, num=50)
    ps = []
    ps2 = []
    ys = []
    n = 20 # number of steps in one axis
    xs = np.linspace(0,1.0,num=n)
    ys = np.linspace(0,1.0,num=n)
    X,Y = np.meshgrid(xs,ys)
    ys = np.zeros(X.shape)
    for i in range(n):
        for j in range(n):
            p = X[i,j]
            p2 = Y[i,j]
            pX.value = p
            pnX.value = 1 - p
            pX2.value = p2
            pnX2.value =  1- p2
            ys[i,j] = tac.forwardTopo([])


    # for p,p2 in zip(X.flatten(),Y.flatten()):
    #     pX.value = p
    #     pnX.value = 1 - p
    #     pX2.value = p2
    #     pnX2.value =  1- p2
    #     v = tac.forwardTopo([])
    #     ps.append(p)
    #     ps2.append(p2)
    #     ys.append(v)
        # print(v)
    # return np.array(ps), np.array(ps2), np.array(ys)
    return X,Y,ys

def compareResult(Z,Z2):
    Z = np.array(Z)
    Z2 = np.array(Z2)
    ct = 0
    zs = Z.flatten()
    zs2 = Z2.flatten()
    diffSum = 0
    for i in range(len(zs)):
        if zs[i]==zs2[i]:
            ct+=1
        else:
            diff = zs[i]-zs2[i]
            print('zs:%lf zs2:%lf'%(zs[i],zs2[i]))
            print('difference:%lf'%diff)
            diffSum += diff*diff
    print('Data Size:%d Match percent:%lf diffSum:%lf'%(len(zs),ct*100/len(zs),(diffSum/len(zs))**0.5))
# tbn = adHocChain()

'''
tbn = genChain(10)
tac = compileTAC(tbn,'Y',['X'],inst=True)
tac.pruneNode()
tac2 = polyTreeTAC(tbn,'Y',['X'],inst=True)
d = genDiagram(tac.root,showID=True)
d2 = genDiagram(tac2.root,showID=True)
# d.view('chaintest.pdf')
# d2.view('chaintest2.pdf')
X, Z = getData(tac)
X2,Z2 = getData(tac2)
compareResult(Z,Z2)
print(len(tac.seq))
print(len(tac2.seq))


tbn = genChain2D(30,0)
# printTBN(tbn)

tac = compileTAC(tbn,'Y',['X','X2'],inst=True)
tac.pruneNode()

print('len1:',len(tac.seq))
# d = genDiagram(tac.root,showID=False)
# d.view('fuck.pdf')
X, Y, Z = genData2D(tac)

tac = polyTreeTAC(tbn,'Y',['X','X2'],inst=True)
prevLen = len(tac.seq)
tac.pruneNode()
newLen = len(tac.seq)
while newLen != prevLen:
    prevLen = newLen
    tac.pruneNode()
    newLen = len(tac.seq)
print('len2:',len(tac.seq))
# tac = compileTAC(tbn,'Y',['X','X2'],inst=True)
X2, Y2, Z2 = genData2D(tac)

# exit(0)
# '''

# '''
# '''

'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
# Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1,cmap='viridis')
# ax.contourf(X, Y, Z, cmap=cm.coolwarm)

plt.show()

'''


import argparse
from ArithmeticCircuit import setUseSigmoid
from utils import file2tac
import re

parser = argparse.ArgumentParser()
parser.add_argument('-s','--sigmoid',help='Use sigmoid rather than step function if specified',action='store_true')
parser.add_argument('-d','--data',help='Data to feed into the tac',nargs='*',type=float)
parser.add_argument('tac',help='Tac input filename without extension',type=str)
parser.add_argument('-f','--file',help='Data input from file,one set at a line,each number separated by tab',type=str)
parser.add_argument('-r','--raw',help='Only output tac values if specified',action='store_true')

args = parser.parse_args()
# prin(args)

tac = file2tac(args.tac)
setUseSigmoid(args.sigmoid)
datas = []
if args.file:
    with open(args.file,'r') as f:
        for l in f:
            data = list(map(float,re.split(r'\s',l.strip())))
            datas.append(data)
elif args.data:
    datas.append(args.data)

ct = 0
if not args.raw:
    print('Use sigmoid:',args.sigmoid)
if datas:
    for d in datas:
        r = tac.forwardSoft(d)
        if args.raw:
            print(r)
        else:
            print('Test case #',ct)
            print('Input: ',d)
            print('Output: ',r)
        ct+=1
else:
    from sys import stdin

    for l in stdin:
        if not l.strip():
            break
        d = list(map(float,re.split(r'\s',l.strip())))
        r = tac.forwardSoft(d)
        if args.raw:
            print(r)
        else:
            print('Test case #',ct)
            print('Output: ',r)
        ct+=1




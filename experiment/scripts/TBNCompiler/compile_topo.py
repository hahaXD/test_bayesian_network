
def preprocess_topo(tbn: TestingBayesianNetwork, y: str, evidences: list[str], inst: bool=True):
  	for n in tbn.nodes.values():
        n.evidence = set() # clear the evidence

    InDeg = {n.name : len(n.parents) for n in tbn.nodes.values()} # a dictionary that maps nodes to num_of_parents
    roots = [n.name for n in tbn.nodes.values() if InDeg[n] == 0] # roots are nodes with no parents

    seq = []
    # sort TBN nodes in topological order
    while roots:
        n = roots.pop()
        nNode = tbn.nodes[n]
        if n != y:
            seq.append(deepcopy(nNode))
        for c in n.children:
            InDeg[c] -= 1
            if InDeg[c] == 0:
                roots = roots + [c]
    
    seq.append(deepcopy(tbn.nodes[y])) # append query to the end of topo order

    # initialize the indicator nodes
	inputs = []
	inputsD = {}
	for e in evidences:
		eNode = tbn.nodes[e]
		for i,state in enumerate(eNode.states):
			inputs.append(IndicatorNode(label=e, inst=state))
			inputsD[(e, i)] = inputs[-1]

	# for each node, intialize the CPT table
	for n in seq: 
		if n in evidences:
			n.evidence.add(n)
		for p in node.parents:
			node.evidence.union(tbn.nodes[p].evidence)

		labels = []
	    for v in n.table.varName[:]:
	        vNode = tbn.nodes[v]
	        if len(vNode.states) == 2:
	            labels.append([ v + ' ', '!' + v + ' '])
	        else:
	            labels.append([ v + '=' + state + ' ' for state in vNode.states])

	    if len(labels) > 1:
            for label in labels[0]:
                label += '\| '

        # !! for testing node, table equals to the positive table(tableP)
        symbolTable = np.zeros(n.table.table.shape, dtype=CPTType)

        thresCache = {}
        if not n.testing:
            for pos,prob in np.ndenumerate(n.table.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=labels[i][v]

                if n.name in e: # for evidence, multiply indicator into corresponding CPT entry
                    symbolTable[pos] = MultiplyNode(children=[ParameterNode(value=prob,label=label), inputsD[(n,pos[0])]]) 

                elif n.name == y: # for query, we need to pass in an instantiation
                    symbolTable[pos] = ParameterNode(value=prob if inst == n.states[pos[0]] else 0, label=label)
                else:
                    symbolTable[pos] = ParameterNode(value=prob, label=label)

        else: # for testing nodes, what if testing node is query?
            for pos,prob in np.ndenumerate(n.tableP.table):
                label = ''
                for i,v in enumerate(pos):
                    label+=labels[i][v]

                labelT = ''
                for i,v in enumerate(pos):
                	if i > 0:
                    	labelT+=labels[i][v]
                if not pos[1:] in thresCache:
                    thresCache[pos[1:]] =  ParameterNode(value=n.thres[pos[1:]],label='*'+labelT+' Thres')

                t = thresCache[pos[1:]]

                symbolTable[pos] = TestingNode(x=None,
                                               thetaD=ParameterNode(value = n.tableN.table[pos],label=label+' -'),
                                               thetaU=ParameterNode(value = prob,label=label+' +'),
                                               thres=MultiplyNode(children=[t]),
                                               # label=f'{n.table.varName}')
                                               label=label)
        n.table.table = symbolTable
        # update the CPT table
    print("Finished initializing symbol table.")

   	# build elimination tree
    t = list(map(lambda x:set(x.table.varName),nodes))
    #print("t: %s" %t)


















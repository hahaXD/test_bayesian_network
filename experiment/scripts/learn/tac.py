#!/usr/bin/env python

from collections import defaultdict
import math
import random
import numpy as np
import tensorflow as tf

# globals
gamma_min =  50000
gamma_max =  50000.1
thresh_min = 1
thresh_max = 1.0001
compiler = "jason"

assert gamma_min < gamma_max
assert thresh_min < thresh_max

def step(x,T,p,n):
    cond = tf.greater_equal(x,T)
    ones = tf.ones(tf.shape(x),dtype=tf.float64)
    out = tf.where(cond,p*ones,n*ones)
    return out

def _my_mean_squared_error(labels,predictions):
    return tf.reduce_mean(tf.squared_difference(labels,predictions))

def _my_regularizer(weights):
    return tf.reduce_sum(weights*weights)

class TacNode:

    # typedef's
    LITERAL,SUM,PRODUCT,TEST,NORMALIZE = "L","+","*","?","Z"
    NODE_TYPES = [LITERAL,SUM,PRODUCT,TEST,NORMALIZE]

    def __init__(self,node_type,node_id,literal=None,children=None):
        self.node_type = node_type
        self.node_id = node_id
        self.literal = literal        # if node is LITERAL
        if children is not None:
            self.size = len(children) # if node is SUM or PRODUCT or TEST
        self.children = children      # if node is SUM or PRODUCT or TEST
        self._bit = False

    def __iter__(self,first_call=True,clear_data=False):
        """post-order (children before parents) generator"""
        if self._bit: return
        self._bit = True

        if not self.is_terminal():
            for child in self.children:
                for node in child.__iter__(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)

    def __repr__(self):
        if self.is_literal():
            st = '%d %s %d' % (self.node_id,self.node_type,self.literal)
        else:
            children_st = " ".join(str(child.node_id)
                                   for child in self.children)
            st = '%d %s %d %s' % (self.node_id,self.node_type,
                                  self.size,children_st)
        return st

    def clear_bits(self,clear_data=False):
        """Recursively clears bits.  For use when recursively navigating an
        arithmetic circuit by marking bits.

        Set clear_data to True to also clear data."""
        if self._bit is False: return
        self._bit = False
        if clear_data: self.data = None
        if not self.is_terminal():
            for child in self.children:
                child.clear_bits(clear_data=clear_data)

    def is_terminal(self):
        return self.node_type == TacNode.LITERAL

    def is_literal(self):
        return self.node_type == TacNode.LITERAL

    def is_sum(self):
        return self.node_type == TacNode.SUM

    def is_product(self):
        return self.node_type == TacNode.PRODUCT

    def is_test(self):
        return self.node_type == TacNode.TEST

    def is_normalize(self):
        return self.node_type == TacNode.NORMALIZE

    def tf_tac(self,lmap,use_sigmoid=True):
        """Generates a tensorflow testing arithmetic circuit.
        """
        for node in self:
            if node.is_literal():
                lit_id = node.literal
                data = lmap[lit_id].tf_node
            elif node.is_sum():
                sum_node = node.children[0].data
                for child in node.children[1:]:
                    sum_node = sum_node + child.data
                data = sum_node
            elif node.is_product():
                prod_node = node.children[0].data
                for child in node.children[1:]:
                    prod_node = prod_node * child.data
                data = prod_node
            elif node.is_test():
                prue = node.children[0].data
                pre = node.children[1].data
                pr = prue/pre
                if compiler == "haiying":
                    th = node.children[4].data
                    if len(node.children) == 6:
                        gamma = node.children[5].data
                    else:
                        gamma = 8
                    i1 = node.children[2].data # ACACAC
                    i0 = node.children[3].data # ACACAC
                elif compiler == "jason":
                    th = node.children[2].data
                    if len(node.children) == 6:
                        gamma = node.children[5].data
                    else:
                        gamma = 8
                    i1 = node.children[4].data
                    i0 = node.children[3].data
                else:
                    raise Exception("invalid compiler: " + compiler)
                data = (i1-i0)*tf.sigmoid(gamma*(pr-th))+i0
            elif node.is_normalize():
                sum_node = node.children[0].data
                for child in node.children[1:]:
                    sum_node = sum_node + child.data
                data = node.children[0].data/sum_node
            node.data = data
        return self.data

    @staticmethod
    def ac_read(filename):
        with open(filename,'r') as f:
            lines = f.readlines()
        node_count = len(lines)
        node_map = {}
        for line in lines:
            line = line.strip().split(" ")
            node_id = int(line[0])
            node_type = line[1]
            assert node_type in TacNode.NODE_TYPES
            if node_type == TacNode.LITERAL:
                literal = int(line[2])
                children = None
            else:
                literal = None
                size = int(line[2])
                children = [ int(child_id) for child_id in line[3:] ]
                children = [ node_map[child_id] for child_id in children ]
                assert size == len(children)
            node = TacNode(node_type,node_id,
                           literal=literal,children=children)
            node_map[node_id] = node
        return node

    @staticmethod
    def ac_save(circuit,filename):
        with open(filename,'w') as f:
            for node in circuit:
                f.write(str(node))
                f.write("\n")

    def eval_all(self,subst):
        for node in self:
            print("====================")
            print(node)
            print(node.data.eval(subst))

class Literal:

    # typedef's
    PARAMETER,INDICATOR = "p","i"
    PARAM_THRESHOLD,PARAM_GAMMA = "t","g"

    def __init__(self,lit_type,lit_id,index,
                 weight=None,param_type=None, # for parameters
                 var=None,val=None,           # for indicators
                 st=None):
        self.lit_type = lit_type
        self.param_type = param_type
        self.lit_id = lit_id
        self.index = index
        self.weight = weight
        self.var = var
        self.val = val
        self.st = st
        self.covarying = set()

    def __repr__(self):
        if self.lit_type == Literal.PARAMETER:
            st = "%d %s %.6f %s" % (self.lit_id,self.lit_type, \
                                       self.weight,self.st)
        elif self.lit_type == Literal.INDICATOR:
            st = "%d %s %s" % (self.lit_id,self.lit_type,self.st)
        return st

    def is_parameter(self):
        return self.lit_type == Literal.PARAMETER

    def is_indicator(self):
        return self.lit_type == Literal.INDICATOR

    def is_threshold(self):
        return self.is_parameter() and \
            self.param_type == Literal.PARAM_THRESHOLD

    def is_gamma(self):
        return self.is_parameter() and \
            self.param_type == Literal.PARAM_GAMMA

    @staticmethod
    def jason_parse_value(st):
        """temporary hack to read Jason's lmap format"""
        return st.replace("state","")

    @staticmethod
    def lmap_read(filename):
        with open(filename,'r') as f:
            lines = f.readlines()
        lit_count = len(lines)

        literal_map = {}
        indicator_map = defaultdict(defaultdict)

        pcount,icount = 0,0
        var_count = 0
        for line in lines:
            line = line.strip().split(" ")
            lit_id = int(line[0])
            lit_type = line[1]
            if lit_type == Literal.PARAMETER:
                weight = float(line[2])
                st = " ".join(line[3:])
                st = Literal.jason_parse_value(st)
                param_type = None
                if st.lower().endswith("thres"):
                    param_type = Literal.PARAM_THRESHOLD
                elif st.lower().endswith("gamma"):
                    param_type = Literal.PARAM_GAMMA
                lit = Literal(lit_type,lit_id,pcount,\
                              weight=weight,param_type=param_type,st=st)
                pcount += 1
                literal_map[lit_id] = lit
            elif lit_type == Literal.INDICATOR:
                var = var_count
                var_count += 1
                st = " ".join(line[2:])
                st = st.strip()
                st = Literal.jason_parse_value(st)
                lit_var,lit_val = st.split("=")
                lit = Literal(lit_type,lit_id,icount,var=var,val=lit_val,st=st)
                icount += 1
                indicator_map[var][lit_val] = lit
                literal_map[lit_id] = lit

        Literal.find_covarying_parameters(literal_map)
        return literal_map,indicator_map

    @staticmethod
    def lmap_write(lmap,weights,filename):
        with open(filename,'w') as f:
            for lit_id in sorted(lmap.keys()):
                lit = lmap[lit_id]
                if lit_id < 0: continue
                if lit.is_indicator():
                    f.write("%d i %s\n" % (lit_id,lit.st))
                elif lit.is_parameter():
                    indices = [ lmap[lit_id].index for lit_id in \
                                lit.covarying ]
                    if lit.is_gamma():
                        tau = math.exp(-weights[lit.index])
                        p = 1.0 / (1.0 + tau)
                        weight = gamma_min + (gamma_max-gamma_min)*p
                    elif len(indices) == 1:
                    	"""
                    	tau = math.exp(-weights[lit.index])
                    	weight = tau / (1.0 + tau)
                    	"""
                    	tau = math.exp(-weights[lit.index])
                    	p = 1.0 / (1.0 + tau)
                    	weight = thresh_min + (thresh_max-thresh_min)*p
                    else:
                        z = 1.0
                        for index in indices[1:]:
                            z = z + math.exp(-weights[index])
                        if lit.index == indices[0]:
                            weight = 1.0 / z
                        else:
                            weight = math.exp(-weights[lit.index]) / z
                    f.write("%d p %.8g %s\n" % (lit_id,weight,lit.st))
                else:
                    # should not reach here
                    pass

    @staticmethod
    def find_covarying_parameters_old(lmap):
        """for Ruocheng's .lmap file"""
        parameter_map = defaultdict(list)
        for lit_id in lmap:
            lit = lmap[lit_id]
            if lit.is_parameter():
                theta_st = lit.st
                if theta_st[0] == "!":
                    theta_st = theta_st[1:]
                parameter_map[theta_st].append(lit_id)
        for theta_st in parameter_map:
            covarying = sorted(parameter_map[theta_st])
            for lit_id in covarying:
                lmap[lit_id].covarying = covarying

    @staticmethod
    def find_covarying_parameters(lmap):
        """for Haiying's .lmap file.
        also sort of works with Jason's"""
        parameter_map = defaultdict(list)
        for lit_id in lmap:
            lit = lmap[lit_id]
            if lit.is_parameter():
                theta_st = lit.st

                if lit.is_threshold() or lit.is_gamma():
                    # does not covary
                    key = theta_st
                else:
                    if "|" in theta_st:
                        x,u = theta_st.split("|")
                        x,u = x.strip(' \\'),u.strip(' \\')
                        var,val = x.split("=")
                        if val.startswith("state"): val = val[5:]
                        key = "%s|%s" %  (var,u)
                    else:
                        var,val = theta_st.split("=")
                        key = "%s|" % var

                parameter_map[key].append(lit_id)

        for theta_st in parameter_map:
            covarying = sorted(parameter_map[theta_st])
            for lit_id in covarying:
                lmap[lit_id].covarying = covarying

def _set_tf_nodes(x,W,lmap,imap,return_weights=False):
    parameter_count = int(W.shape[0])
    weights = np.zeros((parameter_count,))
    warned = False

    exp_W = [None]*parameter_count
    for lit_id in lmap:
        lit = lmap[lit_id]
        if not lit.is_parameter(): continue
        # tf indices of covarying parameters
        indices = [ lmap[lit_id].index for lit_id in lit.covarying ]

        if lit.is_gamma():
            # untied parameter 
            # q = exp{-w} / ( 1 + exp{-w} )
            # p = min + (max-min) * q
            tau = tf.exp(-W[lit.index])
            exp_W[lit.index] = gamma_min + \
                               (gamma_max-gamma_min) * (1.0/(1.0+tau))
            p = lit.weight
            if not gamma_min < p < gamma_max:
                if not warned:
                    print("warning: gamma out of bounds (probably ignorable)")
                    warned = True
                p = 0.5*(gamma_min + gamma_max)
            Gamma = (p - gamma_min)/(gamma_max - gamma_min)
            weights[lit.index] = -math.log(Gamma/(1-Gamma))
        elif len(indices) == 1:
            # untied parameter 
            # p = exp{-w} / ( 1 + exp{-w} )
            """
            tau = tf.exp(-W[lit.index])
            exp_W[lit.index] = tau / ( 1.0 + tau )
            """
            tau = tf.exp(-W[lit.index])
            exp_W[lit.index] = thresh_min + \
                              (thresh_max-thresh_min) * (1.0/(1.0+tau))
            p = lit.weight
            if p == 0 or p == 1:
                if not warned:
                    print("warning: 0/1 parameter (probably ignorable)")
                    warned = True
                p = 0.5
            weights[lit.index] = -math.log(p/(1-p))
        else:
            # first parameter is clamped to 1
            # p = exp{-w_i} / ( 1 + exp{-w_1} + ... exp{-w_k} )
            z = 1.0
            for index in indices[1:]:
                z = z + tf.exp(-W[index])
            if lit.index == indices[0]:
                exp_W[lit.index] = 1.0 / z
            else:
                exp_W[lit.index] = tf.exp(-W[lit.index]) / z
            p0 = lmap[lit.covarying[0]].weight
            if p0 == 0 or lit.weight == 0:
                if not warned:
                    print("warning: 0/1 parameter (probably ignorable)")
                    warned = True
                ratio = 1.0
            else:
                ratio = lit.weight/p0
            weights[lit.index] = -math.log(ratio)

    # assign tensorflow nodes to TAC nodes
    for lit_id in lmap:
        lit = lmap[lit_id]
        index = lit.index
        if lit.is_parameter():
            lit.tf_node = exp_W[index]
        elif lit.is_indicator():
            lit.tf_node = x[:,index]

    if return_weights: return weights

def get_batch(examples,labels,size):
    dataset = list(zip(examples,labels))
    batch = random.sample(dataset,size)
    return zip(*batch)

def learn(node,training_examples,training_labels,testing_examples,
          rate=0.5,iterations=1000,thresh=None, grd_thresh=None, seed=None):
    lmap,imap = node.lmap,node.imap
    indicator_count = sum(len(imap[var]) for var in imap)
    parameter_count = len(lmap)-indicator_count

    # build graph
    dtype = tf.float64
    x = tf.placeholder(dtype, shape=[None, indicator_count])
    y_ = tf.placeholder(dtype, shape=[None])
    if seed is None:
        # AC: seed needs to be near uniform, for training sigmoids
        seed = .1*tf.random_uniform([parameter_count],dtype=dtype)-.05
        #seed = tf.random_uniform([parameter_count],dtype=dtype)-.5
    W = tf.Variable(seed)

    _set_tf_nodes(x,W,lmap,imap,return_weights=False)
    tac = node.tf_tac(lmap,use_sigmoid=True)

    gate = tf.train.GradientDescentOptimizer.GATE_GRAPH
    #loss = tf.losses.mean_squared_error(labels=y_,predictions=tac)
    loss = _my_mean_squared_error(y_,tac)
    #loss += 0.00002*my_regularizer(W) #ACAC
    grad = tf.gradients(loss,W)[0]
    #optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss,gate_gradients=gate)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss,gate_gradients=gate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {x: training_examples, y_: training_labels}
        print( "== initialization (rate %g) ==================" % rate )
        print( "err: %.8g" % loss.eval(feed_dict) )

        for iteration in range(iterations):
            """
            batch_size = len(training_examples) // 100
            batch_examples,batch_labels = get_batch(training_examples,training_labels,batch_size)
            sgd_dict = {x:batch_examples,y_:batch_labels}
            optimizer.run(feed_dict=sgd_dict)
            """
            optimizer.run(feed_dict=feed_dict)
            if iteration % (iterations/10) == 0:
                err = loss.eval(feed_dict)
                print( "== iteration %d ==================" % (iteration+1) )
                print( "err: %.8g" % err )
                if grd_thresh is not None:
                    cur_grd = sum(abs(grad.eval(feed_dict=feed_dict)))
                    print( "grd: %.8g" %  cur_grd)
                    if cur_grd < grd_thresh: break
                if thresh is not None and err < thresh: break
        print( "final-itr: %d" % iteration )
        print( "final-err: %.8g" % loss.eval(feed_dict) )
        print( "final-grd: %.8g" % sum(abs(grad.eval(feed_dict=feed_dict))) )

        weights = W.eval()
        predictions = tac.eval(feed_dict={x:testing_examples,W:weights})

    return predictions,weights

def simulate_tac(node,examples,weights=None,use_sigmoid=False):
    lmap,imap = node.lmap,node.imap
    indicator_count = sum(len(imap[var]) for var in imap)
    parameter_count = len(lmap)-indicator_count

    # build graph
    dtype = tf.float64
    x = tf.placeholder(dtype, shape=[None, indicator_count])
    y_ = tf.placeholder(dtype, shape=[None])
    W = tf.Variable(tf.random_uniform([parameter_count],dtype=dtype))
    lmap_weights = _set_tf_nodes(x,W,lmap,imap,return_weights=True)
    if weights is None: weights = lmap_weights
    tac = node.tf_tac(lmap,use_sigmoid=use_sigmoid)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        labels = tac.eval({W:weights,x:examples})

    return labels,weights

def read_tac(tac_filename,lmap_filename):
    tac = TacNode.ac_read(tac_filename)
    tac.lmap,tac.imap = Literal.lmap_read(lmap_filename)

    return tac

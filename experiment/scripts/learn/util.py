from pypsdd.data import Inst


def step(x):
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out

def step_two(x):
    return tf.maximum(tf.sign(x),tf.zeros(tf.shape(x)))

def as_table(alpha,manager,weights):
    var_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var_count = manager.var_count
    st = [ var_names[:var_count] + "   Pr  " ]
    st.append( ("-" * var_count) + "+------" )
    for model in alpha.models(alpha.vtree,lexical=True):
        wmc = alpha.wmc_model(model,weights)
        st.append( "%s %.4f" % (model,wmc) )
    st = "\n".join(st)
    return st

def as_dataset(alpha,manager,weights):
    data = {}
    var_count = manager.var_count
    for model in alpha.models(alpha.vtree,lexical=True):
        wmc = alpha.wmc_model(model,weights)
        model = Inst.from_dict(dict(model),var_count)
        data[model] = wmc
    return data

def wmc_model(self,inst,weights):
    """Returns Pr(inst) for a complete instantiation inst (where inst is
    an Inst or InstMap).

    Performs recursive test, which can be faster than linear
    traversal as in PSddNode.value."""
    self.is_model_marker(inst,clear_bits=False,clear_data=False)

    if self.data is None:
        pr = 0.0
    else:
        pr = 1.0
        queue = [self] if self.data is not None else []
        while queue:
            node = queue.pop()
            assert node.data is not None
            if node.is_decomposition():
                queue.append(node.data[0]) # prime
                queue.append(node.data[1]) # sub
            else:
                var = node.vtree.var
                val = inst[var]
                pr *= weights[var][val]

    self.clear_bits(clear_data=True)
    return pr

def weighted_model_count(self, lit_weights, clear_data=True):
    """ Compute weighted model count given literal weights

    Assumes the SDD is normalized.
    """
    for node in self.as_list(clear_data=clear_data):
        if node.is_false():
            data = 0
        elif node.is_true():
            data = 1
        elif node.is_literal():
            if node.literal > 0:
                data = lit_weights[node.literal-1][1]
            else:
                data = lit_weights[-node.literal-1][0]
        else: # node is_decomposition
            data = sum(p.data * s.data for p,s in node.elements)
        node.data = data
    return data


class AC():
    def __init__(self, ac_path, weight_path):
        self.testing_node_ind_list = []
        self.test_pos = {}
        self.test_neg = {}
        
        self.w_dict = {}
        self._readWeightFile(weight_path)
        self.num_w = len(self._w_file_lines)
        
        self.ac_dict = {}
        self._readACFile(ac_path)
        self.num_ac = len(self._ac_file_lines)
    def _readACFile(self, path):
        f = open(path, 'r')
        self._ac_file_lines = f.readlines()
        f.close()
        
    def _readWeightFile(self, path):
        f = open(path, 'r')
        self._w_file_lines = f.readlines()
        f.close()
        
        for line in self._w_file_lines:
            line = line.replace('\n', '')
            split = line.split(' ')
            _id = int(split[0])
            _type = split[1]
    
            # parameter
            if _type == 'p':
                self.w_dict[_id] = tf.constant(float(split[2]), 
                                               name = 'l_'+str(_id)+'_p', 
                                               dtype=tf.float32)
            # indicator
            elif _type == 'i':
                self.w_dict[_id] = tf.placeholder(tf.float32, name='l_'+str(_id)+'_i')
    def _createGraph(self, fd):
        
        with tf.variable_scope("TAC"):
            for line in self._ac_file_lines:
                line = line.replace('\n', '')
                split = line.split(' ')
                
                self._node(int(split[0]), split[1:], fd)
    def _node(self, _id, line, fd):
        print('id is', _id)
        _type = line[0]
        if _type == 'L':
            self.ac_dict[_id] = self.w_dict[abs(int(line[1]))]
        elif _type == '+' or _type == '*':
            _op = line[0]
            _n_children = int(line[1])
            suffix = ''
            for c in line[2:]:
                suffix = suffix + '_' + c
                
            print(suffix)
                
            if _type == '+':
                self.ac_dict[_id] = tf.add_n([ self.ac_dict[int(i)] for i in line[2:] ], name='add'+suffix)
            elif _type == '*':
                interm = self.ac_dict[int(line[2])]
                for child_ind in line[3:-1]:
                    child = self.ac_dict[int(child_ind)]
                    interm = interm*child
                self.ac_dict[_id] = tf.multiply(interm, self.ac_dict[int(line[-1])], name='mul'+suffix)
                
        elif _type == '?':
            T = self.ac_dict[int(line[2])]
            inp = self.ac_dict[int(line[3])]
            out_pos = self.ac_dict[int(line[4])]
            out_neg = self.ac_dict[int(line[5])]
            
            sign = tf.sign(inp-T)
            testing = tf.maximum(sign, tf.constant(0.0))
            
            if int(testing.eval(feed_dict=fd)) == 1:
                self.ac_dict[_id] = out_pos
            elif int(testing.eval(feed_dict=fd)) == 0:
                self.ac_dict[_id] = out_neg
    def evaluate(self):
        feed_dict = {self.w_dict[2]: 1,
                     self.w_dict[4]: 0,
                     self.w_dict[6]: 1,
                     self.w_dict[8]: 0,
                     self.w_dict[10]: 1,
                     self.w_dict[12]: 0
                    }
        
        self._createGraph(feed_dict)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
                
        res = self.sess.run(self.ac_dict[self.num_ac], feed_dict)
        print(res)
        
    def printACNodeList(self):
        feed_dict = {self.w_dict[2]: 1,
                     self.w_dict[4]: 0,
                     self.w_dict[6]: 1,
                     self.w_dict[8]: 0,
                     self.w_dict[10]: 1,
                     self.w_dict[12]: 0
                    }
        for i in self.ac_dict:
            print(i, self.ac_dict[i], self.ac_dict[i].eval(feed_dict=feed_dict))

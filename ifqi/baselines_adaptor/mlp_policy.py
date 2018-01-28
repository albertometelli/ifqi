from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype, DiagGaussianPdType
import numpy as np

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(*args, **kwargs)

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers,
              gaussian_fixed_var):
        assert isinstance(ob_space, gym.spaces.Box)
        self.gaussian_fixed_var = gaussian_fixed_var

        self.sess = U.single_threaded_session()
        self.sess.__enter__()

        #Only diagonal Gaussian policy is fully supported
        self.pdtype = pdtype = make_pdtype(ac_space)
        assert isinstance(pdtype,DiagGaussianPdType)
 
        #Normalize observation
        sequence_length = None
        self.ob = ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        #Critic network: not used
        '''
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        '''

        #Actor network
        #Normalized observation as input
        last_out = obz
        #Custom hidden layers of tanh neurons
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        #Mean, logStd as output; the latter may be fixed or learned   
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            self._mean = mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01)) 
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            self._fixed_std = tf.exp(logstd)
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        #Action distribution
        self.pd = pdtype.pdfromflat(pdparam)

        #Act
        self.state_in = []
        self.state_out = []
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        self._ac = ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        #Params
        all_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        outer_params = all_params[-3:] if self.gaussian_fixed_var else all_params[-2:]
        shapes = [t.shape for t in outer_params]
        if self.gaussian_fixed_var:
            theta = all_params[-3:-1]
        else:
            raise NotImplementedError

        #Score (gradlogp)
        self.ac_in = ac_in = U.get_placeholder(name='ac_in', dtype=tf.float32,
                                  shape=[sequence_length] + list(ac_space.shape))
        logp = -self.pd.neglogp(ac_in)
        score = tf.gradients(logp,all_params)
        outer_score = outer_score = tf.gradients(logp,outer_params)

        #Pdf
        self._density = density = tf.exp(-self.pd.neglogp(ac_in))

        #Flatten output
        def flatten(list_of_tensors):
            return tf.concat([tf.reshape(t,[-1]) for t in list_of_tensors],0)

        self._all_params = flatten(all_params)
        self._outer_params = flatten(outer_params)
        self._theta = flatten(theta)
        self._score = flatten(score)
        self._outer_score = flatten(outer_score)

        #Update network
        self.new_param_flat = U.get_placeholder(name='new_param_flat',
                                                dtype=tf.float32,
                                                shape=flatten(all_params).shape)
        start = 0
        self._assign = []
        for t in all_params:
            end = start + tf.size(t)
            self._assign.append(tf.assign(t,tf.reshape(self.new_param_flat[start:end],t.shape)))
            start = end

        self.sess.run(tf.global_variables_initializer())

    def act(self, stochastic, ob):
        return self.sess.run(self._ac, feed_dict={self.stochastic:stochastic,
                                                  self.ob:ob[None]})[0]

    def get_param(self,outer=False): 
        if outer:
            return self.sess.run(self._outer_params)
        else:
            return self.sess.run(self._all_params)

    def get_theta(self):
        return self.sess.run(self._theta)

    def get_mean(self,state):
        return self.sess.run(self._mean,feed_dict={self.ob:state[None]})[0]

    def get_std(self,state=None):
        if self.gaussian_fixed_var:
            return self.sess.run(self._fixed_std)
        else:
            raise NotImplementedError

    def get_score(self,state,action,outer=False):
        if outer:
            return self.sess.run(self._outer_score,
                                 feed_dict={self.ob:state[None],self.ac_in:action[None]})[0]
        else:
            return self.sess.run(self._score,
                                 feed_dict={self.ob:state[None],self.ac_in:action[None]})[0]

    def get_density(self,state,action):
        return self.sess.run(self._density,
                            feed_dict={self.ob:state[None],self.ac_in:action[None]})[0]

    def set_param(self,new_param):
        self.sess.run(self._assign,feed_dict={self.new_param_flat:new_param})

    #Not used:
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


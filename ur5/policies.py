from baselines.common.input import observation_placeholder
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import gym
from baselines.common.distributions import make_pdtype
from stable_baselines.a2c.utils import batch_to_seq, lstm, seq_to_batch

from utils import layernorm


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            # last_out = obz
            # for i in range(num_hid_layers):
            #     last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            print("v_net is shared!")
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []


class LstmPolicy(object):
    recurrent = True
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, nsteps, nlstm, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        # sequence_length = nsteps

        if nsteps > 1:
            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[nsteps] + list(ob_space.shape))
            mask = U.get_placeholder(name="mask", dtype=tf.float32, shape=[nsteps])
        else:
            ob = U.get_placeholder(name="ob_step", dtype=tf.float32, shape=[nsteps] + list(ob_space.shape))
            mask = U.get_placeholder(name="mask_step", dtype=tf.float32, shape=[nsteps])
        state_v = U.get_placeholder(name="state_v", dtype=tf.float32, shape=[1, nlstm * 2])
        state_p = U.get_placeholder(name="state_p", dtype=tf.float32, shape=[1, nlstm * 2])
            # ob = observation_placeholder(ob_space, batch_size=nsteps)
            # mask = tf.placeholder(dtype=tf.float32, shape=[nsteps])
            # state_v = tf.placeholder(dtype=tf.float32, shape=[1, nlstm*2])
            # state_p = tf.placeholder(dtype=tf.float32, shape=[1, nlstm*2])

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            # last_out = obz
            # for i in range(num_hid_layers):
            #     last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))

            ## RNN part
            xs = batch_to_seq(obz, 1, nsteps)
            ms = batch_to_seq(mask, 1, nsteps)
            h5, snew_v = lstm(xs, ms, state_v, 'lstm_v', n_hidden=nlstm,
                                              layer_norm=False)
            h = seq_to_batch(h5)
            self.initial_state_v = np.zeros(state_v.shape.as_list())
            self.state_v = self.initial_state_v.copy()
            # self.rnn_output = layernorm(self.rnn_output)

            self.vpred = tf.layers.dense(h, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            # last_out = obz
            # for i in range(num_hid_layers):
            #     last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))

            ## RNN part
            xs = batch_to_seq(obz, 1, nsteps)
            ms = batch_to_seq(mask, 1, nsteps)
            h5, snew_p = lstm(xs, ms, state_p, 'lstm_p', n_hidden=nlstm,
                            layer_norm=False)
            h = seq_to_batch(h5)
            self.initial_state_p = np.zeros(state_p.shape.as_list())
            self.state_p = self.initial_state_p.copy()

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(h, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(h, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = {'ob':ob, 'mask':mask, 'state_v':state_v, 'state_p':state_p}

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, state_v, state_p, mask], [ac, self.vpred, snew_v, snew_p])

    def act(self, stochastic, ob, state_v, state_p, mask):
        ac1, vpred1, snew_v1, snew_p1 =  self._act(stochastic, ob[None], state_v, state_p, mask[None])
        return ac1[0], vpred1[0], snew_v1[0], snew_p1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
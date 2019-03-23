import tensorflow as tf
import os
import numpy as np
from baselines.common.distributions import make_pdtype
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim, small_convnet_wodense, fc_regboard, layernorm
import tensor2tensor.layers.common_attention as attention

class RnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl,
                 n_env, n_steps, reuse, n_lstm=256, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_env * n_steps
        self.reuse = reuse
        # self.use_tboard = use_tboard
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(self.n_env, self.n_steps) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([self.n_env, self.n_steps], name='ac')
            self.masks_ph = tf.placeholder(tf.float32, [self.n_env, self.n_steps], name="masks_ph")  # mask (done t-1)
            self.flat_masks_ph = tf.reshape(self.masks_ph, [self.n_env * self.n_steps])
            self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2], name="states_ph")  # states
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=self.reuse)
            self.features = unflatten_first_dim(self.flat_features, sh)

            with tf.variable_scope(scope, reuse=self.reuse):
                input_sequence = batch_to_seq(self.flat_features, self.n_env, self.n_steps)
                masks = batch_to_seq(self.masks_ph, self.n_env, self.n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=False)
                rnn_output = seq_to_batch(rnn_output)
                # x = fc(self.rnn_output, units=hidsize, activation=activ, name="pol_fc1")
                # if self.use_tboard:
                #     weights = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')  # New
                #     tf.summary.histogram("kernel", weights)  # New
                #     bias = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')  # New
                #     tf.summary.histogram("bias", bias)  # New
                pdparam = fc(rnn_output, name='pd', units=pdparamsize, activation=None)
                vpred = fc(rnn_output, name='value_function_output', units=1, activation=None)
            pdparam = unflatten_first_dim(pdparam, sh)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)

    def get_features(self, x, reuse):
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob, state=None, mask=None):
        a, vpred, snew, nlp = \
            getsess().run([self.a_samp, self.vpred, self.snew, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.states_ph: state, self.masks_ph: mask[:, None]})
        # print(a.shape, vpred.shape, snew.shape, nlp.shape)
        return a[:, 0], vpred[:, 0], snew, nlp[:, 0]

    # def step(self, obs, state=None, mask=None, deterministic=False):
    #     if deterministic:
    #         return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp],
    #                              {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})
    #     else:
    #         return self.sess.run([self.action, self._value, self.snew, self.neglogp],
    #                              {self.obs_ph: obs, self.states_ph: state, self.masks_ph: mask})

class ErrorRnnPolicy(RnnPolicy):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl,
                 n_env, n_steps, reuse, n_lstm=256, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_env * n_steps
        self.reuse = reuse
        self.hidsize = hidsize
        self.feat_dim = feat_dim
        self.scope = scope
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(self.n_env, self.n_steps) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([self.n_env, self.n_steps], name='ac')
            self.masks_ph = tf.placeholder(tf.float32, [self.n_env, self.n_steps], name="masks_ph")  # mask (done t-1)
            self.flat_masks_ph = tf.reshape(self.masks_ph, [self.n_env * self.n_steps])
            self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2], name="states_ph")  # states
            self.pred_error = tf.placeholder(dtype=tf.float32, shape=(self.n_env, self.n_steps, self.hidsize), name='pred_error') # prediction error
            self.pd = self.vpred = None

            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=self.reuse)
            self.features = unflatten_first_dim(self.flat_features, sh)
            self.flat_pred_error = flatten_two_dims(self.pred_error)

            with tf.variable_scope(scope, reuse=self.reuse):
                ## Concat
                # x = tf.concat([self.flat_features, self.flat_pred_error], axis=1)

                ## Add
                # q = fc(self.flat_pred_error, units=hidsize, activation=activ, use_bias=False, name="error_embed")
                # k = fc(self.flat_features, units=hidsize, activation=activ, use_bias=False, name="feature_embed")
                # x = q + k

                ## ObStErrRe
                x = self.flat_pred_error

                input_sequence = batch_to_seq(x, self.n_env, self.n_steps)
                masks = batch_to_seq(self.masks_ph, self.n_env, self.n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=False)
                rnn_output = seq_to_batch(rnn_output)

                ## Concat
                q = self.flat_features
                q = tf.concat([q, rnn_output], axis=1)
                q = fc(q, units=hidsize, activation=activ, name="fc1")
                q = fc(q, units=hidsize, activation=activ, name="fc2")
                rnn_output = q

                pdparam = fc(rnn_output, name='pd', units=pdparamsize, activation=None)
                vpred = fc(rnn_output, name='value_function_output', units=1, activation=None)
            pdparam = unflatten_first_dim(pdparam, sh)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)

    def get_ac_value_nlp(self, ob, err, state=None, mask=None):
        a, vpred, snew, nlp = \
            getsess().run([self.a_samp, self.vpred, self.snew, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.states_ph: state, self.masks_ph: mask[:, None], self.pred_error: err[:, None]})
        return a[:, 0], vpred[:, 0], snew, nlp[:, 0]
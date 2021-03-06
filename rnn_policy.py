import gym
import tensorflow as tf
import os
import numpy as np
# from baselines.common.distributions import make_pdtype
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim, small_convnet_wodense, \
    fc_regboard, layernorm, get_action_n

class RNN(object):
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
        self.n_lstm = n_lstm
        self.reuse = reuse
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            # self.ac_pdtype = make_pdtype(ac_space)
            self.ac_pdtype = make_proba_dist_type(ac_space)
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
            self.pdparamsize = self.ac_pdtype.param_shape()[0]

            self.sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=self.reuse)
            self.features = unflatten_first_dim(self.flat_features, self.sh)

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

    def get_pdparam(self, x):
        pdparam = fc(x, name='pd', units=self.pdparamsize, activation=None)
        vpred = fc(x, name='value_function_output', units=1, activation=None)
        return pdparam, vpred


class RnnPolicy(RNN):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl,
                 n_env, n_steps, reuse, n_lstm=256, scope="policy"):
        super(RnnPolicy, self).__init__(ob_space, ac_space, hidsize,
                            ob_mean, ob_std, feat_dim, layernormalize, nl,
                            n_env, n_steps, reuse, n_lstm, scope)

        with tf.variable_scope(scope, reuse=self.reuse):
            ## Use features
            x = self.flat_features

            input_sequence = batch_to_seq(x, self.n_env, self.n_steps)
            masks = batch_to_seq(self.masks_ph, self.n_env, self.n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=False)
            rnn_output = seq_to_batch(rnn_output)
            layernorm(rnn_output)

            ## Concat
            q = self.flat_features
            q = tf.concat([q, rnn_output], axis=1)
            q = fc(q, units=hidsize, activation=activ, name="fc1")
            q = fc(q, units=hidsize, activation=activ, name="fc2")

            pdparam, vpred = self.get_pdparam(q)

        self.pdparam = pdparam = unflatten_first_dim(pdparam, self.sh)
        self.vpred = unflatten_first_dim(vpred, self.sh)[:, :, 0]
        self.pd = pd = self.ac_pdtype.proba_distribution_from_flat(pdparam)
        self.a_samp = pd.sample()
        self.entropy = pd.entropy()
        self.nlp_samp = pd.neglogp(self.a_samp)


    def get_ac_value_nlp(self, ob, state=None, mask=None):
        a, vpred, snew, nlp = \
            getsess().run([self.a_samp, self.vpred, self.snew, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.states_ph: state, self.masks_ph: mask[:, None]})
        return a[:, 0], vpred[:, 0], snew, nlp[:, 0]


class ErrorPredRnnPolicy(RNN):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl,
                 n_env, n_steps, reuse, n_lstm=256, scope="policy"):
        super(ErrorPredRnnPolicy, self).__init__(ob_space, ac_space, hidsize,
                            ob_mean, ob_std, feat_dim, layernormalize, nl,
                            n_env, n_steps, reuse, n_lstm, scope)
        with tf.variable_scope(scope):
            self.flat_masks_ph = tf.reshape(self.masks_ph, [self.n_env * self.n_steps])
            self.pred_error = tf.placeholder(dtype=tf.float32, shape=(self.n_env, self.n_steps, self.hidsize),
                                             name='pred_error')  # prediction error
            self.flat_pred_error = flatten_two_dims(self.pred_error)

            self.obs_pred = tf.placeholder(dtype=tf.float32, shape=(self.n_env, self.n_steps, self.hidsize),
                                           name='obs_pred')
            self.flat_obs_pred = flatten_two_dims(self.obs_pred)

            with tf.variable_scope(scope, reuse=self.reuse):
                x = tf.concat([self.flat_features, self.flat_obs_pred, self.flat_pred_error], axis=1)

                input_sequence = batch_to_seq(x, self.n_env, self.n_steps)
                masks = batch_to_seq(self.masks_ph, self.n_env, self.n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=False)
                rnn_output = seq_to_batch(rnn_output)
                rnn_output = layernorm(rnn_output)

                ## Concat
                q = self.flat_features
                q = tf.concat([q, rnn_output], axis=1)
                q = fc(q, units=hidsize, activation=activ, name="fc1")
                q = fc(q, units=hidsize, activation=activ, name="fc2")

                pdparam, vpred = self.get_pdparam(q)
            self.pdparam = pdparam = unflatten_first_dim(pdparam, self.sh)
            self.vpred = unflatten_first_dim(vpred, self.sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.proba_distribution_from_flat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)

    def get_ac_value_nlp(self, ob, err, obpred, state=None, mask=None):
        a, vpred, snew, nlp = \
            getsess().run([self.a_samp, self.vpred, self.snew, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.states_ph: state, self.masks_ph: mask[:, None],
                                     self.pred_error: err[:, None], self.obs_pred: obpred[:, None]})
        return a[:, 0], vpred[:, 0], snew, nlp[:, 0]


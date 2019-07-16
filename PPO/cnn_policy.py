import tensorflow as tf
import os
import numpy as np
# from baselines.common.distributions import make_pdtype
from stable_baselines.common.distributions import make_proba_dist_type

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim, small_convnet_wodense, fc_regboard, layernorm
import tensor2tensor.layers.common_attention as attention


class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            # self.ac_pdtype = make_pdtype(ac_space)
            self.ac_pdtype = make_proba_dist_type(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=False)
            self.features = unflatten_first_dim(self.flat_features, sh)

            with tf.variable_scope(scope, reuse=False):
                x = fc(self.flat_features, units=hidsize, activation=activ, name="fc1")
                x = fc(x, units=hidsize, activation=activ, name="fc1")
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
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

    def get_ac_value_nlp(self, ob):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]


# New
class PredErrorPolicy(CnnPolicy):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        # self.full_tensorboard_log = full_tensorboard_log
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            # self.ac_pdtype = make_pdtype(ac_space)
            self.ac_pdtype = make_proba_dist_type(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            # self.curr_ax_features = tf.placeholder(dtype=tf.float32,
            #                                        shape=(None, None, self.hidsize), name='curr_ax_features')
            # self.next_ax_features = tf.placeholder(dtype=tf.float32,
            #                                        shape=(None, None, self.hidsize), name='next_ax_features')
            self.pred_error = tf.placeholder(dtype=tf.float32,
                                             shape=(None, None, self.hidsize), name='pred_error')

        # def set_dynamics(self, ob_space, ac_space, hidsize,
        #          ob_mean, ob_std, feat_dim, layernormalize, nl, dynamics, scope="policy"):
        # self.dynamics = dynamics
        # self.pred_features = self.dynamics.pred_features

        # with tf.variable_scope(scope):
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=False)
            self.features = unflatten_first_dim(self.flat_features, sh)
            self.flat_pred_error = flatten_two_dims(self.pred_error)
            # self.pred_error = self.dynamics.pred_error

            with tf.variable_scope(scope, reuse=False):
                # x = tf.concat([self.flat_features, self.flat_pred_error], axis=1)
                q = fc(self.flat_pred_error, units=hidsize, activation=activ, use_bias=False, name="error_embed")
                k = fc(self.flat_features, units=hidsize, activation=activ, use_bias=False, name="feature_embed")
                x = q + k
                x = fc(x, units=hidsize, activation=activ, name="fc1")
                x = fc(x, units=hidsize, activation=activ, name="fc2")
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
            pdparam = unflatten_first_dim(pdparam, sh)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)
        # super.__init__(ob_space, ac_space, hidsize,
        #                ob_mean, ob_std, feat_dim, layernormalize, nl, scope)

    # def get_features(self, x, reuse):
    #     super.get_features(x, reuse)
    #
    # def get_ac_value_nlp(self, ob):
    #     super.get_ac_value_nlp(ob)

    def get_ac_value_nlp(self, ob, err):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.pred_error: err[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]

#New
class ErrorAttentionPolicy(CnnPolicy):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        # self.use_tboard = use_tboard
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            # self.ac_pdtype = make_pdtype(ac_space)
            self.ac_pdtype = make_proba_dist_type(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            self.pred_error = tf.placeholder(dtype=tf.float32,
                                             shape=(None, None, self.hidsize), name='pred_error')

            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=False)                                              # (nenvs*nsteps, width, height, chsize)
            self.features = unflatten_first_dim(self.flat_features, sh)                                         # (nenvs, nsteps, width, height, chsize)
            self.flat_pred_error = flatten_two_dims(self.pred_error)                                            # (nenvs*nsteps, hidsize)
            # self.pred_error = self.dynamics.pred_error

            with tf.variable_scope(scope, reuse=False):
                # x = fc(self.flat_features, units=hidsize, activation=activ)
                wid = self.flat_features.get_shape().as_list()[1]
                hei = self.flat_features.get_shape().as_list()[2]
                ch = self.flat_features.get_shape().as_list()[3]
                # print(wid, hei, ch)
                q = fc(self.flat_pred_error, units=hidsize, activation=activ, use_bias=False, name="query_embed")                   # (nenvs*nsteps, hidsize)
                q = layernorm(q)

                k = tf.reshape(self.flat_features, (-1, ch))                                               # (nenvs*nsteps*width*height, chsize)
                k = fc(k, units=hidsize, activation=activ, use_bias=False, name="key_embed")                                      # (nenvs*nsteps*width*height, hidsize)
                k = layernorm(k)

                q_ = tf.reduce_mean(self.flat_features, axis=[1, 2])
                q_ = fc(q_, units=hidsize, activation=activ, use_bias=False, name="key2_embed")
                q_ = layernorm(q_)
                q = q + q_  # Add observation information
                q = tf.expand_dims(q, 1)                                            # (nenvs*nsteps, 1, hidsize)

                v = attention.add_positional_embedding_nd(tf.reshape(k, (-1, wid, hei, hidsize)),
                                                          max_length=7,
                                                          name="pos_embed")
                k = tf.reshape(k, (-1, wid*hei, hidsize))                                                       # (nenvs*nsteps, width*height, hidsize)
                v = tf.reshape(v, (-1, wid*hei, hidsize))                                                        # (n, width*height, hidsize)

                x = attention.scaled_dot_product_attention_simple(q, k, v, bias=None)                           # (nenvs*nsteps, 1, hidsize)

                # num_heads = 8
                # x_list = []
                # for _ in range(num_heads):
                #     Q = tf.reshape(fc(q, units=ch, activation=activ, use_bias=False), (-1, 1, ch))
                #     K = tf.reshape(fc(k, units=ch, activation=activ, use_bias=False), (-1, wid*hei, ch))
                #     V = tf.reshape(fc(v, units=ch, activation=activ, use_bias=False), (-1, wid*hei, ch))
                #     x_list.append(attention.scaled_dot_product_attention_simple(Q, K, V, bias=None))
                #     x = tf.concat([attention.scaled_dot_product_attention_simple(fc(q, units=ch, activation=activ, use_bias=False),
                #                                                              fc(k, units=ch, activation=activ, use_bias=False),
                #                                                              fc(k, units=ch, activation=activ, use_bias=False), bias=None) for _ in range(num_heads)], axis=2)  # (nenvs*nsteps, 1, chsize*num_heads)
                # x = tf.concat(x_list, axis=2)
                # x = tf.layers.dropout(x, 0.2, training=tf.convert_to_tensor(True))
                x = tf.reshape(x, (-1, hidsize))
                x = fc(x, units=hidsize, activation=activ, use_bias=False)
                x = layernorm(x)

                x = fc(x, units=hidsize, activation=activ, name="fc1")
                x = fc(x, units=hidsize, activation=activ, name="fc2")
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
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
            x = small_convnet_wodense(x, nl=self.nl)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob, err):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None], self.pred_error: err[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]


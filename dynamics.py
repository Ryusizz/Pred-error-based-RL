import os

import gym
import numpy as np
import tensorflow as tf
import cloudpickle

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet, SaveLoad, get_action_n


class Dynamics(SaveLoad):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='dynamics', reuse=False):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        # self.full_tensorboard_log = full_tensorboard_log
        self.reuse = reuse
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        # self.rnn_output = self.auxiliary_task.rnn_output
        self.n_env = self.auxiliary_task.policy.n_env
        self.n_steps = self.auxiliary_task.policy.n_steps
        self.n_lstm = self.auxiliary_task.policy.n_lstm
        self.rnn_state = tf.placeholder(tf.float32, [self.n_env, self.n_steps, 2*self.n_lstm], name="rnn_state_ph")
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=self.reuse)
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.out_features = self.auxiliary_task.next_features

        self.pred_features = self.predict_next(self.reuse)
        self.pred_error = self.pred_features - tf.stop_gradient(self.out_features)
        with tf.variable_scope(self.scope + "_loss"):
            # self.loss = self.get_loss(self.pred_features)
            self.loss = tf.reduce_mean(self.pred_error ** 2, -1)
        self.params = tf.trainable_variables(self.scope)

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def predict_next(self, reuse):
        if isinstance(self.ac_space, gym.spaces.Discrete):
            ac = tf.one_hot(self.ac, get_action_n(self.ac_space), axis=2)
        else:
            ac = self.ac
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        rnn_state = flatten_two_dims(self.rnn_state)
        rnn_state = tf.layers.dense(rnn_state, 32, activation=tf.nn.leaky_relu)
        # print(rnn_output.get_shape(), ac.get_shape())

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)
        def add_ac_rnn(x):
            return tf.concat([x, ac, rnn_state], axis=-1)

        with tf.variable_scope(self.scope, reuse=reuse):
            x = flatten_two_dims(self.features)
            x = tf.layers.dense(add_ac_rnn(x), self.hidsize, activation=tf.nn.leaky_relu)

            def residual(x):
                res = tf.layers.dense(add_ac_rnn(x), self.hidsize, activation=tf.nn.leaky_relu)
                # if self.use_tboard:
                #     weights = tf.get_default_graph().get_tensor_by_name(os.path.split(res.name)[0] + '/kernel:0')
                #     tf.summary.histogram("dynamics_kernel1", weights)
                res = tf.layers.dense(add_ac_rnn(res), self.hidsize, activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)
            n_out_features = self.out_features.get_shape()[-1].value
            x = tf.layers.dense(add_ac_rnn(x), n_out_features, activation=None)
            x = unflatten_first_dim(x, sh)
        return x

    # def get_loss(self, x):
    #     return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1)

    # def calculate_loss(self, ob, last_ob, acs, nminibatches=None):
    #     if nminibatches is not None:
    #         n_chunks = nminibatches
    #     else:
    #         n_chunks = 8
    #     n = ob.shape[0]
    #     chunk_size = n // n_chunks
    #     assert n % n_chunks == 0
    #     sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
    #     return np.concatenate([getsess().run(self.loss,
    #                                          {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
    #                                           self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0) # New, maybe error

    def calculate_loss(self, ob, last_ob, acs, states, nminibatches=None):
        # states, _ = np.split(states, 2, axis=2)
        if nminibatches is not None:
            n_chunks = nminibatches
        else:
            n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return np.concatenate([getsess().run(self.loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)], self.rnn_state: states[sli(i)]}) for i in range(n_chunks)], 0) # New, maybe error

    # def calculate_pred(self, ob, acs):
        # return getsess().run(self.pred_error,
        #                      {self.obs: ob, self.ac: acs})

    # def calculate_err(self, ob, last_ob, acs):
    #     return getsess().run([self.pred_error, self.pred_features],
    #                          {self.obs: ob, self.last_ob: last_ob, self.ac: acs})

    def calculate_err(self, ob, last_ob, acs, states):
        # states, _ = np.split(states, 2, axis=2)
        return getsess().run([self.pred_error, self.pred_features],
                             {self.obs: ob, self.last_ob: last_ob, self.ac: acs, self.rnn_state: states})




class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        if isinstance(self.ac_space, gym.spaces.Discrete):
            ac = tf.one_hot(self.ac, get_action_n(self.ac_space), axis=2)
        else:
            ac = self.ac
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])






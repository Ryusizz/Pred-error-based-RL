import multiprocessing
import os
import platform
from functools import partial

import cloudpickle
import gym
import numpy as np
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from mpi4py import MPI


def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))


def get_mean_and_std(array):
    comm = MPI.COMM_WORLD
    task_id, num_tasks = comm.Get_rank(), comm.Get_size()
    local_mean = np.array(np.mean(array))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    mean = sum_of_means / num_tasks

    n_array = array - mean
    sqs = n_array ** 2
    local_mean = np.array(np.mean(sqs))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    var = sum_of_means / num_tasks
    std = var ** 0.5
    return mean, std


def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = guess_available_gpus()

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def guess_available_cpus():
    return int(multiprocessing.cpu_count())


def setup_tensorflow_session():
    num_cpu = guess_available_cpus()

    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)


def random_agent_ob_mean_std(env, nsteps=10000):
    ob = np.asarray(env.reset())
    if MPI.COMM_WORLD.Get_rank() == 0:
        obs = [ob]
        for _ in range(nsteps):
            ac = env.action_space.sample()
            ob, _, done, _ = env.step(ac)
            if done:
                ob = env.reset()
            obs.append(np.asarray(ob))
        mean = np.mean(obs, 0).astype(np.float32)
        std = np.std(obs, 0).mean().astype(np.float32)
    else:
        mean = np.empty(shape=ob.shape, dtype=np.float32)
        std = np.empty(shape=(), dtype=np.float32)
    MPI.COMM_WORLD.Bcast(mean, root=0)
    MPI.COMM_WORLD.Bcast(std, root=0)
    return mean, std


def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)


getsess = tf.get_default_session

fc = partial(tf.layers.dense, kernel_initializer=normc_initializer(1.0))
activ = tf.nn.relu


def flatten_two_dims(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])


def unflatten_first_dim(x, sh):
    return tf.reshape(x, [sh[0], sh[1]] + x.get_shape().as_list()[1:])


def add_pos_bias(x):
    with tf.variable_scope(name_or_scope=None, default_name="pos_bias"):
        b = tf.get_variable(name="pos_bias", shape=[1] + x.get_shape().as_list()[1:], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return x + b


def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x

#New
def small_convnet_wodense(x, nl, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    # x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    return x

def small_deconvnet(z, nl, ch, positional_bias):
    sh = (8, 8, 64)
    z = fc(z, np.prod(sh), activation=nl)
    z = tf.reshape(z, (-1, *sh))
    z = tf.layers.conv2d_transpose(z, 128, kernel_size=4, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [16, 16]
    z = tf.layers.conv2d_transpose(z, 64, kernel_size=8, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [32, 32]
    z = tf.layers.conv2d_transpose(z, ch, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert z.get_shape().as_list()[1:3] == [96, 96]
    z = z[:, 6:-6, 6:-6]
    assert z.get_shape().as_list()[1:3] == [84, 84]
    if positional_bias:
        z = add_pos_bias(z)
    return z


def unet(x, nl, feat_dim, cond, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    layers = []
    x = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])
    x = bn(tf.layers.conv2d(cond(x), filters=32, kernel_size=8, strides=(3, 3), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    layers.append(x)
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [8, 8]

    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = fc(cond(x), units=feat_dim, activation=nl)

    def residual(x):
        res = bn(tf.layers.dense(cond(x), feat_dim, activation=tf.nn.leaky_relu))
        res = tf.layers.dense(cond(res), feat_dim, activation=None)
        return x + res

    for _ in range(4):
        x = residual(x)

    sh = (8, 8, 64)
    x = fc(cond(x), np.prod(sh), activation=nl)
    x = tf.reshape(x, (-1, *sh))
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 32, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    x += layers.pop()
    x = tf.layers.conv2d_transpose(cond(x), 4, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert x.get_shape().as_list()[1:3] == [96, 96]
    x = x[:, 6:-6, 6:-6]
    assert x.get_shape().as_list()[1:3] == [84, 84]
    assert layers == []
    return x


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

# New
def fc_regboard(name):
    fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    tf.summary.histogram("kernel", fc_vars[0])
    tf.summary.histogram("bias", fc_vars[1])
    # tf.summary.histogram("act", layer1)

# def lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False):
#     """
#     Creates an Long Short Term Memory (LSTM) cell for TensorFlow
#
#     :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell
#     :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell
#     :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell
#     :param scope: (str) The TensorFlow variable scope
#     :param n_hidden: (int) The number of hidden neurons
#     :param init_scale: (int) The initialization scale
#     :param layer_norm: (bool) Whether to apply Layer Normalization or not
#     :return: (TensorFlow Tensor) LSTM cell
#     """
#     _, n_input = [v.value for v in input_tensor[0].get_shape()]
#     with tf.variable_scope(scope):
#         weight_x = tf.get_variable("wx", [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
#         weight_h = tf.get_variable("wh", [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
#         bias = tf.get_variable("b", [n_hidden * 4], initializer=tf.constant_initializer(0.0))
#
#         if layer_norm:
#             # Gain and bias of layer norm
#             gain_x = tf.get_variable("gx", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
#             bias_x = tf.get_variable("bx", [n_hidden * 4], initializer=tf.constant_initializer(0.0))
#
#             gain_h = tf.get_variable("gh", [n_hidden * 4], initializer=tf.constant_initializer(1.0))
#             bias_h = tf.get_variable("bh", [n_hidden * 4], initializer=tf.constant_initializer(0.0))
#
#             gain_c = tf.get_variable("gc", [n_hidden], initializer=tf.constant_initializer(1.0))
#             bias_c = tf.get_variable("bc", [n_hidden], initializer=tf.constant_initializer(0.0))
#
#     cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
#     for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
#         cell_state = cell_state * (1 - mask)
#         hidden = hidden * (1 - mask)
#         if layer_norm:
#             gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) \
#                     + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
#         else:
#             gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
#         in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)
#         in_gate = tf.nn.sigmoid(in_gate)
#         forget_gate = tf.nn.sigmoid(forget_gate)
#         out_gate = tf.nn.sigmoid(out_gate)
#         cell_candidate = tf.tanh(cell_candidate)
#         cell_state = forget_gate * cell_state + in_gate * cell_candidate
#         if layer_norm:
#             hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
#         else:
#             hidden = out_gate * tf.tanh(cell_state)
#         input_tensor[idx] = hidden
#     cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
#     return input_tensor, cell_state_hidden

class SaveLoad(object):
    def save(self, save_path, prefix=None):
        save_path += "/" + self.scope
        if prefix is not None:
            if not isinstance(prefix, str):
                prefix = str(prefix)
            save_path += "_" + prefix
        params = getsess().run(self.params)
        _save_to_file(save_path, params=params)

    def load(self, load_path):
        load_path += "/" + self.scope
        _, params = _load_from_file(load_path)

        restores = []
        for param, loaded_p in zip(self.params, params):
            restores.append(param.assign(loaded_p))
        getsess().run(restores)

# @staticmethod
def _save_to_file(save_path, data=None, params=None):
    if isinstance(save_path, str):
        _, ext = os.path.splitext(save_path)
        if ext == "":
            save_path += ".pkl"

        with open(save_path, "wb") as file_:
            cloudpickle.dump((data, params), file_)
    else:
        # Here save_path is a file-like object, not a path
        cloudpickle.dump((data, params), save_path)

# @staticmethod
def _load_from_file(load_path):
    if isinstance(load_path, str):
        if not os.path.exists(load_path):
            if os.path.exists(load_path + ".pkl"):
                load_path += ".pkl"
            else:
                raise ValueError("Error: the file {} could not be found".format(load_path))

        with open(load_path, "rb") as file:
            data, params = cloudpickle.load(file)
    else:
        # Here load_path is a file-like object, not a path
        data, params = cloudpickle.load(load_path)

    return data, params

def get_action_n(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        assert len(action_space.shape) == 1
        return action_space.shape[0]
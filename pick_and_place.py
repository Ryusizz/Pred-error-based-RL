import mujoco_py
import tensorflow as tf
import numpy as np
import gym
import math
import random

from gym.wrappers import Monitor

from wrappers import ExtraTimeLimit, ExternalForceWrapper


# class FlattenDictWrapper(gym.Wrapper):
#     def __init__(self, env, dict_keys):
#         super(FlattenDictWrapper, self).__init__(env)
#         # most recent raw observations (for max pooling across time steps)
#         # self._obs_buffer = deque(maxlen=2)
#         # self._skip = skip
#         self.dict_keys = dict_keys
#
#     def step(self, action):
#         """Repeat action, sum reward, and max over last observations."""
#         ob, r, d, info = self.env.step(action)
#         ob_flat = [ ob[k] for k in self.dict_keys ]
#         ob_flat = np.concatenate(ob_flat)
#         return ob_flat, r, d, info
#
#     def reset(self):
#         """Clear past frame buffer and init. to first obs. from inner env."""
#         ob = self.env.reset()
#         ob_flat = [ ob[k] for k in self.dict_keys ]
#         ob_flat = np.concatenate(ob_flat)
#         return ob_flat
mujoco_py.ignore_mujoco_warnings().__enter__()

def ob_process(ob, is_achieved, threshold):
    ob = ob[[0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1]]
    d = np.sqrt((ob[0] - ob[-6]) ** 2 + (ob[1] - ob[-5]) ** 2 + (ob[2] - ob[-4]) ** 2)
    print("distance:{}".format(d))
    if not is_achieved and d < threshold:
        is_achieved = True
    if not is_achieved:
        ob = ob[:-3]
    else:
        ob = ob[[0, 1, 2, 3, 4, 5, 9, 10, 11]]
    return ob, is_achieved



def start_test(**args):
    with tf.Session() as sess: #TODO: Seeding

        print(args['env'])
        env = gym.make(args['env'])
        # env.env.reward_type = "incentive"
        # print(env.reset())
        # env = Monitor(env, args['load_path'])
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'achieved_goal','desired_goal'])
        env = ExternalForceWrapper(env, 0.5)
        print(env.reset())
        has_err = args['has_err']
        threshold = 0.03

        i = 0
        is_achieved = False
        is_achieved_p = False
        grip_margin = 0
        while(True):
            i += 1
            print("Loading {}th model".format(i))

            saver = tf.train.import_meta_graph(args['load_path'] + args['model_name'])
            saver.restore(sess, tf.train.latest_checkpoint(args['load_path']))
            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)

            graph = tf.get_default_graph()

            a_mode = graph.get_tensor_by_name("policies/policy/action_mode:0")
            vpred = graph.get_tensor_by_name("policies/policy/vpred:0")
            snew = graph.get_tensor_by_name("policies/policy/snew:0")
            # snew = tf.zeros(shape=(8, 256))
            nlp_samp = graph.get_tensor_by_name("policies/policy/nlp_samp:0")

            states_ph = graph.get_tensor_by_name("policies/policy/states_ph:0")
            masks_ph = graph.get_tensor_by_name("policies/policy/masks_ph:0")
            ph_ob = graph.get_tensor_by_name("policies/policy/ob:0")

            if has_err:
                ac_ph = graph.get_tensor_by_name("policies/policy/ac:0")
                obs_pred_ph = graph.get_tensor_by_name("policies/policy/obs_pred_ph:0")
                pred_error_ph = graph.get_tensor_by_name("policies/policy/pred_error_ph:0")
                last_ob = graph.get_tensor_by_name("last_ob:0")
                rnn_state = graph.get_tensor_by_name("rnn_state_ph:0")
                pred_features = graph.get_tensor_by_name("pred_features:0")
                pred_error = graph.get_tensor_by_name("pred_error:0")

            eprew = 0

            # idx_array = [0,1,2,3,4,5,10,11,12]
            ob = env.reset()
            is_achieved_p = is_achieved
            ob, is_achieved = ob_process(ob, is_achieved, threshold)
            # ob = ob[idx_array]
            # ac_prev = np.zeros(4)
            ob_shaped = np.zeros((8, 1, 9))
            ob_shaped[0, ...] = ob
            if has_err:
                ob_prev_shaped = ob_shaped.copy()
                ac_prev_shaped = np.zeros((8, 1, 4))
            states_shaped = np.random.normal(size=(8, 256))
            masks_shaped = np.zeros((8, 1))
            # print(ob)
            # print(ob_shaped)

            # state = None
            # mask = None
            eprew_record = np.zeros(20)
            j = 0
            # for j in range(50):
            while j < 20:
                # get action from policy
                if has_err:
                    feed_dict = {ph_ob:ob_shaped, last_ob:ob_prev_shaped, ac_ph:ac_prev_shaped, rnn_state:states_shaped[:, None]}
                    err, obpred = sess.run([pred_error, pred_features], feed_dict)
                    feed_dict = {ph_ob:ob_shaped, states_ph:states_shaped, masks_ph:masks_shaped, obs_pred_ph:obpred, pred_error_ph:err}
                    a, v, s, nlp = sess.run([a_mode, vpred, snew, nlp_samp], feed_dict)
                else:
                    feed_dict = {ph_ob: ob_shaped, states_ph: states_shaped, masks_ph: masks_shaped}
                    a, v, s, nlp = sess.run([a_mode, vpred, snew, nlp_samp], feed_dict)
                # print(a.shape, v.shape, s.shape, nlp.shape)
                # states_shaped[0, ...] = s
                states_shaped = s
                if not is_achieved_p and is_achieved:
                    grip_margin = 10

                if not is_achieved:
                    a[0, 0, 3] = -1
                else:
                    a[0, 0, 3] = 5
                if 5 < grip_margin < 15:
                    a[0, 0, :] = a[0, 0, :]*0.01
                    a[0, 0, 3] = 5
                    grip_margin -= 1
                if 0 < grip_margin <= 5:
                    a[0, 0, :] = a[0, 0, :] * 0.01
                    a[0, 0, 2] = 1
                    grip_margin -= 1

                # step the environememtn
                ob, rew, done, info = env.step(a[0, 0, :]/4)
                is_achieved_p = is_achieved
                ob, is_achieved = ob_process(ob, is_achieved, threshold)
                # ob = ob[idx_array]

                # Oracle
                # a = np.zeros(4)
                # a[:3] = 2*(ob[-3:] - ob[:3])
                # ob, rew, done, info = env.step(a)
                env.render()

                if has_err:
                    ob_prev_shaped = ob_shaped.copy()
                    ob_shaped[0, ...] = ob
                    ac_prev_shaped = a.copy()
                    masks_shaped[0, :] = done
                else:
                    ob_shaped[0, ...] = ob
                    masks_shaped[0, :] = done

                eprew += rew
                if done:
                    eprew_record[j] = eprew
                    j += 1
                    print("Episode {} reward: {}".format(j, eprew))

                    eprew = 0
                    ob = env.reset()
                    ob, is_achieved = ob_process(ob, is_achieved, threshold)
                    # ob = ob[idx_array]
                    ob_shaped[0, ...] = ob
                    states_shaped = np.random.normal(size=(8, 256))
                    masks_shaped = np.zeros((8, 1))
                    is_achieved = False
            print("Average reward for model {} : {}".format(i, np.mean(eprew_record)))


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='UR5PickAndPlaceDense-v1',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=30000, type=int)
    parser.add_argument('--env_kind', type=str, default="robotics")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    # add_optimization_params(parser)
    # add_rollout_params(parser)

    # parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # parser.add_argument('--feat_sharedWpol', type=int, default=1) # New
    # parser.add_argument('--save_dynamics', type=int, default=0)
    # parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--load_path', type=str, default='/result/UR5ReachDense-v1/ErrPred_ppo1like_noObRewNorm_scopeFix_Force50_3/')
    parser.add_argument('--model_name', type=str, default='model.meta')
    parser.add_argument('--has_err', type=int, default=1)

    args = parser.parse_args()

    start_test(**args.__dict__)

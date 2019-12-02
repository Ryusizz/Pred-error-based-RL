import tensorflow as tf
import numpy as np
import gym

def start_test(**args):
    with tf.Session() as sess: #TODO: Seeding

        env = gym.make(args['env'])
        env.env.reward_type = "dense"
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        i = 0
        while(True):
            i += 1
            print("Loading {}th model".format(i))

            saver = tf.train.import_meta_graph(args['load_path'] + args['model_name'])
            saver.restore(sess, tf.train.latest_checkpoint(args['load_path']))
            # for op in tf.get_default_graph().get_operations():
            #     print(op.name)

            graph = tf.get_default_graph()

            a_samp = graph.get_tensor_by_name("policy/action_sample:0")
            vpred = graph.get_tensor_by_name("policy/vpred:0")
            snew = graph.get_tensor_by_name("policy/snew:0")
            nlp_samp = graph.get_tensor_by_name("policy/nlp_samp:0")

            states_ph = graph.get_tensor_by_name("policy/states_ph:0")
            masks_ph = graph.get_tensor_by_name("policy/masks_ph:0")
            ph_ob = graph.get_tensor_by_name("policy/ob:0")
            eprew = 0

            ob = env.reset()
            ob_shaped = np.zeros((128, 1, 13))
            ob_shaped[0, ...] = ob
            states_shaped = np.random.normal(size=(128, 256))
            masks_shaped = np.zeros((128, 1))
            # print(ob)
            # print(ob_shaped)

            # state = None
            # mask = None
            eprew_record = np.zeros(20)
            j = 0
            # for j in range(50):
            while j < 20:
                # get action from policy
                feed_dict = {ph_ob:ob_shaped, states_ph:states_shaped, masks_ph:masks_shaped}
                a, v, s, nlp = sess.run([a_samp, vpred, snew, nlp_samp], feed_dict)
                # print(a.shape, v.shape, s.shape, nlp.shape)
                # states_shaped[0, ...] = s
                states_shaped = s
                # ob, rew, done, info = env.step(a[0, 0, :])

                #
                ob, rew, done, info = env.step(ob[:3] - ob[-3:])
                env.render()
                ob_shaped[0, ...] = ob
                masks_shaped[0, :] = done
                eprew += rew
                if done:
                    eprew_record[j] = eprew
                    j += 1
                    print("Episode {} reward: {}".format(j, eprew))

                    eprew = 0
                    ob = env.reset()
                    ob_shaped[0, ...] = ob
                    states_shaped = np.random.normal(size=(128, 256))
                    masks_shaped = np.zeros((128, 1))
            print("Average reward for model {} : {}".format(i, np.mean(eprew_record)))


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='FetchReach-v1',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=30000, type=int)
    parser.add_argument('--env_kind', type=str, default="robotics")
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--dyn_coeff', type=float, default=1)
    parser.add_argument('--aux_coeff', type=float, default=1)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e7))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    # parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=1.)
    parser.add_argument('--int_coeff', type=float, default=0.)
    parser.add_argument('--layernorm', type=int, default=0)
    # parser.add_argument('--feat_learning', type=str, default="none",
    #                     choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    # parser.add_argument('--policy_mode', type=str, default="rnnerrpred",
    #                     choices=["rnn", "rnnerr", "rnnerrac", "rnnerrpred", "rnnerrprede2e", 'rnn_noconv', 'rnnerrpred_noconv']) # New
    # parser.add_argument('--full_tensorboard_log', type=int, default=1) # New
    # parser.add_argument('--tboard_period', type=int, default=10) # New
    # parser.add_argument('--feat_sharedWpol', type=int, default=1) # New
    # parser.add_argument('--save_dynamics', type=int, default=0)
    # parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--load_path', type=str, default='/result/FetchReach-v1/PPO_UseNews_Test_3/')
    parser.add_argument('--model_name', type=str, default='model.meta')
    # parser.add_argument('--hidsize', type=int, default=512)
    # parser.add_argument('--n_lstm', type=int, default=256)

    args = parser.parse_args()

    start_test(**args.__dict__)

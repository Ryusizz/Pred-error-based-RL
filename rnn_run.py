#!/usr/bin/env python
import datetime

from rnn_policy_noconv import RnnPolicy_NoConv, ErrorPredRnnPolicy_NoConv

try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from stable_baselines.common import tf_util
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from rnn_policy import *
from rppo_agent import RnnPpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit

# Troubleshooting
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)
    logdir = osp.join("/result", args['env'], args['exp_name'], datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    log = logger.scoped_configure(dir=logdir, format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])

    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'],
                      logdir=logdir)
    tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        print("results will be saved to ", logdir)
        with open("{}/args.txt".format(logdir), 'w') as argfile:
            print("saving argments...")
            for k, v in args.items():
                argfile.write(str(k) + ' >>> ' + str(v) + '\n')

        trainer.train(tf_sess)


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process, logdir):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self.logdir = logdir
        self._set_env_vars()

        self.policy = {"rnn" : RnnPolicy,
                       "rnnerr" : ErrorRnnPolicy,
                       "rnnerrac" : ErrorActRnnPolicy,
                       "rnnerrpred" : ErrorPredRnnPolicy,
                       'rnnerrprede2e' : ErrorPredE2ERnnPolicy,
                       'rnn_noconv' : RnnPolicy_NoConv,
                       'rnnerrpred_noconv' : ErrorPredRnnPolicy_NoConv}[hps['policy_mode']]
        self.action_policy = self.policy(
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=hps['hidsize'],
            feat_dim=hps['hidsize'],
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu,
            n_env=hps['envs_per_process'],
            n_steps=1,
            reuse=False,
            n_lstm=hps['hidsize'] / 2,
        )
        with tf.variable_scope("train_model", reuse=True,
                               custom_getter=tf_util.outer_scope_getter("train_model")):
            self.train_policy = self.policy(
                ob_space=self.ob_space,
                ac_space=self.ac_space,
                hidsize=hps['hidsize'],
                feat_dim=hps['hidsize'],
                ob_mean=self.ob_mean,
                ob_std=self.ob_std,
                layernormalize=False,
                nl=tf.nn.leaky_relu,
                n_env=hps['envs_per_process'] // hps['nminibatches'],
                n_steps=hps['nsteps_per_seg'],
                reuse=True,
                n_lstm=hps['hidsize'] / 2,
            )
        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.action_feature_extractor = self.feature_extractor(policy=self.action_policy,
                                                               features_shared_with_policy=hps['feat_sharedWpol'],
                                                               feat_dim=hps['hidsize'],
                                                               layernormalize=hps['layernorm'])
        self.train_feature_extractor = self.feature_extractor(policy=self.train_policy,
                                                              features_shared_with_policy=hps['feat_sharedWpol'],
                                                              feat_dim=hps['hidsize'],
                                                              layernormalize=hps['layernorm'],
                                                              reuse=True)

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.action_dynamics = self.dynamics(auxiliary_task=self.action_feature_extractor,
                                             predict_from_pixels=hps['dyn_from_pixels'],
                                             feat_dim=hps['hidsize'])
        self.train_dynamics = self.dynamics(auxiliary_task=self.train_feature_extractor,
                                            predict_from_pixels=hps['dyn_from_pixels'],
                                            feat_dim=hps['hidsize'],
                                            reuse=True)
        if 'e2e' in hps['policy_mode']:
            self.action_policy.prepare_else(self.action_dynamics)
            self.train_policy.prepare_else(self.train_dynamics)

        self.agent = RnnPpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            actionpol=self.action_policy,
            trainpol=self.train_policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            action_dynamics=self.action_dynamics,
            train_dynamics=self.train_dynamics,
            policy_mode=hps['policy_mode'],
            logdir=logdir,
            full_tensorboard_log=hps['full_tensorboard_log'],
            tboard_period=hps['tboard_period'],
            hidsize=hps['hidsize'],
            n_lstm=hps['n_lstm']
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.train_feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux'] * self.hps['aux_coeff']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.train_dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss'] * self.hps['dyn_coeff']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.train_feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        # self.ob_mean = np.zeros_like(self.ob_mean) # don't use the observation normalization
        # self.ob_std = np.ones_like(self.ob_std)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self, tf_sess):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.action_dynamics)
        expdir = osp.join("/result", self.hps['env'], self.hps['exp_name'])
        save_checkpoints = []
        if self.hps['save_interval'] is not None:
            save_checkpoints = [i*self.hps['save_interval'] for i in range(1, self.hps['num_timesteps']//self.hps['save_interval'])]
        if self.hps['load_dir'] is not None: #Todo: Bug Fix!
            self.train_feature_extractor.load(self.hps['load_dir'])
            self.train_dynamics.load(self.hps['load_dir'])

        saver = tf.train.Saver()
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
                if MPI.COMM_WORLD.Get_rank() == 0:
                    saver.save(tf_sess, osp.join(expdir, 'model'))
            # if len(save_checkpoints) > 0:
            #     if self.agent.rollout.stats['tcount'] > save_checkpoints[0]:
            #         # self.train_feature_extractor.save(expdir, self.agent.rollout.stats['tcount'])
            #         # self.train_dynamics.save(expdir, self.agent.rollout.stats['tcount'])
            #         self.action_policy.save(expdir, self.agent.rollout.stats['tcount'])
            #         self.action_feature_extractor.save(expdir, self.agent.rollout.stats['tcount'])
            #         self.action_dynamics.save(expdir, self.agent.rollout.stats['tcount'])
            #         save_checkpoints.remove(save_checkpoints[0])
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        if self.hps['save_dynamics'] and MPI.COMM_WORLD.Get_rank()== 0:       # save auxilary task and dynamics parameter
            # self.train_feature_extractor.save(expdir)
            # self.train_dynamics.save(expdir)
            # self.action_policy.save(expdir)
            # self.action_feature_extractor.save(expdir)
            # self.action_dynamics.save(expdir)
            saver.save(tf_sess, '/result/test')
        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()
    elif args["env_kind"] == 'field':
        import gym_fieldedmove
        env = gym.make(args['env'])
        # env.set_mode('human')
    elif args["env_kind"] == "bird":
        import gym_ple
        env = gym.make('FlappyBird-v0')
        # env = NoopResetEnv(env, noop_max=args['noop_max'])
        env._max_episode_steps = args['max_episode_steps']
        # env.game_state.rng = np.random.RandomState()
        env = ProcessFrame84(env, crop=False)
        # env = ExtraTimeLimit(env, args['max_episode_steps'])
        env = FrameStack(env, 4)
    elif args["env_kind"] == 'carracing':
        # import roboenvs as robo
        env = gym.make("CarRacing-v0")
        # env = NoopResetEnv(env, noop_max=args['noop_max'])
        # env = robo.DiscretizeActionWrapper(env, 2)
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'robotics':
        env = gym.make(args['env'])
        env.env.reward_type = "dense"
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    tf_context = setup_tensorflow_session()
    return tf_context


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

    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=1.)
    parser.add_argument('--int_coeff', type=float, default=0.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--policy_mode', type=str, default="rnnerrpred",
                        choices=["rnn", "rnnerr", "rnnerrac", "rnnerrpred", "rnnerrprede2e", 'rnn_noconv', 'rnnerrpred_noconv']) # New
    parser.add_argument('--full_tensorboard_log', type=int, default=1) # New
    parser.add_argument('--tboard_period', type=int, default=10) # New
    parser.add_argument('--feat_sharedWpol', type=int, default=1) # New
    parser.add_argument('--save_dynamics', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--hidsize', type=int, default=512)
    parser.add_argument('--n_lstm', type=int, default=256)

    args = parser.parse_args()

    start_experiment(**args.__dict__)

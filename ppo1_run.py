#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import mujoco_py

from ur5 import policies


def train(env_id, num_timesteps, seed, network):
    # from baselines.ppo1 import mlp_policy, pposgd_simple
    from ur5 import pposgd_simple
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    mujoco_py.ignore_mujoco_warnings().__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    env = make_robotics_env(env_id, workerseed, rank=rank)
    def policy_fn(network, name, ob_space, ac_space, nsteps):
        if network == 'mlp':
            return policies.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=256, num_hid_layers=3)
        elif network == 'lstm':
            return policies.LstmPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=256, num_hid_layers=3, nsteps=nsteps, nlstm=128)

    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=256,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
            gamma=0.99, lam=0.95, schedule='linear', network=network,
        )
    env.close()
    return pi

def main():
    args = robotics_arg_parser().parse_args()

    pi = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, network=args.network)
    if args.play:
        # pi = train(num_timesteps=1, seed=args.seed)
        # U.load_state(args.model_path)
        # env = make_mujoco_env('Humanoid-v2', seed=0)
        rank = MPI.COMM_WORLD.Get_rank()
        env = make_robotics_env(args.env, args.seed + 10000 * rank, rank=rank)
        ob = env.reset()
        if rank == 0:
            while True:
                action = pi.act(stochastic=False, ob=ob)[0]
                ob, _, done, _ =  env.step(action)
                env.render()
                if done:
                    ob = env.reset()

if __name__ == '__main__':
    main()

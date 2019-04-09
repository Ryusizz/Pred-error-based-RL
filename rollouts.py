from collections import deque, defaultdict

import numpy as np
from mpi4py import MPI

from recorder import Recorder
from utils import unflatten_first_dim


class Rollout(object):
    def __init__(self, ob_space, ac_space, nenvs, nminibatches, nsteps_per_seg, nsegs_per_env, nlumps, envs, policy,
                 int_rew_coeff, ext_rew_coeff, record_rollouts, train_dynamics, policy_mode, action_dynamics=None):
        self.nenvs = nenvs
        self.nminibatches = nminibatches
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.train_dynamics = train_dynamics
        if action_dynamics is not None:
            self.action_dynamics = action_dynamics
        else:
            self.action_dynamics = self.train_dynamics
        self.policy_mode = policy_mode

        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_acs_first = np.empty((nenvs, self.nsegs_per_env, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        # self.buf_fpred = np.empty((nenvs, self.nsteps, *self.dynamics.pred_error.shape), self.dynamics.dtype) # New
        # print(dynamics.pred_error.shape)
        # print(dynamics.pred_error.dtype)
        self.buf_errs = np.zeros((nenvs, self.nsteps, 512), np.float32) # New
        self.buf_errs_last = self.buf_errs[:, 0, ...].copy() # New
        self.buf_obpreds = np.zeros((nenvs, self.nsteps, 512), np.float32)
        self.buf_obpreds_last = self.buf_obpreds[:, 0, ...].copy()
        self.buf_states = np.zeros((nenvs, self.nsteps, 512), np.float32) # RNN
        self.buf_states_last = self.buf_states[:, 0, ...].copy()
        self.buf_states_first = self.buf_states[:, 0, ...].copy()

        self.env_results = [None] * self.nlumps
        # self.prev_feat = [None for _ in range(self.nlumps)]
        # self.prev_acs = [None for _ in range(self.nlumps)]
        self.int_rew = np.zeros((nenvs,), np.float32)

        self.recorder = Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []

        self.step_count = 0

    def collect_rollout(self):
        self.ep_infos_new = []
        for t in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward()
        self.update_info()

    def calculate_reward(self):
        if 'rnn' in self.policy_mode:
            int_rew = self.train_dynamics.calculate_loss(ob=self.buf_obs,
                                                   last_ob=self.buf_obs_last,
                                                   acs=self.buf_acs,
                                                   nminibatches=self.nminibatches)
        else:
            int_rew = self.train_dynamics.calculate_loss(ob=self.buf_obs,
                                               last_ob=self.buf_obs_last,
                                               acs=self.buf_acs)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)

    def rollout_step(self):
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for l in range(self.nlumps):
            obs, prevrews, news, infos = self.env_get(l)

            for info in infos:
                epinfo = info.get('episode', {})
                mzepinfo = info.get('mz_episode', {})
                retroepinfo = info.get('retro_episode', {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    self.ep_infos_new.append((self.step_count, epinfo))

            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)

            policy_input = [obs]
            if 'err' in self.policy_mode:
                if t == 0:
                    obpreds = self.buf_obpreds_last[sli]
                    errs = self.buf_errs_last[sli]
                elif t < self.nsteps:
                    a = np.expand_dims(self.buf_obs[sli, t - 1], 1)
                    b = np.expand_dims(obs, 1)
                    c = np.expand_dims(self.buf_acs[sli, t - 1], 1)
                    errs, obpreds = np.squeeze(self.action_dynamics.calculate_err(a, b, c))
                policy_input.append(errs)
                if 'pred' in self.policy_mode:
                    policy_input.append(obpreds)
            if 'ac' in self.policy_mode:
                if t == 0:
                    self.buf_acs_first = np.expand_dims(self.buf_acs[sli, -1], 1)
                acs_before = self.buf_acs[sli, t-1]
                policy_input.append(acs_before)
            if 'rnn' in self.policy_mode:
                if t == 0:
                    states = self.buf_states_last[sli]
                    self.buf_states_first[sli] = states
                elif t < self.nsteps:
                    states = self.buf_states[sli, t-1]
                policy_input.append(states)
                policy_input.append(news)

            policy_output = self.policy.get_ac_value_nlp(*policy_input)
            if len(policy_output) == 3:
                acs, vpreds, nlps = policy_output
            elif len(policy_output) == 4:
                acs, vpreds, states, nlps = policy_output
            # print("state means :", states.mean(), " err means :", errs.mean())

            # if self.policy_mode in ['naiveerr', 'erratt'] :
            #     if t == 0 :
            #         errs = self.buf_errs_last[sli]
            #         obpreds = self.buf_obpreds_last[sli]
            #     elif t < self.nsteps :
            #         a = np.expand_dims(self.buf_obs[sli, t - 1], 1)
            #         b = np.expand_dims(obs, 1)
            #         c = np.expand_dims(self.buf_acs[sli, t - 1], 1)
            #         errs, obpreds = np.squeeze(self.action_dynamics.calculate_err(a, b, c))
            #     acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs, errs, obpreds)
            #
            # elif self.policy_mode in ['rnn']:
            #     if t == 0:
            #         # acs, vpreds, states, nlps = self.policy.get_ac_value_nlp(obs)
            #         states = self.buf_states_last[sli]
            #         self.buf_states_first[sli] = states
            #     elif t < self.nsteps:
            #         states = self.buf_states[sli, t-1]
            #     acs, vpreds, states, nlps = self.policy.get_ac_value_nlp(obs, states, news)
            #
            # elif self.policy_mode in ['rnnerr']:
            #     if t == 0:
            #         states = self.buf_states_last[sli]
            #         self.buf_states_first[sli] = states
            #         errs = self.buf_errs_last[sli]
            #     elif t < self.nsteps:
            #         states = self.buf_states[sli, t - 1]
            #         a = np.expand_dims(self.buf_obs[sli, t - 1], 1)
            #         b = np.expand_dims(obs, 1)
            #         c = np.expand_dims(self.buf_acs[sli, t - 1], 1)
            #         errs, obpreds = np.squeeze(self.action_dynamics.calculate_err(a, b, c))
            #     acs, vpreds, states, nlps = self.policy.get_ac_value_nlp(obs, errs, states, news)
            # elif self.policy_mode in ['rnnerrac']:
            #     if t == 0:
            #         self.buf_acs_first = np.expand_dims(self.buf_acs[sli, -1], 1)
            #         states = self.buf_states_last[sli]
            #         self.buf_states_first[sli] = states
            #         errs = self.buf_errs_last[sli]
            #     elif t < self.nsteps:
            #         states = self.buf_states[sli, t - 1]
            #         a = np.expand_dims(self.buf_obs[sli, t - 1], 1)
            #         b = np.expand_dims(obs, 1)
            #         c = np.expand_dims(self.buf_acs[sli, t - 1], 1)
            #         errs, obpreds = np.squeeze(self.action_dynamics.calculate_err(a, b, c))
            #     acs_before = self.buf_acs[sli, t-1]
            #     acs, vpreds, states, nlps = self.policy.get_ac_value_nlp(obs, errs, states, acs_before, news)
            # else :
            #     acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)

            self.env_step(l, acs)

            # self.prev_feat[l] = dyn_feat
            # self.prev_acs[l] = acs
            self.buf_obs[sli, t] = obs
            self.buf_news[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_nlps[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            if 'err' in self.policy_mode:
                self.buf_errs[sli, t] = errs
                self.buf_obpreds[sli, t] = obpreds
            if 'rnn' in self.policy_mode:
                self.buf_states[sli, t] = states

            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews
            if self.recorder is not None:
                self.recorder.record(timestep=self.step_count, lump=l, acs=acs, infos=infos, int_rew=self.int_rew[sli],
                                     ext_rew=prevrews, news=news)
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            for l in range(self.nlumps):
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, _ = self.env_get(l)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews
                    policy_input = [nextobs]
                    if 'err' in self.policy_mode:
                        a = np.expand_dims(obs, 1)
                        b = np.expand_dims(nextobs, 1)
                        c = np.expand_dims(acs, 1)
                        nexterrs, nextobpreds = np.squeeze(self.action_dynamics.calculate_err(a, b, c))
                        self.buf_errs_last[sli] = nexterrs
                        self.buf_obpreds_last[sli] = nextobpreds
                        policy_input.append(nexterrs)
                        if 'pred' in self.policy_mode:
                            policy_input.append(nextobpreds)
                        # nextacs, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(nextobs, nexterrs)
                    if 'ac' in self.policy_mode:
                        policy_input.append(acs)
                        # nextacs, self.buf_vpred_last[sli], self.buf_states_last[sli], _ = self.policy.get_ac_value_nlp(nextobs, nexterrs, acs, states, nextnews)
                    if 'rnn' in self.policy_mode:
                        policy_input.append(states)
                        policy_input.append(nextnews)

                        # nextacs, self.buf_vpred_last[sli], self.buf_states_last[sli], _ = self.policy.get_ac_value_nlp(nextobs, states, nextnews) # RNN!
                    policy_output = self.policy.get_ac_value_nlp(*policy_input)
                    if len(policy_output) == 3:
                        nextacs, self.buf_vpred_last[sli], _ = policy_output
                    elif len(policy_output) == 4:
                        nextacs, self.buf_vpred_last[sli], self.buf_states_last[sli], _ = policy_output

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists['eprew'].extend(all_ep_infos['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos['r'])
            self.statlists['eplen'].extend(all_ep_infos['l'])
            self.stats['epcount'] += len(all_ep_infos['l'])
            self.stats['tcount'] += sum(all_ep_infos['l'])
            if 'visited_rooms' in keys_:
                # Montezuma specific logging.
                self.stats['visited_rooms'] = sorted(list(set.union(*all_ep_infos['visited_rooms'])))
                self.stats['pos_count'] = np.mean(all_ep_infos['pos_count'])
                self.all_visited_rooms.extend(self.stats['visited_rooms'])
                self.all_scores.extend(all_ep_infos["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited rooms")
                    print(self.all_visited_rooms)
                    print("All scores")
                    print(self.all_scores)
            if 'levels' in keys_:
                # Retro logging
                temp = sorted(list(set.union(*all_ep_infos['levels'])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited levels")
                    print(self.all_visited_rooms)

            current_max = np.max(all_ep_infos['r'])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out

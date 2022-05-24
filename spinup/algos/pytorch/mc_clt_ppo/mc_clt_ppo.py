from email.mime import base
import math
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.mc_clt_ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import bisect

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, d_output=5):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.ue_val_buf = np.zeros(size, dtype=np.float32) # uncertainty bonus from ue network
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.d_val_buf = np.zeros(size, dtype=np.float32)
        self.ep_time_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.d_output = d_output


    def store(self, obs, act, rew, logp, d_val, ep_time):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.d_val_buf[self.ptr] = d_val
        self.ep_time_buf[self.ptr] = ep_time

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        d_vals = np.append(self.d_val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation 
        deltas = rews[:-1] + self.gamma * d_vals[1:] - d_vals[:-1] # with ue reward r' and s_v
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + d_vals[:-1]
    
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, ep_time=self.ep_time_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def mc_clt_ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=250, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, d_vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=100, d_output=8, mn_std=20, alpha=0.05):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    import random
    random.seed(seed)   
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cur_l = 10 # episode length. This value increases as the agent gets better. 
    cur_var = 10
    upper_bonus = 1 # upper bound for uncertainty bonus. This value increases as the agent gets better. 

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    env.seed(seed)

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs, d_ouput=d_output)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.d_v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, d_output)

    # z value range
    z_bar_range = np.linspace(-3, 3, num = 2000000)
    z_cdf = np.zeros_like(z_bar_range) 
    cum_prob = []
    for i in range(d_output):
        cum_prob.append((i + 1) / (d_output + 1))
    cum_prob = torch.from_numpy(np.array(cum_prob).reshape(1, -1))

    def compute_standard_normal_locations(n):
        dx = z_bar_range[1] - z_bar_range[0]
        per_sum = 0.00135
        z_cdf[0] = per_sum
        amount = 1.0 / (n + 1)
        target = 1.0 / (n + 1)
        locations = []
        constant = 1.0 / np.sqrt(2*np.pi)
        for i in range(len(z_bar_range)-1):
            per_sum += 0.5 * dx * (constant * np.exp((-z_bar_range[i]**2) / 2.0) + constant * np.exp((-z_bar_range[i+1]**2) / 2.0))
            # per_sum += dx * constant * np.exp((-z_bar_range[i]**2) / 2.0)
            z_cdf[i+1] = min(per_sum, 1.0) 
            if per_sum >= target and len(locations) < int(n/2):
                locations.append((z_bar_range[i]+z_bar_range[i+1])/2.0)
                target += amount

        z_cdf[len(z_cdf) - 1] = 1.0
        lim = int(n/2)
        if n % 2 == 1:
            locations.append(0.0)
        for j in range(lim):
            i = lim - j - 1
            locations.append(-locations[i])    
        return np.array(locations)
    
    def compute_normality_bonus(n, mean1, bars, mean2, std2):
        for i in range(len(bars) -1):
            if bars[i] > bars[i+1]:
                return upper_bonus
        av_std = 0.0
        for i in range(n):
            cur_portion = (i + 1.0) / (n + 1)
            idx = bisect.bisect_left(z_cdf, cur_portion)
            cand_std = (bars[i] - mean1) / (z_bar_range[idx] + 1e-6) 
            if cand_std < 0:
                return upper_bonus
            av_std += cand_std
        std1 = av_std / (n)

        # compute D_KL( N(mean2, std2) || N(mean1, std1)) How much N(mean1, std1) is different from  N(mean2, std2)
        ue_bonus = np.log(std1/(std2+1e-6)) + (std2**2 ) / (2*std1**2) - 0.5
        return ue_bonus
        # return min(ue_bonus, upper_bonus)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_d_v(data):
        obs, ep_times, ret  = data['obs'], data['ep_time'], data['ret']
        expected_var_factors = (mn_std**2 - cur_var) / cur_l
        expected_stds = np.sqrt(np.maximum(expected_var_factors * ep_times + cur_var, mn_std**2))
        ret = np.expand_dims(ret, axis=1)
        expected_stds = np.expand_dims(expected_stds, axis=1)
        # target for V^{D, \pi}
        loc = torch.as_tensor(expected_stds * base_loc + ret, dtype=torch.float32)

        # Quantile regression
        pairwise_delta = ac.d_v(obs) - loc
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta ** 2 * 0.5)
        loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
        return loss.mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    d_vf_optimizer = Adam(ac.d_v.parameters(), lr=d_vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        d_v_l_old = compute_loss_d_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Distributional Value function learning (d_v)
        for i in range(train_v_iters):
            d_vf_optimizer.zero_grad()
            loss_d_v = compute_loss_d_v(data)
            loss_d_v.backward()
            mpi_avg_grads(ac.d_v)    # average grads across MPI processes
            d_vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))

        logger.store(LossDV=d_v_l_old,
            DeltaLossDV=(loss_d_v.item() - d_v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len, ep_ue_ret = env.reset(), 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    base_loc = compute_standard_normal_locations(d_output)
    ep_rewards = []
    ep_lens = []
    for epoch in range(epochs):
        ep_time = 0
        order_count = 0
        expected_stds = []
        for t in range(local_steps_per_epoch):
            a, d_v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            order_count += ac.order_check(torch.as_tensor(o, dtype=torch.float32))
            d_v_bars = ac.get_d_v(torch.as_tensor(o, dtype=torch.float32)) 
            next_o, r, d, _ = env.step(a)

            # compute standard deviation with episode time step
            expected_var_factor = (mn_std**2 - cur_var) / cur_l
            expected_std = np.sqrt(max(expected_var_factor * ep_time + cur_var, mn_std**2))
            expected_stds.append(expected_std)

            # normality bonus
            u_bonus = alpha * compute_normality_bonus(d_output, d_v, d_v_bars, d_v, expected_std) # calculate uncertainty bonus
            ep_ue_ret += u_bonus

            ep_ret += r
            ep_len += 1
            
            # print(loc[d_mid], v, d_v_bars[d_mid], np.mean(d_v_bars))

            buf.store(o, a, r + u_bonus, logp, d_v, ep_time)
            ep_time += 1

            logger.store(VVals=d_v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, UEBonus=ep_ue_ret)
                    if len(ep_rewards) >= 50:
                        ep_rewards.pop(0)
                    ep_rewards.append(ep_ret)
                    if len(ep_lens) >= 50:
                        ep_lens.pop(0)
                    ep_lens.append(ep_len)
                o, ep_ret, ep_len, ep_time, ep_ue_ret = env.reset(), 0, 0, 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # update R_t and l
        cur_l = np.mean(ep_lens)
        cur_var = np.std(ep_rewards)**2
        upper_bonus = max(np.mean(ep_rewards) / (cur_l + 0.001), 1.0)
        print('cur_l:', cur_l, 'cur_std:', np.sqrt(cur_var), 'upper_bonus:', upper_bonus)
        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('UEBonus', with_min_and_max=True)
        logger.log_tabular('LossDV', average_only=True)
        logger.log_tabular('DeltaLossDV', average_only=True)
        logger.log_tabular('MeanStd', np.mean(expected_stds))
        logger.log_tabular('OrderRatio', order_count / local_steps_per_epoch)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    mc_clt_ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
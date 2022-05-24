# MC_CLT

## Descriptions
Learning a predictive model of the mean return, or value function, plays a critical role in many reinforcement learning algorithms. Distributional reinforcement learning (DRL) methods instead model the value distribution, which has been shown to improve performance in many settings. In this paper, we model the value distribution as approximately normal using the Markov Chain central limit theorem. We analytically compute quantile bars to provide a new DRL target that is informed by the decrease in standard deviation that occurs over the course of an episode. In addition, we suggest an exploration strategy based on how closely the learned value distribution resembles the target normal distribution to make the value function more accurate for better policy improvement. The approach we outline is compatible with many DRL structures. We use proximal policy optimization as a testbed and show that both the normality-guided target and exploration bonus produce performance improvements. We demonstrate our method outperforms DRL baselines on a number of continuous control tasks.

## Dependencies
To install the dependencies below:
<pre>
<code>
pip install -e .
</code>
</pre>
* cloudpickle==1.2.1
* gym[atari,box2d,classic_control]~=0.15.3
* ipython
* joblib
* matplotlib==3.1.1
* mpi4py
* numpy
* pandas
* pytest
* psutil
* scipy
* seaborn==0.8.1
* tensorflow>=1.8.0,<2.0
* torch==1.3.1
* tqdm

### Train MC_CLT_PPO 
* Walk
<pre>
<code>
python -m spinup.run mc_clt_ppo --hid "[64,32]" --env Hopper-v3 --exp_name Hopper-v3 --seed 5000 --alpha 0.05 --mn_std 500
</code>
</pre>

### References 
* https://github.com/openai/spinningup
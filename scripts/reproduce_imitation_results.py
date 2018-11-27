# reproduce the imitation learning results (table 2 from the paper)

from joblib import Parallel, delayed
import pandas as pd
import itertools
import tensorflow as tf

from airl.models.imitation_learning import GAIL, AIRLStateAction, GAN_GCL
from airl.models.airl_state import AIRL

from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv

from airl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from joblib import Memory
from airl.algos.irl_trpo import IRLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from airl.version import MUJOCO_VERSION
JOBLIB_DIR = 'joblib_cache_mj' + MUJOCO_VERSION
DATA_DIR = 'data_mj' + MUJOCO_VERSION

imitation_algos = ['airl_state_action']#['gan_gcl','gail','airl','airl_state_action']
baseline_algos = ['expert','random'] # compare RL algorithms
environments = ['pendulum','ant','swimmer','half_cheetah']

algo_string_to_model = {
    'gan_gcl': GAN_GCL,
    'gail': GAIL,
    'airl_state_action': AIRLStateAction,
    'airl': AIRL
}

def dict_product(**items):
    def dictify(*values):
        return {k: v for k, v in zip(items.keys(), values)}
    return itertools.starmap(dictify, itertools.product(*items.values()))

# expt_configs = dict_product(algo=imitation_algos,
#                             environment=environments,
#                             seed=range(5))

expt_configs = dict_product(algo=imitation_algos,
                            seed=range(5),
                            environment=['pendulum']
                            )

# cartesian product of data frames
# HT https://stackoverflow.com/a/49325914/351392
def df_prod(df1,df2):
    return (df1.assign(__key__=1)
    .merge(df2.assign(__key__=1), on="__key__")
    .drop("__key__", axis=1)
    )

def get_env(env_name):
    if env_name == 'pendulum':
        return TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))
    else:
        raise ValueError('no env builder for' + env_name)


def get_demos(env_name):
    if env_name == 'pendulum':
        return load_latest_experts(DATA_DIR + '/pendulum', n=5)
    else:
        raise ValueError('no demo loader for ' + env_name)



def run_expt(config):
    env_name = config['environment']
    env = get_env(env_name)
    experts = get_demos(env_name)
    irl_model = algo_string_to_model[config['algo']](env_spec=env.spec,expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    # use params for each env
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=200,
        batch_size=2000 if env_name=='pendulum' else 10000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=1.0 if env_name == 'pointmass' else 0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )
    dirname = DATA_DIR + "/___".join([str(k) + "=" + str(v) for k,v in config.items()])
    with rllab_logdir(algo=algo, dirname=dirname):
        with tf.Session():
            algo.train()
    # a little clumsy but it's the easiest way, as rllab logger doesn't keep data around after
    # it's been written to disk
    train_results = pd.read_csv(dirname + '/progress.csv')
    # return originaltaskaverageReturn for last iteation
    output = config.copy()
    output['return'] = train_results.iloc[-1]['OriginalTaskAverageReturn']
    return output
# HT code from https://github.com/joblib/joblib/issues/490#issue-205090793
memory = Memory(JOBLIB_DIR, verbose=0)
run_expt_cached = memory.cache(run_expt)

results = Parallel(n_jobs=2)(delayed(run_expt_cached)(config) for config in expt_configs)
df = pd.DataFrame(results)
print(df)

print(df.groupby(['algo','environment']).agg({'return': ['mean','sem']}))
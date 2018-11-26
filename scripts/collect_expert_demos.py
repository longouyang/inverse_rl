from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from airl.utils.log_utils import rllab_logdir

import airl.envs # registers custom envs for us

from airl.version import MUJOCO_VERSION
JOBLIB_DIR = 'joblib_cache_mj' + MUJOCO_VERSION
DATA_DIR = 'data_mj' + MUJOCO_VERSION

if MUJOCO_VERSION == '1.3.1':
    env_names_to_ids = {'swimmer': 'Swimmer-v1',
                        'pendulum': 'Pendulum-v0',
                        'halfcheetah': 'HalfCheetah-v1',
                        'ant': 'airl/CustomAnt-v0'
                        }
else:
    env_names_to_ids = {'swimmer': 'Swimmer-v2',
                        'pendulum': 'Pendulum-v0',
                        'halfcheetah': 'HalfCheetah-v2',
                        'ant': 'airl/CustomAnt-v0'
                        }

def main(env_name, n_itr, batch_size, max_path_length, ent_wt):
    env_id = env_names_to_ids[env_name]
    env = TfEnv(GymEnv(env_id, record_video=False, record_log=False))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=n_itr,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname=DATA_DIR + env_id):
        algo.train()



if __name__ == "__main__":
    main(env_name='ant',
         n_itr = 200,
         batch_size = 1000,
         max_path_length= 100
         )

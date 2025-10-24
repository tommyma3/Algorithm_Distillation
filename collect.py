import os
from datetime import datetime
import yaml
import multiprocessing

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted
from algorithm import ALGORITHM, HistoryLoggerCallback
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from utils import get_config, get_traj_file_name



def worker(arg, config, traj_dir, env_idx, history, file_name):
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    config['device'] = 'cpu'

    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)
    callback = HistoryLoggerCallback(config['env'], env_idx, history)
    log_name = f'{file_name}_{env_idx}'
    
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=1,
              tb_log_name=log_name,
              reset_num_timesteps=True,
              progress_bar=True)
    env.close()



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    config = get_config("config/env/darkroom.yaml")
    config.update(get_config("config/algorithm/ppo_darkroom.yaml"))

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        
    traj_dir = 'datasets'

    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)
    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'datasets/{file_name}.hdf5'
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    with multiprocessing.Manager() as manager:
        history = manager.dict()

        # Create a pool with a maximum of n_workers
        with multiprocessing.Pool(processes=config['n_process']) as pool:
            # Map the worker function to the environments with the other arguments
            pool.starmap(worker, [(total_args[i], config, traj_dir, i, history, file_name) for i in range(n_envs)])

        # Save the history dictionary
        with h5py.File(path, 'w-') as f:
            for i in range(n_envs):
                env_group = f.create_group(f'{i}')
                for key, value in history[i].items():
                    env_group.create_dataset(key, data=value)

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
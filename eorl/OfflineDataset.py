import os, gzip, subprocess
import numpy as np


class OfflineDataset:

    def __init__(self, env, dataset_size=500000, train_split=1., obs_only=False, verbose=1):
        """
        ds = OfflineDataset(
            env = 'Pong',            # one env only
            dataset_size = 500000,   # [0, 1e7) frames of atari
            train_split = 0.9,       # 90% training, 10% held out for testing
            obs_only = False,        # only get observations (no actions, rewards, dones)
            verbose = 1              # 0 = silent, >0 for reporting
        )
        """
        assert(0 < dataset_size < 1e7), 'dataset_size must be in (0, 1e7)'
        self.dataset_size = dataset_size
        assert (0 <= train_split <= 1.), 'train_split must be in [0, 1.]'
        self._train_hx = int(train_split*dataset_size)
        self._obs_only = obs_only
        self._verbose = verbose
        self._env_list = self._list_envs()
        assert env in self._env_list, f'\'env\' must be one of {self._env_list}'
        self.env = env
        self.dataset = self._get_dataset(self.env)
        self._train_ix = 0
        self._test_ix = 0


    def _list_envs(self):
        stdout, stderr = subprocess.Popen(
            ['gsutil', 'ls', 'gs://atari-replay-datasets/dqn'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).communicate()
        if stderr: raise Exception('gsutil failed with the following error: ' + stderr)
        ls = stdout.replace('gs://atari-replay-datasets/dqn/', '').replace('/', '')
        ls = ls.split('\n')[1:-1]
        return ls


    def _unzip(self, fn):
        f = gzip.GzipFile(fn, "rb")
        ds = np.load(f)
        f.close()
        cp = ds[-self.dataset_size:]
        del ds
        return cp


    def _get_dataset(self, env):
        """ download batch-rl data to /.data if it doesn't already exist """
        want = ['observation'] if self._obs_only else ['observation','action','reward','terminal']
        fns = [f'./.data/{env}/$store$_{x}_ckpt.50.gz' for x in want]

        if all(os.path.exists(x) for x in fns):
            if self._verbose: print('decompressing data...')
            return {want[i]: self._unzip(fns[i]) for i in range(len(want))}

        os.makedirs(f'./.data/{env}', exist_ok=True)
        cmd = ['gsutil', '-m', 'cp', '-R',
               f'gs://atari-replay-datasets/dqn/{env}/1/replay_logs/*50*',
               f'./.data/{env}']  # 50 is ~expert data

        if self._verbose: print('downloading data...')

        stdout, stderr = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).communicate()

        if 'Operation completed' in stdout+stderr:
            if self._verbose: print('decompressing data...')
            return {want[i]: self._unzip(fns[i]) for i in range(len(want))}
        else:
            raise Exception(f'gsutil download for {env} failed: \n{stdout}\n{stderr}')


    def batch(self, batch_size=128, shuffle=False, split='train'):
        """
        obs, acts, rwds, dones = ds.batch(
            batch_size = 128   # number of samples
            shuffle = False    # chronological samples if False, randomly sampled if true
            split = 'train'    # `train` or `test`; specify split in class constructor
        )
        if self.only_obs is true, then you only get observations
        """
        if shuffle:
            if split == 'train':
                mask = np.random.randint(0, self._train_hx, size=batch_size)
            elif split == 'test':
                mask = np.random.randint(self._train_hx, self.dataset_size, size=batch_size)
            else:
                raise Exception('\'split\' must be either \'train\' or \'test\'')
        else:
            if split == 'train':
                if self._train_ix+batch_size >= self._train_hx:
                    wrap = np.arange(self._train_ix, self._train_hx-1)
                    self._train_ix = (self._train_ix + batch_size) % self._train_hx
                    xtra = np.arange(0, self._train_ix)
                    mask = np.hstack((wrap, xtra))
                else:
                    mask = slice(self._train_ix, self._train_ix+batch_size)
                    self._train_ix += batch_size
            elif split == 'test':
                if self._test_ix+batch_size >= self.dataset_size:
                    wrap = np.arange(self._test_ix, self.dataset_size-1)
                    self._test_ix = self._train_hx + ((self._test_ix+batch_size) % (self.dataset_size-self._train_hx))
                    xtra = np.arange(self._train_hx, self._test_ix)
                    mask = np.hstack((wrap, xtra))
                else:
                    mask = slice(self._test_ix, self._test_ix+batch_size)
                    self._test_ix += batch_size
            else:
                raise Exception('\'split\' must be either \'train\' or \'test\'')

        if self._obs_only:
            return self.dataset['observation'][mask]
        else:
            return tuple(self.dataset[x][mask] for x in ['observation', 'action', 'reward', 'terminal'])


if __name__ == '__main__':
    ds = OfflineDataset('Pong')
    obs, actions, rewards, dones = ds.batch()
    print(obs.shape)

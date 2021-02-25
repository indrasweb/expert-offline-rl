import os, gzip, subprocess
import numpy as np
from itertools import cycle

class OfflineDataset:

    def __init__(self,
                 env,
                 dataset_size=500000,
                 train_split=1.,
                 obs_only=False,
                 framestack=1,
                 shuffle=False,
                 stride=1,
                 verbose=1):
        """
            ds = OfflineDataset(
                env = 'Pong',            # one env only
                dataset_size = 500000,   # [0, 1e6) frames of atari
                train_split = 0.9,       # 90% training, 10% held out for testing
                obs_only = False,        # only get observations (no actions, rewards, dones)
                framestack = 1,          # number of frames per sample
                shuffle = False          # chronological samples if False, randomly sampled if true
                stride = 1               # return every stride`th chunk (where chunk size == `framestack)
                verbose = 1              # 0 = silent, >0 for reporting
            )
        """
        assert type(dataset_size) == int and 0 < dataset_size < 1e6, 'dataset_size must be in (0, 1e6)'
        self.dataset_size = dataset_size

        assert (framestack >= 1), 'framestack must be >= 1'
        self.framestack = framestack

        assert (0 <= train_split <= 1.), 'train_split must be in [0, 1.]'
        self._split_ix = int(train_split*dataset_size)

        assert obs_only in [True, False], 'obs_only must be true or false'
        self.obs_only = obs_only

        assert type(verbose) == int and verbose >= 0, 'verbose must be an int >= 0'
        self.verbose = verbose

        assert obs_only in [True, False], 'shuffle (return batches chronologically or not) must be true or false'
        self.shuffle = shuffle

        assert type(stride) == int and stride > 0, 'stride must be an int >0'

        self._env_list = self._list_envs()
        assert env in self._env_list, f'\'env\' must be one of {self._env_list}'
        self.env = env

        self.dataset = self._get_dataset(self.env)
        self.dataset_size = len(self.dataset['observation'])

        if self.shuffle:
            self._train_indices = self.random_chunks(np.arange(self._split_ix), framestack)
            self._test_indices = self.random_chunks(np.arange(self._split_ix, self.dataset_size), framestack)
        else:
            self._train_indices = self.rolling_chunks(np.arange(self._split_ix), framestack, stride)
            self._test_indices = self.rolling_chunks(np.arange(self._split_ix, self.dataset_size), framestack, stride)


    @staticmethod
    def _list_envs():
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


    @staticmethod
    def rolling_chunks(L, chunk_size, stride=1):
        """ slide a chunk_size window over L with given stride.
        """
        for i in cycle(range(0, len(L) - chunk_size, stride)):
            yield L[i:i + chunk_size]


    @staticmethod
    def random_chunks(L, chunk_size):
        """ slide a chunk_size window over L with given stride.
        """
        idx = np.arange(0, len(L) - chunk_size - 1)
        np.random.shuffle(idx)
        for i in cycle(idx):
            yield L[i:i + chunk_size]


    def _unzip(self, fn):
        f = gzip.GzipFile(fn, "rb")
        d = np.load(f)
        f.close()
        cp = d[-self.dataset_size:].copy()
        del d
        return cp


    def _get_dataset(self, env):
        """ download batch-rl data to /.data if it doesn't already exist """
        want = ['observation'] if self.obs_only else ['observation', 'action', 'reward', 'terminal']
        fns = [f'./data/{env}/$store$_{x}_ckpt.50.gz' for x in want]

        if all(os.path.exists(x) for x in fns):
            if self.verbose: print('decompressing data...')
            return {want[i]: self._unzip(fns[i]) for i in range(len(want))}

        os.makedirs(f'./data/{env}', exist_ok=True)
        cmd = ['gsutil', '-m', 'cp', '-R',
               f'gs://atari-replay-datasets/dqn/{env}/1/replay_logs/*50*',
               f'./data/{env}']  # 50 is ~expert data

        if self.verbose:
            print('downloading data...')

        stdout, stderr = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ).communicate()

        if 'Operation completed' in stdout+stderr:
            if self.verbose:
                print('decompressing data...')
            return {want[i]: self._unzip(fns[i]) for i in range(len(want))}
        else:
            raise Exception(f'gsutil download for {env} failed: \n{stdout}\n{stderr}')


    def batch(self, batch_size=128, split='train'):
        """
            obs, acts, rwds, dones, next_obs = ds.batch(
                batch_size = 128   # number of samples
                split = 'train'    # `train` or `test`; specify split in constructor
            )
        """
        assert split in ['train', 'test'], '`split` must be `train` or `test`'

        if split == 'train':
            mask = list(next(self._train_indices) for _ in range(batch_size))
        else:
            mask = list(next(self._test_indices) for _ in range(batch_size))

        obs = np.stack([self.dataset['observation'][m] for m in mask])
        if self.obs_only:
            return obs
        next_obs = np.stack([self.dataset['observation'][m+1] for m in mask])
        actions = np.stack([self.dataset['action'][m[-1]] for m in mask])
        rewards = np.stack([self.dataset['reward'][m[-1]] for m in mask])
        dones = np.stack([self.dataset['terminal'][m[-1]] for m in mask])

        return obs, actions, rewards, dones, next_obs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = OfflineDataset('Pong', dataset_size=100000, framestack=4)
    obs, actions, rewards, dones, next_obs = ds.batch()
    print('obs', obs.shape)
    print('actions', actions.shape)
    print('rewards', rewards.shape)
    print('dones', dones.shape)
    print('next_obs', next_obs.shape)
    print(actions[0])
    for i in range(4):
        plt.imshow(np.concatenate((obs[0, i, :, :], next_obs[0, i, :, :])))
        plt.show()


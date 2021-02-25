## Expert Offline Atari RL Datasets

A simple interface to [batch-rl](https://github.com/google-research/batch_rl). Expert data for all 60 Atari 2006 environments.

### Install

```bash
$ pip install git+https://github.com/indrasweb/expert-offline-rl.git
```

### Usage

```python
from eorl import OfflineDataset


ds = OfflineDataset(
    env = 'Pong',            # one env only
    dataset_size = 200000,   # [0, 1e7) frames of atari
    train_split = 0.9,       # 90% training, 10% held out for testing
    obs_only = False,        # only get observations (no actions, rewards, dones)
    verbose = 1              # 0 = silent, >0 for reporting
)

obs, actions, rewards, dones, next_obs = ds.batch(batch_size=128, shuffle=False)
```

Dataset is loaded into memory. Large `dataset_size` needs large amount of memory.

### Supported Environments

All 60 Atari 2006 environments. Each dataset was collected from a DQN agent trained on 200 million frames of the following `NoFrameSkip-v4` gym environments:

```
AirRaid
Alien
Amidar
Assault
Asterix
Asteroids
Atlantis
BankHeist
BattleZone
BeamRider
Berzerk
Bowling
Boxing
Breakout
Carnival
Centipede
ChopperCommand
CrazyClimber
DemonAttack
DoubleDunk
ElevatorAction
Enduro
FishingDerby
Freeway
Frostbite
Gopher
Gravitar
Hero
IceHockey
Jamesbond
JourneyEscape
Kangaroo
Krull
KungFuMaster
MontezumaRevenge
MsPacman
NameThisGame
Phoenix
Pitfall
Pong
Pooyan
PrivateEye
Qbert
Riverraid
RoadRunner
Robotank
Seaquest
Skiing
Solaris
SpaceInvaders
StarGunner
Tennis
TimePilot
Tutankham
UpNDown
Venture
VideoPinball
WizardOfWor
YarsRevenge
Zaxxon
```

#### Contribute

Could decompress to disk and only load ~50k samples into memory on demand.
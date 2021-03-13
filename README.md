## Expert Offline Reinforcement Learning Datasets

A simple interface to [batch-rl](https://github.com/google-research/batch_rl). Expert data for all 60 Atari 2006 and 2 continuous control environments.

### Install

```bash
$ pip install git+https://github.com/indrasweb/expert-offline-rl.git
```

### Usage

```python
from eorl import OfflineDataset


ds = OfflineDataset(
    env = 'Pong',            # pass name in `supported environments` below
    dataset_size = 200000,   # [0, 1e7) frames of atari
    train_split = 0.9,       # 90% training, 10% held out for testing
    obs_only = False,        # only get observations (no actions, rewards, dones)
    verbose = 1              # 0 = silent, >0 for reporting
)

obs, actions, rewards, dones, next_obs = ds.batch(batch_size=128, shuffle=False)
```

Dataset is loaded into memory. Large `dataset_size` needs large amount of memory. Use <400k in Colab.

### Supported Environments

Continuous control environments (Box2D) - 100k expert steps of each:

```
LunarLanderContinuous-v2
MountainCarContinuous-v0
BipedalWalker-v3
```

All Atari 2006 environments (Discrete)- 1 million steps of each. 
Each dataset was collected from a DQN agent trained on 200 million frames of the following `NoFrameSkip-v4` gym environments:

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

Decompressing to disk on Colab takes >10 mins which is annoying, hence we decompress to memory. Could add a switch that lets the user choose, thus allowing for larger DS size (e.g. grab ~50k samples from disk on demand).

# Deep Learning Planner

## Install

```shell
git clone git@gitlab.geometryrobot.com:learning/rl/deep_learning_planner.git
```

## Usage

#### lfd training

```shell
python scripts/transformer_train.py
```

#### drl training

```shell
roslaunch isaac_sim simple_navigation map:=$scene

$ISAAC_SIM_PYTHON scripts/drl_train.py --scene $scene
```

#### deploy

the imitation mode or reinforcement mode could be adjusted with shell args, sim or real robot as well

```
python scripts/deploy/transformer_planner.py
```

if deploy to agv234,  mapping lidar point cloud from high resolution to low resolution in the simulator is necessary

```
python scripts/real_robot/transfer_lidar.py
```

#### evaluation

not totally implemented yet, only a basic framework without fully test

## Reference

- DRL-VO
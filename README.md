# Reinforcement Learning-based Parameter Control for Evolutionary and Swarm-based Algorithms
This repository stores the implementation of the parameter control method proposed in the paper "Distributed Reinforcement Learning forOut-of-the-box Parameter Control in Evolutionaryand Swarm-based Algorithms" submitted to the IEEE Transactions on Evolutionary Computation, still waiting for approval.

These are the softwares and their respective versions that I used in the experiments to produce the paper:
* Python 3.6.6;
* GCC 7.3.0;
* OpenMPI 3.1.1;
* Ray 0.8.4 (RLLib);
* Tensorflow 2.1.0;
* Numpy1.18.1;
* Scipy 1.4.1;
* gym 0.15.4.

The proposed controller was implemented using the library Ray, where distributed implementations of the TD3 and PBT algorithms are available. The metaheuristics were implemented manually and are available in the file metaheuristics.py. The file metaheuristic_environment.py implements the gym.Env interface so that it can be used as environment for RL algorithms implemented in the library Ray. The file trainable_wrapper.py implements the interface ray.tune.Trainable so that it can be used as trainable by the PBT algorithm implemented in the module ray.tune. The file run_rl.py starts an experiment passing the metaheuristic, the RL algorithms, the optimization problem and their parameters. The parameters of the TD3 and ApexDDPG are hardcoded (it will be different in the future). You can use this file as an example of an experiment so that you can build yours, or call it as the following example:

```
python run_rl.py "127.0.0.1:6379" td3.TD3Trainer HCLPSO hclpso_config.json cec17_config/cec17_func1_10dim.json CEC17 16 2 td3_hclpso_func1 validation_data 1
```

It is important to mention that it assumes that there is a Ray's head node started running locally in the port 6379. You will find an example of the command to start such a node:

```
ray start --block --head --redis-port=6379 --redis-password="123456" --memory=20000000000 --object-store-memory=20000000000 --num-cpus=48 &
```

Also, it is necessary to build the file cec17_test_func.c as follows (if you want to use any of the CEC17 benchmark functions as the optimization problem):

```
gcc -fPIC -shared -lm -o cec17_test_func.so cec17_test_func.c
```


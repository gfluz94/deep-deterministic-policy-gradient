## TRAINING A DEEP DETERMINISTIC POLICY GRADIENT (DDPG) AGENT

![](./media/agent.gif)


This repository gathers all the code required for an agent that uses Deep Deterministic Policy Gradient, in order to interact with a MuJoCo environment.

First of all, we used **Python** to write our scripts not only for algorithm training and serving but also for the orchestration of the whole process. Important packages within this environment are listed below:

* `torch` so we could wdesign and train our neural networks;
* `numpy` so we could easily manipulate arrays and data structures;
* `gym` and `pybullets` so we could interact with RL environments.

Finally, we used GitHub actions to build CI pipeline, with the help of a `Makefile`:

* __Installing packages__: we used `pip` and a `requirements.txt` file to list all required packages (`make install`);
* __Formatting__: `black` was used (`make format`);
* __Linting__: `pylint` was used (`make lint`);
* __Testing__: `pytest` was used (`make test`).

____

This project is structured as follows:

#### main.py

Script where code is executed either for training or testing (interaction with environment) once agent has already been trained.

```py
python3 main.py [train|test] --env [ENV_NAME] [+ parameters]
```

#### tests

Folder where all unit tests are located.

#### ddpg

Project folder structure, where all classes and methods are contained.

```sh
ddpg
├── __init__.py
├── agent.py
├── models
│   ├── __init__.py
│   ├── actor.py
│   ├── critic.py
│   ├── errors.py
│   └── utils.py
└── replay_buffer
    ├── __init__.py
    ├── errors.py
    ├── experience.py
    └── transition.py
```
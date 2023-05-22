# LuxAI-S1
LuxAI Season 1 Challange with Multi-agent Reinforcement Learning

LuxAI gym environment from glmcdona:
https://github.com/glmcdona/LuxPythonEnvGym
## Get started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)

Install dependencies:
```
poetry install
```
Needs to be manually installed:
```
poetry shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip3 install gym=="0.22.0" stable-baselines3
```


Train agents against fixed opponent:
```
poetry run python -m luxai.luxrl.trainer.train 
```
Train agents via self-play:
```
poetry run python -m luxai.luxrl.ma.train_pfsp
```
Train agents with team-sprit:
```
poetry run python -m luxai.luxrl.ma.train_pfsp --team-sprit 0.2
```

Train agents with experiment tracking in weight&biases:
```
poetry run python -m luxai.luxrl.ma.train_pfsp --team-sprit 0.2 --use-wandb
```

For more configuration, check
[config.py](https://github.com/birajsth/LuxAI-S1/blob/main/luxai/luxrl/config.py)
##

## Implementation Details


### Observation Space
LuxAI is a game of complete information where both the competitors have the same game information. The game observation consists of information about general game states including phase i.e Night/Day, turn, information about the resources that are spread on the square grid, team information which includes the number of citytiles and units the corresponding team holds, research points, total fuel and each entities have their own individual information such as resource cargo, fuel, cooldown level etc. These shared observations are processed by the player to generate separate individual observation for each unit and citytile. 
We structure the observation for the learning agent into two main components:
1.	Scalar Features:
Scalar features include information about the game state, team state and individual agent information which can be city tile or unit. Some additional information including available action that the agent is able to perform and last actions performed by the agent is also included in the scalar feature.
2.	Spatial Features:
Spatial feature is a matrix representing the game map which consist of information about the resources and game entities.  LuxAI map are of  variable size ranging from 12x12 to 32x32,  but the spatial range for the individual agent are cropped to be of 12x12 size. The individual agent make decision based on their local spatial information but does have knowledge about other game stages and team information.

### Action Spaces
Action space refers to the set of actions that an agent can take in the environment.  At each turn, the game-playing agents are required to provide actions for the units and city tiles. In LuxAI, units and city tiles can perform actions each turn given certain conditions. In general, all actions are simultaneously applied and are validated against the state of the game at the start of a turn.  
LuxAI includes both discrete action such as move action and continuous action: the amount of resource to transfer. For simplicity, all the action are mapped to be discrete action.

Action Spaces can be divided into two main categories:
1.	Main Actions: It includes 7 actions: Move, Transfer, Build Citytile, Spawn Worker, Spawn Cart, Pillage, None. 
2.	Sub Actions: It includes 3 actions which further have its own sub actions:
1.	Move Direction: North, West, East, South, Center
2.	Transfer Direction: North, West, East, South, Center
3.	Transfer Amount: 0, 20,40, 60, 80
Certain action was hand-scripted logic rather than the policy: the resources to be transferred. Resources are selected on the order of uranium, coal and wood if the target unit was Cart and whereas wood was prioritized first for Worker. While it is possible the agent could learn more complex policy if these actions were not scripted, it would add a lot of complexity and training time while having only relatively small boost in the game performance even if the agent were to learn optimal resource selection policy. Additionally, the agent was only allowed to take viable actions, with illegal actions masked by setting the logits to negative infinity.


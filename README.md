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
![](https://github.com/birajsth/LuxAI-S1/blob/main/src/spatial%20range.jpg)
### Action Spaces
Action space refers to the set of actions that an agent can take in the environment.  At each turn, the game-playing agents are required to provide actions for the units and city tiles. In LuxAI, units and city tiles can perform actions each turn given certain conditions. In general, all actions are simultaneously applied and are validated against the state of the game at the start of a turn.  
LuxAI includes both discrete action such as move action and continuous action: the amount of resource to transfer. For simplicity, all the action are mapped to be discrete action.

Action Spaces can be divided into two main categories:
1.	Main Actions: It includes 7 actions: Move, Transfer, Build Citytile, Spawn Worker, Spawn Cart, Pillage, None. 
2.	Sub Actions: It includes 3 actions which further have its own sub actions:
1.	Move Direction: North, West, East, South, Center
2.	Transfer Direction: North, West, East, South, Center
3.	Transfer Amount: 0, 20,40, 60, 80
<br>Certain action was hand-scripted logic rather than the policy: the resources to be transferred. Resources are selected on the order of uranium, coal and wood if the target unit was Cart and whereas wood was prioritized first for Worker. While it is possible the agent could learn more complex policy if these actions were not scripted, it would add a lot of complexity and training time while having only relatively small boost in the game performance even if the agent were to learn optimal resource selection policy. Additionally, the agent was only allowed to take viable actions, with illegal actions masked by setting the logits to negative infinity.

## Neural network architecture
The model comprises two networks: the Actor Network (policy network) and the Critic Network (value network). The Actor Network has 2 encoder parts, 1 core part, and 4 action head parts. The Critic Network also has 2 encoder parts, 1 core part, but only 1 head part for value output.
The encoders in the model consist of a scalar encoder and a spatial encoder. The scalar encoder processes statistical information such as the game's current turn, agent-specific details, and team statistics. The spatial encoder processes feature maps like unit placement, cities, and resources. The outputs from both encoders are concatenated and fed into an LSTM (actor network) or MLP (critic network) in the core part, along with historical information. The core part's output is then passed to the output heads, which can be action heads or a baseline head.
### Encoder Structure

In the scalar encoder, the input is divided into separate elements representing game statistics, team statistics, and agent-specific information. Each element is embedded using a linear layer with ReLU activation, and the embedded outputs are appended into a scalar list. The elements in the list are concatenated and passed through another linear layer with ReLU activation, producing the encoder's output.

In the spatial encoder, the input is projected using a 2D convolution layer and ReLU activation. It then passes through several Residual blocks with squeeze excitation layer. Squeeze excitation layer improves the representational power of a network by adaptively weighting each feature channels, learning to prioritize informative channels while suppressing less relevant ones. The output is then through a linear layer with ReLU activation, generating the embedded spatial representation.


### Core Structure

In the Actor Network, the core consists of an LSTM layer. It takes the embedded scalar and embedded spatial representations as inputs, along with the previous hidden states. The embedded tensors are concatenated before being fed into the LSTM block, which outputs the new embedding and hidden states.

In the Critic Network, the core is an MLP layer. It also takes the embedded scalar and embedded spatial representations as input, concatenates them into one tensor, and passes them through a linear layer with ReLU activation to generate the new embedding.

### Head Structure

The Actor Network has four heads: action type head, move direction head, transfer direction head, and transfer amount head. Each head consists of multiple linear layers with ReLU activation. They take the embedding from the core as input, along with additional information such as available actions and the previous actions. The available actions are used for action masking, as certain actions depend on the game state. The action type head determines the action to be performed, which is then used by the other three heads to choose their respective actions.

The Critic Network includes a baseline head, which consists of linear layers with ReLU activation. It takes the embedding from the core as input and outputs a value representing the state's value.
![](https://github.com/birajsth/LuxAI-S1/blob/main/src/detail%20arch.jpg)


## Reinforcement Learning Algorithm
The policy is trained using Proximal Policy Optimization (PPO) algorithm, a variant of advantage actor-critic policy gradient algorithm, with slight modifications to accommodate the multi-agent LuxAI environment. 
Multi-agent proximal policy algorithm (MAPPO) trains two separate neural networks: an actor network, and a value function network (referred to as a critic).  Policy losses are computed by summing over the log probabilities of all the selected partial actions from 4 action heads, effectively computing the log of the joint probability of the whole action.
The learning agent is first trained with heavily shaped individual rewards (without team objective) against fixed opponent which can sustain in the environment for number of turns by mining resources from adjacent tiles. This allows agent to learn fundamental behaviours such as navigating and gathering resources. The agent is then trained via self-play with reward that also include team objective, allowing agent to learn both cooperative and competitive behaviours. The agent's final reward r is computed as a combination of its individual reward a ,and the team reward T, weighted by a team spirit parameter (τ). <br>
&emsp;&emsp;&emsp;&emsp;&emsp;                           r =(1- τ)xa + τxT
<br>The implementation uses prioritized fictitious self-play method, with variance weighted probability distribution on win-rates, to select the opponent.



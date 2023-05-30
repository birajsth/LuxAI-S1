import os
import random
import argparse
from distutils.util import strtobool



def get_config():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="LuxAI",
        help="the name of this experiment")
    parser.add_argument("--env-name", type=str, default="Luxai",
        help="the id of the gym environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--use-wandb", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="luxai",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--wandb-key", type=str, default=None,
        help="the wandb's login key")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--id', help='Identifier of this run', type=str, default=str(random.randint(0, 10000)))
    parser.add_argument("--use-artifact", type=str, default=None,
        help="load model from wandb artifact")
    parser.add_argument("--model-dir", type=str, default=None, 
        help="path to model")
    parser.add_argument("--checkpoint-dir", type=str, default=None, 
        help="path to checkpoint")
    parser.add_argument("--save-checkpoints", type=bool, default=True,
        help="whether to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=100,
        help="save frequency")
    parser.add_argument("--log-interval", type=int, default=10,
        help="log frequency")
    parser.add_argument("--use-eval", type=bool, default=False,
        help="whether to evaluate")
    parser.add_argument("--eval-interval", type=int, default=100000,
        help="eval frequency")
    
    # policy parameters
    parser.add_argument("--use-lstm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use Lstm in actor network")
    parser.add_argument("--use-squeeze-excitation", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use Squeeze Excitation in ResNet")
    
    # reward parameters
    parser.add_argument("--team-sprit", type=float, default=0.2,
        help="the factor of team sprit agent to be trained for")
    
    
    # optimizer parameters
    parser.add_argument("--lr", type=float, default=2.5e-4,
        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=2.5e-4,
        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
        help='optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # Algorithm specific arguments
    parser.add_argument("--algorithm-name", type=str, default="MAPPO",
        help="the name of the algorithm")
    parser.add_argument("--num-envs", type=int, default=2,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=360,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--use-gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.997225,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--use-popart", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles normalization with popart")
    parser.add_argument("--use-valuenorm", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles values normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    return parser
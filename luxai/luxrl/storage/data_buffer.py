import torch
import numpy as np
from ..utils.shapes import *
from ..utils.utils import index_nested_list



class DataBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space):

        self.num_envs = args.num_envs

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self.use_lstm = args.use_lstm

        self.obs_shape = get_shape_from_obs_space(obs_space)
        #self.share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.act_shape = get_shape_from_act_space(act_space)
        self.available_actions_shape = get_available_actions_shape()
        self.last_actions_shape = (3,)
        self.lstm_states_shape = get_lstm_states_shape()

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        self.obs_scalar = torch.empty((0,) + self.obs_shape["scalar"], dtype=torch.float32).to(self.device)
        self.obs_spatial = torch.empty((0,) + self.obs_shape["spatial"], dtype=torch.float32).to(self.device)
        self.available_actions = torch.empty((0,) + self.available_actions_shape, dtype=torch.float32).to(self.device)

        self.actions = torch.empty((0,) + self.act_shape, dtype=torch.float32).to(self.device)
        self.logprobs = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.rewards = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.dones = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.values = torch.empty((0,), dtype=torch.float32).to(self.device)

        if self.use_lstm:
            self.lstm_states_hidden= torch.empty((0,) + self.lstm_states_shape, dtype=torch.float32).to(self.device)
            self.lstm_states_cell = torch.empty((0,) + self.lstm_states_shape, dtype=torch.float32).to(self.device)
            self.lstm_states_dict = defaultdict(lambda: None)
            
        
        # to keep track of envs, episode and agents 
        # tuple(env_id, episode, agent_id)
        self.ids = []
        

        # to keep track of envs episode
        self.episodes = [0] * self.num_envs


    def insert(self, actions, action_log_probs, value_preds, rewards, hidden_states, dones):
        """
        Insert data into the buffer.
        
        :param actions:(tensor) actions taken by agents.
        :param action_log_probs:(tensor) log probs of actions taken by agents
        :param value_preds: (tensor) value function prediction
        :param rewards: (tensor) reward collected at agents
        :param hidden_states: tuple(tensor,tensor)  lstm states of agents
        :param dones: (tensor) denotes whether the agent has terminated or not.
        """
        self.obs_scalar = torch.cat((self.obs_scalar, self.next_obs_scalar), dim=0)
        self.obs_spatial = torch.cat((self.obs_spatial, self.next_obs_spatial), dim=0)
        self.available_actions = torch.cat((self.available_actions, self.next_available_actions), dim=0)
        if self.use_lstm:
            self.lstm_states_hidden = torch.cat((self.lstm_states_hidden, self.next_lstm_states_hidden), dim=0)
            self.lstm_states_cell = torch.cat((self.lstm_states_cell, self.next_lstm_states_cell), dim=0)

        self.actions = torch.cat((self.actions, actions), dim=0)
        self.logprobs = torch.cat((self.logprobs, action_log_probs), dim=0)
        self.values = torch.cat((self.values, value_preds), dim=0)
        self.rewards = torch.cat((self.rewards, rewards), dim=0)
        self.dones = torch.cat((self.dones, dones), dim=0)

        # update ids
        self.ids += self.next_ids
        # update lstm states dict
        if self.use_lstm:
            self.update_lstm_states(hidden_states)

    def update_lstm_states(self, hidden_states):
        """
        :param next_hidden_states: tuple(tensor,tensor)  
        """
        lstm_states_hidden, lstm_states_cell = hidden_states
        for i, id in enumerate(self.next_ids):
            self.lstm_states_dict[id] = [None, None]
            self.lstm_states_dict[id][0] = lstm_states_hidden[i]
            self.lstm_states_dict[id][1] = lstm_states_cell[i]
    

    def nextinsert(self, next_obs_scalar, next_obs_spatial, next_available_actions, next_ids):
        """
        :param obs_scalar: (tensor) local agent scalar observations.
        :param obs_spatial: (tensor) local agent spatial observations.
        :param available_actions: (tensor) actions available to each agent
        :param next_ids: (list) identifier(env, episode, agent_id) for each agent 
        """
        self.next_obs_scalar = next_obs_scalar
        self.next_obs_spatial = next_obs_spatial
        self.next_available_actions = next_available_actions
        self.next_ids = next_ids

        if self.use_lstm: 
            self.next_lstm_states_hidden = torch.zeros((len(next_ids),) + self.lstm_states_shape, dtype=torch.float32).to(self.device)
            self.next_lstm_states_cell = torch.zeros_like(self.next_lstm_states_hidden)
            for i, id in enumerate(next_ids):
                if self.lstm_states_dict[id]:
                    self.next_lstm_states_hidden[i] = self.lstm_states_dict[id][0]
                    self.next_lstm_states_cell[i] = self.lstm_states_dict[id][1] 


    def reset(self):
        """
        clears buffer data
        """
        del self.obs_scalar, self.obs_spatial, self.available_actions, self.actions, self.logprobs, self.rewards, self.dones, self.values
        self.obs_scalar = torch.empty((0,) + self.obs_shape["scalar"], dtype=torch.float32).to(self.device)
        self.obs_spatial = torch.empty((0,) + self.obs_shape["spatial"], dtype=torch.float32).to(self.device)
        self.available_actions = torch.empty((0,) + self.available_actions_shape, dtype=torch.float32).to(self.device)

        self.actions = torch.empty((0,) + self.act_shape, dtype=torch.float32).to(self.device)
        self.logprobs = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.rewards = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.dones = torch.empty((0,), dtype=torch.float32).to(self.device)
        self.values = torch.empty((0,), dtype=torch.float32).to(self.device)
        if self.use_lstm:
            del self.lstm_states_hidden, self.lstm_states_cell, self.lstm_states_dict
            self.lstm_states_hidden= torch.empty((0,) + self.lstm_states_shape, dtype=torch.float32).to(self.device)
            self.lstm_states_cell = torch.empty((0,) + self.lstm_states_shape, dtype=torch.float32).to(self.device)
            self.lstm_states_dict = defaultdict(lambda: None)
        self.ids = []

    @torch.no_grad()
    def compute_returns(self, next_values, next_ids, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        values = self.values.clone().detach()
        if self._use_popart or self._use_valuenorm:
            next_values = value_normalizer.denormalize(next_values) 
            values = value_normalizer.denormalize(values)

        # nested list of agent indices
        seperated_agent_index = index_nested_list(self.ids)
        next_value_dict = defaultdict(lambda : 0., zip(next_ids, next_values.detach()))
        
    
        if self._use_gae:
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            for agent_indx in seperated_agent_index:
                agent_id = self.ids[agent_indx[0]]
                lastgaelam = 0
                for t in reversed(range(len(agent_indx))):
                    nonterminal = 1.0 - self.dones[agent_indx[t]]
                    next_value = 0.
                    if nonterminal==1.0:
                        if t == len(agent_indx) - 1:
                            next_value = next_value_dict[agent_id]
                        else:
                            next_value = values[agent_indx[t+1]]
                    delta = self.rewards[agent_indx[t]] + self.gamma * next_value - values[agent_indx[t]]
                    self.advantages[agent_indx[t]] = lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam
            self.returns = self.advantages + values
        else:
            self.returns = torch.zeros_like(self.rewards).to(self.device)
            for agent_indx in seperated_agent_index:
                agent_id = self.ids[agent_indx[0]]
                for t in reversed(range(len(agent_indx))):
                    nonterminal = 1.0 - self.dones[agent_indx[t]]
                    next_value = 0.
                    if nonterminal==1.0:
                        if t == len(agent_indx) - 1:
                            next_value = next_value_dict[agent_id]
                        else:
                            next_value = values[agent_indx[t+1]]
                    self.returns[agent_indx[t]] = self.rewards[agent_indx[t]] + self.gamma * next_value 
            self.advantages = self.returns - values

    def feed_forward_generator(self, num_minibatch):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = len(self.ids)
        minibatch_size = batch_size // num_minibatch
        b_inds = np.arange(batch_size)
        sampler = []
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            sampler.append(b_inds[start:end])
        
        for indices in sampler:
            mb_obs_scalar = self.obs_scalar[indices]
            mb_obs_spatial = self.obs_spatial[indices] 
            mb_available_actions = self.available_actions[indices]

            mb_hidden_state = None
            if self.use_lstm:
                mb_hidden_state = (self.lstm_states_hidden[indices], self.lstm_states_cell[indices])
        
            mb_actions = self.actions[indices]  
            mb_logprobs = self.logprobs[indices]
            mb_values = self.values[indices]
            mb_returns = self.returns[indices]
            mb_advantages = self.advantages[indices]

            yield mb_obs_scalar, mb_obs_spatial, mb_available_actions, mb_hidden_state, \
                mb_actions, mb_logprobs, mb_values, mb_returns, mb_advantages
        
    
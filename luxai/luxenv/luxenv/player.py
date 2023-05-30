from collections import defaultdict
import numpy as np

from gym import spaces
from gym.spaces import Box, Dict, MultiBinary

from .agent import  AgentWithModel
from ..game.actions import *


from .observation import Observation
from .reward import DenseReward
from .action_spaces import action_code_to_action, heuristic_actions


AVAILABLE_ACTIONS = 19
NUM_SCALAR_FEATURES = 15 + 12 + 8
NUM_SPATIAL_FEATURES = 17

SPATIAL_WIDTH = 11
SPATIAL_HEIGHT = 11
AGENT_SPATIAL_SIZE = 11




########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class LuxPlayer(AgentWithModel):
    def __init__(self, mode="train", model=None, team_sprit=0.2) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects      
        self.agent_action_space = spaces.MultiDiscrete([7, 5, 5, 5])

        # Observation space: 
        self.agent_observation_space = Dict({
            "scalar": Box(0., 1., shape=(NUM_SCALAR_FEATURES,)),
            "spatial": Box(0., 1., shape=(NUM_SPATIAL_FEATURES, SPATIAL_WIDTH, SPATIAL_HEIGHT)),
            "available_actions": MultiBinary(AVAILABLE_ACTIONS)
            })
        
        self.reward_space = DenseReward(team_sprit=team_sprit, team=self.team)
       
        self.obs = None

        self.step = 0
        
    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.
            self.obs = Observation(game, team)
            #print("Num team city_tiles: ", self.obs.num_team_citytiles)
            #print("Num team units: ", self.obs.num_team_units)
            #print("total city_tiles: ", self.obs.num_total_citytiles)
        return self.obs.get_agent_obs(unit, city_tile)
        

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        unit, city_tile = self.obs.last_observation_objects.pop(0)
        team=self.team
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        if action:
            self.match_controller.take_action(action)
    
    def take_actions(self, game, action_codes, ignore_cooldown=True):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        team=self.team
        i = 0
        actions = []
        units = game.state["teamStates"][self.team]["units"].values()

        for unit in units:
            if ignore_cooldown:
                action = self.action_code_to_action(action_codes[i], game, unit, None, team)
                if action:
                    actions.append(action)
                i += 1 
            elif unit.can_act():
                action = self.action_code_to_action(action_codes[i], game, unit, None, team)
                if action:
                    actions.append(action)
                i += 1 
             
        citytiles_action = self.heuristic_actions(game, team)
        actions += citytiles_action
        self.match_controller.take_actions(actions)

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        return action_code_to_action(action_code, game, unit, city_tile, team)
    
    def heuristic_actions(self, game, team):
        return heuristic_actions(game, team)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0
        self.hidden_states = defaultdict(lambda : None)
        self.last_actions = defaultdict(lambda: np.array([AVAILABLE_ACTIONS-1]*3, dtype=np.float32))
    

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
    

    
    def get_agent_obs(self, game, agent_id=None, new_turn=False, ignore_cooldown=True):
        '''
        Returns agent's observation.
        If agent_id is not given, Iterates over all the team units and citytiles and returns their observation
        '''
        if agent_id is None:
            new_turn = True
            units = game.state["teamStates"][self.team]["units"].values()
            for unit in units:
                if ignore_cooldown:
                    yield unit.id, self.get_observation(game, unit, None, unit.team, new_turn)
                elif unit.can_act():
                    yield unit.id, self.get_observation(game, unit, None, unit.team, new_turn)
                new_turn = False
                    
        else:
            unit = game.get_unit(self.team, agent_id)
            return agent_id, self.get_observation(game, unit, None, unit.team, new_turn)
        

    def get_reward_done(self, game, agent_id, action, team_reward, match_over, game_won):
        '''
        Returns reward and done for agent.
        '''
        return self.reward_space.compute_rewards_and_done(game, agent_id, action, team_reward, match_over, game_won)

    

if __name__ == '__main__':
    pass
    

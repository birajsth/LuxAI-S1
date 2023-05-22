from .city import CityTile
import random
import sys
import time
import numpy as np
import traceback
import os

from .constants import Constants
from ..luxenv.agent import Agent



class GameStepFailedException(Exception):
    pass


class MatchController:
    def __init__(self, game, agents=[None, None], replay_validate=None) -> None:
        """

        :param game:
        :param agents:
        """
        self.action_buffer = []
        self.game = game
        self.agents = agents
        self.replay_validate = replay_validate

        if len(agents) != 2:
            raise ValueError("Two agents must be specified.")

        # Validate the agents
        self.training_agent_count = 0
        for i, agent in enumerate(agents):
            if not (issubclass(type(agent), Agent) or isinstance(agent, Agent)):
                raise ValueError("All agents must inherit from Agent.")
            if agent.get_agent_type() == Constants.AGENT_TYPE.LEARNING:
                self.training_agent_count += 1

            # Initialize agent
            agent.set_team(i)
            agent.set_controller(self)
        
        # Reset the agents, without resetting the game
        self.reset(reset_game=False)
        
        '''
        if self.training_agent_count > 1:
            raise ValueError("At most one agent must be trainable.")

        elif self.training_agent_count == 1:
            print("Running in training mode.", file=sys.stderr)

        elif self.training_agent_count == 0:
            print("Running in inference-only mode.", file=sys.stderr)
        '''

    def reset(self, reset_game=True, randomize_team_order=True):
        """

        :return:
        """  
        '''
        # Randomly re-assign teams of the agents
        if randomize_team_order:
            r = random.randint(0, 1)
            self.agents[0].set_team(r)
            self.agents[1].set_team((r + 1) % 2)
            '''

        # Reset the game as well if needed
        if reset_game:
            self.game.reset()
        self.action_buffer = []
        self.accumulated_stats = dict( {Constants.TEAM.A: {}, Constants.TEAM.B: {}} )

        # Call the agent game_start() callbacks
        for agent in self.agents:
            agent.game_start(self.game)

    def take_action(self, action):
        """
         Adds the specified action to the action buffer
         """
        if action is not None:
            # Validate the action
            try:
                if action.is_valid(self.game, self.action_buffer, self.accumulated_stats):
                    # Add the action
                    self.action_buffer.append(action)
                    self.accumulated_stats = action.commit_action_update_stats(self.game, self.accumulated_stats)
                else:
                    #print(f'action is invalid {action} turn {self.game.state["turn"]}: {vars(action)}', file=sys.stderr)
                    pass
                    
            except KeyError:
                print(f'action failed, probably a dead unit {action}: {vars(action)}', file=sys.stderr)

        # Mark the unit or city as not able to perform another action this turn
        actionable = None
        if hasattr(action, 'unit_id') and action.unit_id is not None:
            # Mark the unit as already-actioned this turn
            if action.unit_id in self.game.state["teamStates"][0]["units"]:
                actionable = self.game.state["teamStates"][0]["units"][action.unit_id]
            elif action.unit_id in self.game.state["teamStates"][1]["units"]:
                actionable = self.game.state["teamStates"][1]["units"][action.unit_id]

        elif hasattr(action, 'x') and action.x is not None:
            # Mark the city as already-actioned this turn
            cell = self.game.map.get_cell(action.x, action.y)
            if cell.is_city_tile():
                actionable = cell.city_tile

        if actionable is not None:
            actionable.set_can_act_override(False)

    def take_actions(self, actions):
        """
         Adds the specified action to the action buffer
        """
        if actions != None:
            for action in actions:
                self.take_action(action)

    def log_error(self, text):
        # Ignore errors caused by logger
        try:
            if text is not None:
                with open(self.log_dir + "match_errors.txt", "a") as o:
                    o.write(text + "\n")
        except Exception:
            print("Critical error in logging")

    def set_opponent_team(self, agent, team):
        """
        Sets the team of the opposing team
        """
        for a in self.agents:
            if a != agent:
                a.set_team(team)

    def run_to_next_observation(self):
        """ 
            Generator function that gets the observation at the next Unit/City
            to be controlled.
            Returns: bool, whether the game is over.
        """
        game_over = False
        turn = self.game.state["turn"]
        # Process this turn
        for agent in self.agents:
            if agent.get_agent_type() == Constants.AGENT_TYPE.AGENT:
                # Call the agent for the set of actions
                actions = agent.process_turn(self.game, agent.team)
                self.take_actions(actions)
            elif agent.get_agent_type() == Constants.AGENT_TYPE.LEARNING:
                pass 
        
        # Reset the can_act overrides for all units and city_tiles
        units = list(self.game.state["teamStates"][0]["units"].values()) + list(self.game.state["teamStates"][1]["units"].values())
        for unit in units:
            unit.set_can_act_override(None)
        for city in self.game.cities.values():
            for cell in city.city_cells:
                cell.city_tile.set_can_act_override(None)

        # Now let the game actually process the requested actions and play the turn
        try:
            game_over = self.game.run_turn_with_actions(self.action_buffer)
        except Exception as e:
            # Log exception
            self.log_error("ERROR: Critical error occurred in turn simulation.")
            self.log_error(repr(e))
            self.log_error(''.join(traceback.format_exception(None, e, e.__traceback__)))
            raise GameStepFailedException("Critical error occurred in turn simulation.")

        self.action_buffer = []

        if self.replay_validate is not None:
            self.game.process_updates(self.replay_validate['steps'][turn+1][0]['observation']['updates'], assign=False)
    
        return game_over
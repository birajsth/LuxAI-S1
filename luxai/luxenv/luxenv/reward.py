from abc import ABC, abstractmethod
import copy
import logging
import numpy as np
from scipy.stats import rankdata
from typing import Dict, NamedTuple, NoReturn, Tuple
from collections import defaultdict

from ..game.game import Game
from ..game.unit import Unit
from ..game.city import CityTile
from ..game.game_constants import GAME_CONSTANTS

from ..game.actions import *
from .action_spaces import action_types


from ..game.game_constants import GAME_CONSTANTS

WORKER_LIGHT_UPKEEP = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER"]
CART_LIGHT_UPKEEP = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CART"]
# weights
resources = {
    "wood": 0.01,
    "coal": 0.10,
    "uranium": 0.40
}

MAX_FUEL_COLLECT = 2 * 40



def count_city_tiles(game_state: Game) -> np.ndarray:
    city_tile_count = [0,0]
    for player in range(2):
        for city in game_state.cities.values():
            if city.team==player:
                city_tile_count[player] += 1
            else:
                city_tile_count[(player + 1) % 2] += 1
    #print("city_tile_count: ", city_tile_count)
    return np.array(city_tile_count)


def count_units(game_state: Game) -> np.ndarray:
    return np.array([len(game_state.state["teamStates"][player]["units"]) for player in range(2)])


def count_total_fuel(game_state: Game) -> np.ndarray:
    city_fuel = [0,0]
    for player in range(2):
        for city in game_state.cities.values():
            if city.team==player:
                city_fuel[player] += city.fuel
            else:
                city_fuel[(player + 1) % 2] += city.fuel
    #print("fuel: ", city_fuel)
    return np.array(city_fuel)


def count_research_points(game_state: Game) -> np.ndarray:
    return np.array([game_state.state["teamStates"][player]["researchPoints"] for player in range(2)])


def should_early_stop(game_state: Game) -> bool:
    ct_count = count_city_tiles(game_state)
    unit_count = count_units(game_state)
    ct_pct = ct_count / max(ct_count.sum(), 1)
    unit_pct = unit_count / max(unit_count.sum(), 1)
    return ((ct_count == 0).any() or
            (unit_count == 0).any() or
            (ct_pct >= 0.75).any() or
            (unit_pct >= 0.75).any())

MAX_FUEL_COLLECT = 2 * 40


def calculate_fuel(game_state: Game, unit: Unit = None, city_tile: CityTile = None):
    assert unit is not None or city_tile is not None
    if unit:
        fuel = sum([unit.cargo[key] * w for key, w in resources.items()])
    else:
        # get city_tile fuel from corresponding city
        for city in game_state.cities.values():
            if city.id == city_tile.city_id:
                fuel = city.fuel / len(city.city_cells)
                break
    return fuel

def check_night_on_city(game_state_new: Game, unit: Unit) -> bool:
    return game_state_new.map.get_cell_by_pos(unit.pos).is_city_tile() and game_state_new.is_night()

def is_new_day_cycle(game_state:Game) -> bool:
    return game_state.state["turn"] % 40 == 0

class AgentState:
    def __init__(self, fuel) -> None:
        self.fuel = fuel

class RewardSpec(NamedTuple):
    reward_min: float
    reward_max: float
    zero_sum: bool
    only_once: bool


class BaseRewardSpace(ABC):
    """
    A class used for defining a reward space and/or done state for either the full game or a sub-task
    """
    def __init__(self, **kwargs):
        if kwargs:
            logging.warning(f"RewardSpace received unexpected kwargs: {kwargs}")

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        pass

    def get_info(self) -> Dict[str, np.ndarray]:
        return {}


# Full game reward spaces defined below

class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        return self.compute_rewards(game_state, done), done

    @abstractmethod
    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        pass

class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=True,
            only_once=True
        )

    def __init__(self, early_stop: bool = False, **kwargs):
        super(GameResultReward, self).__init__(**kwargs)
        self.early_stop = early_stop

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        if not done:
            return 0., 0.

        # reward here is defined as the sum of number of city tiles with unit count as a tie-breaking mechanism
        rewards = [int(GameResultReward.compute_player_reward(game_state, p)) for p in range(2)]
        rewards = (rankdata(rewards) - 1.) * 2. - 1.
        return tuple(rewards)

    @staticmethod
    def compute_player_reward(game_state: Game, player:int):
        ct_count = sum([1 for city in game_state.cities.values() if city.team==player])
        unit_count = len(game_state.state["teamStates"][player]["units"])
        # max board size is 32 x 32 => 1024 max city tiles and units,
        # so this should keep it strictly so we break by city tiles then unit count
        '''
        print("city_tile_count: ", ct_count)
        print("unit_count: ", unit_count)
        '''
        return ct_count * 10000 +  unit_count
    

class TeamReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.,
            early_stop: bool = False,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.early_stop = early_stop

        self.city_count = np.empty((2,), dtype=float)
        self.unit_count = np.empty_like(self.city_count)

        self.weights = {
            "game_result": 3.,
            "city": 1.,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        super(TeamReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, done: bool) -> Tuple[Tuple[float, float], bool]:
        if self.early_stop:
            done = done or should_early_stop(game_state)
        return self.compute_rewards(game_state, done), done

    def compute_rewards(self, game_state: Game, done: bool) -> Tuple[float, float]:
        new_city_count = count_city_tiles(game_state)
        new_unit_count = count_units(game_state)
        self.unit_count[self.unit_count==0] = 1
        reward_items_dict = {
            "city": (new_city_count - self.city_count)/self.unit_count,
        }

        if done:
            game_result_reward = [int(GameResultReward.compute_player_reward(game_state, p)) for p in range(2)]
            game_result_reward = (rankdata(game_result_reward) - 1.) * 2. - 1.
            self._reset()
        else:
            game_result_reward = np.array([0., 0.])
            self.city_count = new_city_count
            self.unit_count = new_unit_count
        reward_items_dict["game_result"] = game_result_reward

        assert self.weights.keys() == reward_items_dict.keys()
        reward = np.stack(
            [self.weight_rewards(reward_items_dict[key] * w) for key, w in self.weights.items()],
            axis=0
        ).sum(axis=0)
        return tuple(reward / 4.0 /max(self.positive_weight, self.negative_weight))

    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward

    def _reset(self) -> NoReturn:
        self.city_count = np.ones_like(self.city_count)
        self.unit_count = np.ones_like(self.unit_count)


class AgentReward(BaseRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-1.,
            reward_max=1.,
            zero_sum=False,
            only_once=False
        )

    def __init__(
            self,
            positive_weight: float = 1.,
            negative_weight: float = 1.,
            team_sprit: float = .2,
            **kwargs
    ):
        assert positive_weight > 0.
        assert negative_weight > 0.
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.team_sprit = team_sprit
        self.weights = {
            "fuel_increase":0.01,
            "deposit": 0.01,
            "transfer": 0.01,
            "survived_game": 3.0,
            "dead": -3.0,
        }
        self.weights.update({key: val for key, val in kwargs.items() if key in self.weights.keys()})
        for key in copy.copy(kwargs).keys():
            if key in self.weights.keys():
                del kwargs[key]
        self.agent_states = defaultdict(lambda : 0.)
        super(AgentReward, self).__init__(**kwargs)
        self._reset()

    def compute_rewards_and_done(self, game_state: Game, agent_id: str,team_reward: float, match_over:bool) -> Tuple[float, bool]:
        # check if agent still exists
        if agent_id[0]=="u":
            unit = game_state.get_unit(0, agent_id)
            done = unit == None
            return self.compute_reward(game_state, agent_id, unit, team_reward, match_over, done), done
        return 0
    
    def compute_reward(self, game_state: Game, agent_id: str, unit: Unit, team_reward:float, match_over:bool, done: bool) -> float:
        fuel_old = self.agent_states[agent_id]
        reward_items_dict = {}
        if done:
            reward_items_dict["dead"] = 1
            self.agent_states[agent_id] = 0
        else:
            fuel_new = calculate_fuel(game_state, unit, None)
            self.agent_states[agent_id] = fuel_new
            reward_items_dict["fuel_increase"]: fuel_new -fuel_old
            if game_state.map.get_cell_by_pos(unit.pos).is_city_tile():
                reward_items_dict["deposit"]: min(fuel_old-fuel_new, 0)
            elif not game_state.is_night():
                reward_items_dict["transfer"]: min(fuel_old-fuel_new, 0)
        if match_over:
            reward_items_dict["survived_game"] = 1
            # clear state
            self.agent_states[agent_id] = 0.
        agent_reward = sum([self.weight_rewards(self.weights[key] * r) for key, r in reward_items_dict.items()])
        agent_reward = agent_reward /4./ max(self.positive_weight, self.negative_weight)
        return (1 - self.team_sprit) * agent_reward + self.team_sprit * team_reward


    def weight_rewards(self, reward: np.ndarray) -> np.ndarray:
        reward = np.where(
            reward > 0.,
            self.positive_weight * reward,
            reward
        )
        reward = np.where(
            reward < 0.,
            self.negative_weight * reward,
            reward
        )
        return reward
    
    def _reset(self) -> NoReturn:
        self.agent_states = defaultdict(lambda : 0.)
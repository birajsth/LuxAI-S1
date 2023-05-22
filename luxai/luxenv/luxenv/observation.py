import itertools
import copy
import numpy as np

from gym.spaces import Dict, MultiBinary, Box

from ..game.constants import Constants
from ..game.game_constants import GAME_CONSTANTS
from ..game.position import Position

from .utils.util import cropped_spatial
from .action_spaces import get_available_actions

MAX_RESOURCE = {
    Constants.RESOURCE_TYPES.WOOD: GAME_CONSTANTS["PARAMETERS"]["MAX_WOOD_AMOUNT"],
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L253
    Constants.RESOURCE_TYPES.COAL: 425.,
    # https://github.com/Lux-AI-Challenge/Lux-Design-2021/blob/master/src/Game/gen.ts#L269
    Constants.RESOURCE_TYPES.URANIUM: 350.
}
WORKER_CAPACITY = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
CART_CAPACITY = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
DN_CYCLE_LEN = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] + GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]

WORKER_COOLDOWN = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * 2. - 1.
CART_COOLDOWN = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["CART"] * 2. - 1.
CITY_LIGHT_UPKEEP = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"]
WORKER_LIGHT_UPKEEP = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["WORKER"]
CART_LIGHT_UPKEEP = GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CART"]
CITY_COOLDOWN = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]

MAX_ROAD = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

cargo_size = [WORKER_CAPACITY, CART_CAPACITY]
unit_cooldown = [WORKER_COOLDOWN, CART_COOLDOWN]
unit_light_upkeep = [WORKER_LIGHT_UPKEEP, CART_LIGHT_UPKEEP]
MAX_FUEL = 30 * 10 * 9
MAX_DISTANCE_CENTER = 16


AVAILABLE_ACTIONS = 22
NUM_AGENT_FEATURES = 15
NUM_TEAM_FEATURES = 10
NUM_GAME_FEATURES = 8
NUM_SCALAR_FEATURES = 15 + 10 + 8
NUM_SPATIAL_FEATURES = 16

SPATIAL_WIDTH = 12
SPATIAL_HEIGHT = 12
AGENT_SPATIAL_SIZE = 12



class Observation:
    def __init__(self, game, team) -> None:
        super(Observation, self).__init__()
        self._empty_obs = {}
        self.game = game
        self.team = team

        self.scalar_obs = np.zeros((NUM_SCALAR_FEATURES,), dtype=np.float32)
        self.spatial_obs = np.zeros((NUM_SPATIAL_FEATURES, self.game.map.width, self.game.map.height), dtype=np.float32)
    
        self.num_total_citytiles = 0
        self.num_total_units = 0
        self.num_team_citytiles = 0
        self.num_team_units = 0
        self.num_team_workers = 0
        self.num_team_carts = 0
        
        # number of night turns left for current phase
        self.rem_night_turns = min(DN_CYCLE_LEN - (self.game.state["turn"] % DN_CYCLE_LEN), GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"])

        self.get_spatial_obs()
        self.get_scalar_obs()
        self.team_features = self.get_team_features()
        assert self.team_features.shape == (NUM_TEAM_FEATURES, )
        self.game_features = self.get_game_features()
        assert self.game_features.shape == (NUM_GAME_FEATURES, )

        
    
    def get_agent_obs_spec(self):
        return Dict({
            "scalar": Box(0., 1., shape=(NUM_SCALAR_FEATURES,)),
            "spatial": Box(0., 1., shape=(NUM_SPATIAL_FEATURES, SPATIAL_WIDTH, SPATIAL_HEIGHT)),
            "available_actions": MultiBinary((AVAILABLE_ACTIONS,))
        })
    
    def get_agent_obs(self, unit=None, city_tile=None):
        assert (unit is not None or city_tile is not None), "Both unit and citytile are None"
        agent_scalar = copy.deepcopy(self.scalar_obs)
        agent_scalar[:NUM_AGENT_FEATURES] = self.get_agent_features(unit, city_tile)[:]
        assert agent_scalar.shape == self.get_agent_obs_spec().spaces["scalar"].shape

        agent_pos = [unit.pos.x, unit.pos.y] if unit is not None else [city_tile.pos.x, city_tile.pos.y]
        agent_spatial = copy.deepcopy(self.spatial_obs)
        agent_spatial[0, agent_pos[0], agent_pos[1]] = 1
        agent_spatial = cropped_spatial(agent_spatial, agent_pos , AGENT_SPATIAL_SIZE)
        
        assert agent_spatial.shape == self.get_agent_obs_spec().spaces["spatial"].shape
    
        available_actions = get_available_actions(self.game, unit, city_tile, self.num_team_citytiles)
        return {
            "scalar": agent_scalar,
            "spatial": agent_spatial,
            "available_actions": available_actions
        }

    def get_scalar_obs(self) -> None:
        team_features = self.get_team_features()
        assert team_features.shape == (NUM_TEAM_FEATURES, )
        game_features = self.get_game_features()
        assert game_features.shape == (NUM_GAME_FEATURES, )
        self.scalar_obs[NUM_AGENT_FEATURES:NUM_AGENT_FEATURES+NUM_TEAM_FEATURES] = team_features[:]
        self.scalar_obs[NUM_AGENT_FEATURES+NUM_TEAM_FEATURES:] = game_features
        
    
    def get_spatial_obs(self) -> None:
        idx = {
            # Whether agent's present
            "agent": 0,
            # Whether team's present
            "team": 1,
            # Whether opponent's present
            "opponent": 2,
            # has citytile/worker/cart
            "citytile": 3,
            "worker": 4,
            "cart": 5,
            # number of unit present, Normalized from 0-5
            "num_units": 6,
            # Normalized from 0-MAX_FUEL_CAPACITY(citytile)
            "fuel": 7,
            # Normalized from 0-MAX_CARGO_CAPACITY(worker/cart), averaged if multiple present
            "cargo": 8,
            # has team that needs resources(won't survive the night)
            "need_resource": 8,
            # Normalized from 0-MAX_COOLDOWN(citytile/worker/cart)
            "cooldown": 10,
            # has resource
            "resource": 11,
            # Amount of wood in tile, Normalized from 0-MAX_WOOD
            "wood": 12,
            # Amount of coal in tile, Normalized from 0-MAX_COAL
            "coal": 13,
            # Amount of uranium in tile, Normalized from 0-MAX_URANIUM
            "uranium": 14,
            # Normalized from 0-MAX_ROAD_LEVEL
            "road_level": 15,
        }

        # add resources and road_level to obs
        for cell_column in itertools.chain(self.game.map.map):
            for cell in cell_column:
                x, y = cell.pos.x, cell.pos.y
                if cell.has_resource():
                    self.spatial_obs[idx["resource"], y, x] = 1
                    self.spatial_obs[idx[f"{cell.resource.type}"], y, x] = cell.resource.amount / MAX_RESOURCE[cell.resource.type]
                else:
                    if len(cell.units) > 0:
                        self.spatial_obs[idx["num_units"], y, x] = min(len(cell.units), 5) / 5
                self.spatial_obs[idx["road_level"], y, x] = cell.road / MAX_ROAD
                    
        
        # add units to obs
        for t in [self.team, (self.team + 1) % 2]:
            for u in self.game.state["teamStates"][self.team]["units"].values():
                x, y = u.pos.x, u.pos.y
                self.num_total_units += 1
                if t == self.team:
                    self.spatial_obs[idx["team"], y, x] = 1  
                    self.num_team_units += 1
                else:
                    self.spatial_obs[idx["opponent"], y, x] = 1

                cargo = u.cargo["wood"] + u.cargo["coal"] + u.cargo["uranium"]
                if u.is_worker():
                    self.num_team_workers += 1 if t == self.team else 0
                    self.spatial_obs[idx["worker"], y, x] = 1 
                    self.spatial_obs[idx["cargo"], y, x] = cargo / WORKER_CAPACITY if t == self.team else 0
                    self.spatial_obs[idx["need_resource"], y, x] = 1 if t == self.team and cargo < WORKER_LIGHT_UPKEEP * self.rem_night_turns else 0
                    self.spatial_obs[idx["cooldown"], y, x] = u.cooldown / WORKER_COOLDOWN
                else:
                    self.num_team_carts += 1 if t == self.team else 0
                    self.spatial_obs[idx["cart"], y, x] = 1
                    self.spatial_obs[idx["cargo"], y, x] = cargo / CART_CAPACITY if t == self.team else 0
                    self.spatial_obs[idx["need_resource"], y, x] = 1 if t == self.team and cargo < CART_LIGHT_UPKEEP * self.rem_night_turns else 0 
                    self.spatial_obs[idx["cooldown"], y, x] = u.cooldown / CART_COOLDOWN


        # add citytiles to obs
        for city in self.game.cities.values():
            for cell in city.city_cells:
                self.num_total_citytiles += 1
                x, y = cell.city_tile.pos.x, cell.city_tile.pos.y
                if city.team == self.team:
                    self.num_team_citytiles += 1
                    self.spatial_obs[idx["team"], y, x] = 1  
                else:
                    self.spatial_obs[idx["opponent"], y, x] = 1
                self.spatial_obs[idx["citytile"], y, x] = 1
                self.spatial_obs[idx["fuel"], y, x] = city.fuel / MAX_FUEL / len(city.city_cells)
                self.spatial_obs[idx["need_resource"], y, x] = 1 if t == self.team and (city.fuel / len(city.city_cells)) < CITY_LIGHT_UPKEEP * self.rem_night_turns else 0
                self.spatial_obs[idx["cooldown"], y, x] = cell.city_tile.cooldown / CITY_COOLDOWN
    
    def get_team_features(self) -> dict:
        team_features = {
            # For Both Teams
            # Normalized from 0-Total_citytiles
            "team_citytiles": self.num_team_citytiles / max(self.num_total_citytiles, 1),
            # Normalized form 0-Total_units
            "team_units": self.num_team_units / max(self.num_total_units, 1),
            # Normalized from 0-Total_Team_units
            "team_workers": self.num_team_workers / max(self.num_team_units, 1),
            # Normalized from 0-Total_Team_units
            "team_carts": self.num_team_carts / max(self.num_team_units, 1),
            # Normalized from 0-200
            "team_research_points": self.game.state["teamStates"][self.team]["researchPoints"] / MAX_RESEARCH,
            "opponent_research_points": self.game.state["teamStates"][(self.team + 1) % 2]["researchPoints"] / MAX_RESEARCH,
            # Coal is researched
            "team_researched_coal": 1 if self.game.state["teamStates"][self.team]["researched"]["coal"] else 0,
            "opponent_researched_coal": 1 if self.game.state["teamStates"][(self.team + 1) % 2]["researched"]["coal"] else 0,
            # Uranium is researched
            "team_researched_uranium": 1 if self.game.state["teamStates"][self.team]["researched"]["uranium"] else 0,
            "opponent_researched_uranium": 1 if self.game.state["teamStates"][(self.team + 1) % 2]["researched"]["uranium"] else 0,
        }
        return np.array(list(team_features.values()), dtype=np.float32)


    def get_game_features(self) -> dict:
        game_features = {
            # Normalized from 0-360
            "turn": self.game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"],
            # The turn number // 40, Normalized from 0-TOTAL_DAY_NIGHT_CYCLE
            "phase": min(
                self.game.state["turn"] // DN_CYCLE_LEN,
                GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"] / DN_CYCLE_LEN - 1
                ),
            # The turn number % 40, Normalized from 0-DAY_NIGHT_LENGTH
            "day_night_cycle": (self.game.state["turn"] % DN_CYCLE_LEN)/DN_CYCLE_LEN,
            # Whether it's night 
            "is_night": 1.0 if (self.game.state["turn"] % DN_CYCLE_LEN) >= GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] else 0.0,
            "board_size_12": 1.0 if self.game.map.width == 12 else 0.0,
            "board_size_16": 1.0 if self.game.map.width == 12 else 0.0,
            "board_size_24": 1.0 if self.game.map.width == 12 else 0.0,
            "board_size_32": 1.0 if self.game.map.width == 12 else 0.0,
        }
        return np.array(list(game_features.values()), dtype=np.float32)
    
    def get_agent_features(self, unit=None, city_tile=None) -> np.array:
        assert unit is not None or city_tile is not None
        idx = {
            # none, citytile/worker/cart
            "citytile": 0,
            "worker": 1,
            "cart": 2,
            # Normalized from 0-MAX_FUEL(citytile)
            "fuel": 3,
            # Normalized form 0-MAX_CARGO_CAPACITY
            "wood": 4,
            "coal": 5,
            "uranium": 6,
            # Percentage of cargo filled
            "cargo": 5,
            # Can build citytile
            "can_build_citytile": 6,
            # Normalized from Citytile/Worker/Cart MAX_COOLDOWN
            "cooldown": 7,
            # Whether the citytile/worker/cart can survive the night
            "will_survive_night": 8,
            # Direction to center, [CENTER, NORTH, WEST, SOUTH, EAST]
            "direction_to_center": [9, 10, 11, 12, 13],
            # Distance to center, Divided by MAX_DISTANCE_CENTER
            "distance_to_center": 14,
        }
        agent_features = np.zeros(NUM_AGENT_FEATURES, dtype=np.float32)
        pos = None
        if city_tile is not None:
            pos = city_tile.pos
            agent_features[0] = 1
            # get city_tile fuel from corresponding city
            for city in self.game.cities.values():
                if city.id == city_tile.city_id:
                    agent_features[idx["fuel"]] = city.fuel / MAX_FUEL / len(city.city_cells)
                    break
            agent_features[idx["cooldown"]] = city_tile.cooldown / CITY_COOLDOWN
            agent_features[idx["will_survive_night"]] = 1.0 
        elif unit is not None:
            pos = unit.pos
            cargo = unit.cargo["wood"] + unit.cargo["coal"] + unit.cargo["uranium"]
            if unit.is_worker():
                agent_features[idx["worker"]]= 1.0
                agent_features[idx["wood"]] = unit.cargo["wood"] / WORKER_CAPACITY
                agent_features[idx["coal"]] = unit.cargo["coal"] / WORKER_CAPACITY
                agent_features[idx["uranium"]] = unit.cargo["uranium"] / WORKER_CAPACITY
                agent_features[idx["cooldown"]] = unit.cooldown / WORKER_COOLDOWN
                agent_features[idx["can_build_citytile"]] = 1 if cargo == WORKER_CAPACITY else 0
                agent_features[idx["will_survive_night"]] = 1 if cargo >= WORKER_LIGHT_UPKEEP * self.rem_night_turns else 0
            else:
                agent_features[idx["cart"]] = 1.0
                agent_features[idx["wood"]] = unit.cargo["wood"] / CART_CAPACITY
                agent_features[idx["coal"]] = unit.cargo["coal"] / CART_CAPACITY
                agent_features[idx["uranium"]] = unit.cargo["uranium"] / CART_CAPACITY
                agent_features[idx["cooldown"]] = unit.cooldown / CART_COOLDOWN
                agent_features[idx["will_survive_night"]] = 1 if cargo >= CART_LIGHT_UPKEEP * self.rem_night_turns else 0

        center = Position(self.game.map.width // 2, self.game.map.height //2)
        direction = pos.direction_to(center)
        mapping = {
            Constants.DIRECTIONS.CENTER: 0,
            Constants.DIRECTIONS.NORTH: 1,
            Constants.DIRECTIONS.WEST: 2,
            Constants.DIRECTIONS.SOUTH: 3,
            Constants.DIRECTIONS.EAST: 4,
        }
        agent_features[idx["direction_to_center"][mapping[direction]]] = 1.0  
        agent_features[idx["distance_to_center"]] = pos.distance_to(center) / MAX_DISTANCE_CENTER

        return agent_features
    

       
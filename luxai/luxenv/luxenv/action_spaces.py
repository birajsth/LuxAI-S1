
import numpy as np

from gym.spaces import MultiDiscrete

from ..game.actions import *
from ..game.game_constants import GAME_CONSTANTS


    
action_types = [
    MoveAction,
    TransferAction,
    SpawnCityAction,
    None,
]
ACTION_SPACE = MultiDiscrete([4, 4, 4])


WORKER_CAPACITY = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
CART_CAPACITY = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"]
UNIT_TYPES = Constants.UNIT_TYPES
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=True)
RESOURCES = Constants.RESOURCE_TYPES
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

def action_code_to_action(action_code, game, unit=None, city_tile=None, team=None):
    # Map action_code index into to a constructed Action object

    x = None
    y = None
    if unit is not None:
        x = unit.pos.x
        y = unit.pos.y
    else: 
        return None 
    action_type = action_types[action_code[0]]
    if action_type == MoveAction:
        return MoveAction(game=game,
                            unit_id=unit.id if unit else None,
                            team=team,
                            direction=DIRECTIONS[action_code[1]])
    elif action_type == TransferAction:
        dest_pos = unit.pos.translate(direction=DIRECTIONS[action_code[2]], units=1)
        dest_cell = game.map.get_cell_by_pos(dest_pos)
        try:
            dest_unit = list(dest_cell.units.values())[0]
        except:
            return None

        # Prioritize transfering urainum than coal than wood if worker(transfering to cart)
        if unit.is_worker():
            if unit.cargo["uranium"] > 0:
                resource_type = "uranium"       
            elif unit.cargo["coal"] > 0:
                resource_type = "coal"
            else:
                resource_type = "wood"
            # transfer all 
            amount = unit.cargo[resource_type]
        # Prioritize transfering wood first if cart(transfering to worker)
        else:
            if unit.cargo["wood"] > 0:
                resource_type = "wood"
            elif unit.cargo["coad"] > 0:
                resource_type= "coal"
            else:
                resource_type="uranium"
            # transfer amount that fit in worker 
            dest_unit_cargo_rem = dest_unit.cargo["wood"] + dest_unit.cargo["coal"] + dest_unit.cargo["uranium"] - WORKER_CAPACITY
            amount = min(dest_unit_cargo_rem - WORKER_CAPACITY, unit.cargo[resource_type])
        return TransferAction(game=game,
                            team=team,
                            source_id=unit.id,
                            destination_id=dest_unit.id, 
                            resource_type = resource_type,
                            amount=amount)
    elif action_type == None:
        return None
    else:
        return action_type(game=game,
                            unit_id=unit.id ,
                            unit=unit,
                            team=team,
                            x=x,
                            y=y)
   

def heuristic_actions(game, team):
    '''heuristic action for citytile'''
    actions = []

    is_night = game.is_night()
    research_points = game.state["teamStates"][team]["researchPoints"]
    
    cities = [city for city in game.cities.values() if city.team == team]
    num_citytiles = sum([len(city.city_cells) for city in cities])
    num_workers = 0
    num_carts = 0
    for unit in game.state["teamStates"][team]["units"].values():
        if unit.is_worker():
            num_workers += 1
        else:
            num_carts += 1

    num_spawnable_units = max(num_citytiles - (num_workers+num_carts), 0)
    for city in cities:
        for cell in city.city_cells:
            city_tile = cell.city_tile
            if city_tile and city_tile.cooldown<1:
                
                # SpwanUnitAction
                # create 1 cart for every 5 units if workers > max size // 2
                if num_spawnable_units > 0:
                    if num_workers>(game.map.width//2) and  num_workers / max(num_workers+num_carts, 1) > .8:
                        actions.append(SpawnCartAction(game=game,
                                        city_id=city_tile.city_id,
                                        citytile=city_tile,
                                        unit_id=None,
                                        unit=None,
                                        team=team,
                                        x=city_tile.pos.x,
                                        y=city_tile.pos.y))
                        num_carts += 1
                    else:
                        actions.append(SpawnWorkerAction(game=game,
                                        city_id=city_tile.city_id,
                                        citytile=city_tile,
                                        unit_id=None,
                                        unit=None,
                                        team=team,
                                        x=city_tile.pos.x,
                                        y=city_tile.pos.y))
                        num_workers += 1
                    num_spawnable_units -= 1
                # research only if citytile number > map size//2
                elif num_citytiles>game.map.width//2 and research_points<MAX_RESEARCH:
                    actions.append(ResearchAction(game=game,
                                        city_id=city_tile.city_id,
                                        citytile=city_tile,
                                        unit_id=None,
                                        unit=None,
                                        team=team,
                                        x=city_tile.pos.x,
                                        y=city_tile.pos.y))
                    research_points += 1

    return actions
    
    

def get_available_actions(game, unit=None):
    '''
    Returns: available actions
    '''
    action_types = np.zeros((4,), dtype= np.int64)
    action_types[3] = 1
    move_directions = np.zeros((4,), dtype= np.int64)
    transfer_directions = np.zeros((4,), dtype= np.int64)
    
    if unit is not None and unit.cooldown<1:
        # Move Actions
        # Move Center
        for i, direction in enumerate(DIRECTIONS[:4]):
            new_cell = game.map.get_cell_by_pos(
                unit.pos.translate(direction, 1)
            )
            if new_cell is not None:
                if new_cell.is_city_tile():
                    if new_cell.city_tile.team == unit.team:
                        action_types[0] = move_directions[i] = 1 
                elif not new_cell.has_units():
                    action_types[0] = move_directions[i] = 1
                else:
                    #Transfer Actions
                    target_unit = list(new_cell.units.values())[0]
                    # Workers can only transfer to Cart and viceversa
                    if target_unit.team == unit.team and target_unit.type != unit.type:
                        if unit.cargo["wood"]>0 or unit.cargo["coal"]>0 or unit.cargo["uranium"]>0:
                            action_types[1] = transfer_directions[i] = 1

        # SpawnCityAction
        if unit.type == Constants.UNIT_TYPES.WORKER and unit.can_build(game.map):
            action_types[2] = 1 

    return np.concatenate([action_types, move_directions, transfer_directions],dtype=np.float32)
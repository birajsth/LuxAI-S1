
import numpy as np
from functools import partial

from gym.spaces import MultiDiscrete

from ..game.actions import *
from ..game.game_constants import GAME_CONSTANTS


    
action_types = [
    MoveAction,
    TransferAction,
    SpawnCityAction,
    SpawnWorkerAction,
    SpawnCartAction,
    ResearchAction,
    None,
]
action_amounts = [0, .2, .4, .6, .8]

ACTION_SPACE = MultiDiscrete([8, 5, 5])


UNIT_TYPES = Constants.UNIT_TYPES
DIRECTIONS = Constants.DIRECTIONS.astuple(include_center=True)
RESOURCES = Constants.RESOURCE_TYPES
MAX_RESEARCH = max(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"].values())

def action_code_to_action(action_code, game, unit=None, city_tile=None, team=None):
    # Map action_code index into to a constructed Action object

    x = None
    y = None
    if city_tile is not None:
        x = city_tile.pos.x
        y = city_tile.pos.y
    elif unit is not None:
        x = unit.pos.x
        y = unit.pos.y
    
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

        # Prioritize transfering urainum than coal than wood
        if unit.cargo["uranium"] > 0:
            resource_type = "uranium"
                    
        elif unit.cargo["coal"] > 0:
            resource_type = "coal"
            amount = unit.cargo["coal"]*action_code[3]
        else:
            resource_type = "wood"
            amount = unit.cargo["wood"]*action_code[3]

        amount = unit.cargo[resource_type]*action_code[3]
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
                            unit_id=unit.id if unit else None,
                            unit=unit,
                            city_id=city_tile.city_id if city_tile else None,
                            citytile=city_tile,
                            team=team,
                            x=x,
                            y=y)
   


    
    

def get_available_actions(game, unit=None, city_tile=None, num_team_city_tiles=0):
    '''
    Returns: available actions
    '''

    action_types = np.zeros((7,), dtype= np.int64)
    move_directions = np.zeros((5,), dtype= np.int64)
    move_directions[4] = 1
    transfer_directions = np.zeros((5,), dtype= np.int64)
    transfer_directions[4] = 1
    transfer_amounts = np.ones((5,), dtype= np.int64)
    
    if unit is not None and unit.cooldown<1:
        # Move Actions
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
                    if target_unit.team == unit.team:
                        if unit.cargo["wood"]>0 or unit.cargo["coal"]>0 or unit.cargo["uranium"]>0:
                            action_types[1] = transfer_directions[i] = 1

        # SpawnCityAction
        if unit.type == Constants.UNIT_TYPES.WORKER and unit.can_build(game.map):
            action_types[2] = 1 
        
        # PilageAction
        # Not used for now
        #if unit_cell.road > 0 and not unit_cell.city_tile:
            #action_types[6] = 1
        
    # City Actions
    elif city_tile is not None and city_tile.cooldown<1:
        # ResearchAction
        if game.state["teamStates"][city_tile.team]["researchPoints"] < MAX_RESEARCH:
           action_types[5] = 1
        # SpwanUnitAction
        if len(game.state["teamStates"][city_tile.team]["units"]) < num_team_city_tiles:
           action_types[3] = 1
           action_types[4] = 1

    # None action is no action available
    if np.all(action_types==0):
        action_types[6] = 1

    return np.concatenate([action_types, move_directions, transfer_directions, transfer_amounts],dtype=np.float32)
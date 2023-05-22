import enum
from collections import namedtuple



# for the scalar feature
ScalarFeatureSize = namedtuple('ScalarFeature',['num_agent_features', 'num_team_features', 'num_game_features', 'num_available_actions', 'num_last_actions_encoding', 'num_last_actions',
                                                'total'])

Scalar_Feature_Size = ScalarFeatureSize(num_agent_features = 15,
                                        num_team_features = 10,
                                        num_game_features = 8,
                                        num_available_actions = 22,
                                        num_last_actions_encoding=14,
                                        num_last_actions=3,
                                        total = 42 
                                        )

# for the spatial feature
SpatialFeatureSize = namedtuple('SpatialFeature',['height', 'width', 'num_spatial_features'])

Spatial_Feature_Size = SpatialFeatureSize(height = 11,
                                          width = 11,
                                          num_spatial_features = 16,
                                          )

ActionSize = namedtuple('ActionSize', ['num_unit_actions','num_citytile_actions','num_action_types', 'num_move_directions', 'num_transfer_directions', 'num_transfer_amounts', 'num_available_actions'])
Action_Size = ActionSize(num_unit_actions=10,
                         num_citytile_actions=4,
                         num_action_types=7,
                         num_move_directions=5,
                         num_transfer_directions=5,
                         num_transfer_amounts=5,
                         num_available_actions=22)

# for the arch model parameters
ArchHyperParameters = namedtuple('ArchHyperParameters', ['batch_size',
                                                         'scalar_encoder_fc1_input',
                                                         'scalar_encoder_fc2_input',
                                                         'scalar_feature_size',
                                                         'n_resblocks',
                                                         'n_resblocks_out',
                                                         'temperature',
                                                         'use_action_type_mask',
                                                         'core_hidden_dim',
                                                         'lstm_layers',
                                                         'original_512',
                                                         'original_256',
                                                         'original_128',
                                                         'original_64',
                                                         'original_32',
                                                         'original_16',
                                                         'original_8',
                                                         'context_size'])

Arch_Hyper_Parameters = ArchHyperParameters(batch_size=16,
                                            scalar_encoder_fc1_input=64,
                                            scalar_encoder_fc2_input=64,
                                            scalar_feature_size=10,
                                            n_resblocks=6,
                                            n_resblocks_out=4,
                                            temperature=0.8,
                                            use_action_type_mask=1,
                                            core_hidden_dim=128,
                                            lstm_layers=2,
                                            original_512=512,
                                            original_256=256,
                                            original_128=128,
                                            original_64=64,
                                            original_32=32,
                                            original_16=16,
                                            original_8=8,
                                            context_size=64)



if __name__ == '__main__':
    pass
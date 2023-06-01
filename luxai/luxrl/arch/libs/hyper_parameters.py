import enum
from collections import namedtuple



# for the scalar feature
ScalarFeatureSize = namedtuple('ScalarFeature',['num_agent_features', 'num_team_features', 'num_game_features', 'num_action_types',
                                                'total'])

Scalar_Feature_Size = ScalarFeatureSize(num_agent_features = 15,
                                        num_team_features = 12,
                                        num_game_features = 8,
                                        num_action_types = 4,
                                        total = 40 
                                        )

# for the spatial feature
SpatialFeatureSize = namedtuple('SpatialFeature',['height', 'width', 'num_spatial_features'])

Spatial_Feature_Size = SpatialFeatureSize(height = 11,
                                          width = 11,
                                          num_spatial_features = 17,
                                          )

ActionSize = namedtuple('ActionSize', ['num_action_types', 'num_move_directions', 'num_transfer_directions', 'num_available_actions'])
Action_Size = ActionSize(num_action_types=4,
                         num_move_directions=4,
                         num_transfer_directions=4,
                         num_available_actions=12)

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
                                                         'lstm_layers',])

Arch_Hyper_Parameters = ArchHyperParameters(batch_size=16,
                                            scalar_encoder_fc1_input=64,
                                            scalar_encoder_fc2_input=64,
                                            scalar_feature_size=10,
                                            n_resblocks=6,
                                            n_resblocks_out=4,
                                            temperature=0.8,
                                            use_action_type_mask=1,
                                            core_hidden_dim=128,
                                            lstm_layers=1)



if __name__ == '__main__':
    pass
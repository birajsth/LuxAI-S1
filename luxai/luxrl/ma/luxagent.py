from ...luxenv.luxenv.player import LuxPlayer

class LuxAgent:
    """A alphastar agent for starcraft.
    Demonstrates agent interface.
    In practice, this needs to be instantiated with the right neural network
    architecture.
    """

    def __init__(self, policy=None):
        self.policy = policy
        self.weights = policy.state_dict() if policy else None
        self.steps = 0

    def get_player(self, mode="train", team_sprit=0.2):
        return LuxPlayer(mode=mode, model=self.policy, team_sprit=team_sprit)
    
    def get_obs_space(self):
        return LuxPlayer().agent_observation_space
    
    def get_action_space(self):
        return LuxPlayer().agent_action_space

    def reset(self):
        pass

    def get_policy(self):
        return self.policy
    
    def set_policy(self, policy):
        self.policy = policy
        self.weights = policy.state_dict()

    def set_weights(self, weights):
        self.weights = weights
        self.policy.load_state_dict(weights)

    def get_weights(self):
        if self.policy is not None:
            return self.policy.state_dict()
        else:
            return None

    def get_parameters(self):
        return self.policy.parameters()
    
    def get_steps(self):
        """How many agent steps the agent has been trained for."""
        return self.steps

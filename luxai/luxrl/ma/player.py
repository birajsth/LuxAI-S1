" The code for the player, including MainPlayer, MainExploiter, LeagueExploiter"

# modified from AlphaStar pseudo-code

import numpy as np

from ..ma.pfsp import pfsp 

from .luxagent import LuxAgent
from ...luxenv.luxenv.agent import Agent

debug = False


class Player(object):
    @property
    def learner(self):
        return self._learner

    def set_learner(self, learner):
        self._learner = learner


    def get_match(self):
        pass

    def ready_to_checkpoint(self):
        return False

    def _create_checkpoint(self):
        # AlphaStar： return Historical(self, self.payoff)
        return Historical(self.agent.policy, self._payoff)

    @property
    def payoff(self):
        return self._payoff

    def checkpoint(self):
        raise NotImplementedError

    


class Historical(Player):

    def __init__(self, model, payoff):
        self.agent = LuxAgent(policy=model)
        self._payoff = payoff
        self._parent = Agent()
        self.name = "Historical"

    @property
    def parent(self):
        return self._parent

    def get_match(self):
        raise ValueError("Historical players should not request matches")

    def ready_to_checkpoint(self):
        return False


class MainPlayer(Player):

    def __init__(self, policy, payoff):
        self.agent = LuxAgent(policy)
        # actually the _payoff maintains all the players and their fight results
        # maybe this should be the league, making it more reasonable
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = "MainPlayer"

    def _pfsp_branch(self):
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="squared")), True

    def _selfplay_branch(self, opponent):
        # Play self-play match
        if self._payoff[self, opponent] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance")), True

    def _verification_branch(self, opponent):
        # Check exploitation
        exploiters = set([
            player for player in self._payoff.players
            if isinstance(player, MainExploiter)
        ])
        # Q: What is the player.parent?
        # A: This is only the property of Historical
        exp_historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent in exploiters
        ]
        win_rates = self._payoff[self, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            return np.random.choice(
                exp_historical, p=pfsp(win_rates, weighting="squared")), True

        # Check forgetting
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]

        def remove_monotonic_suffix(win_rates, players):
            if not win_rates:
                return win_rates, players

            for i in range(len(win_rates) - 1, 0, -1):
                if win_rates[i - 1] < win_rates[i]:
                    return win_rates[:i + 1], players[:i + 1]

            return np.array([]), []

        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            return np.random.choice(
                historical, p=pfsp(win_rates, weighting="squared")), True

        return None

    def get_match(self):
        coin_toss = np.random.random()

        # Make sure you can beat the League
        if coin_toss < 0.5:
            return self._pfsp_branch()

        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < 2e6:
            return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 

    def checkpoint(self):
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class MainExploiter(Player):

    def __init__(self, model, payoff):
        self.agent = LuxAgent(model=model)
        self._initial_weights = model.state_dict()
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = "MainExploiter"

    def get_match(self):
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        if self._payoff[self, opponent] > 0.1:
            return opponent, True

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]

        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance")), True

    def checkpoint(self):
        self.agent.model.load_state_dict(self._initial_weights)
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < 2e9:
            return False

        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        win_rates = self._payoff[self, main_agents]
        return win_rates.min() > 0.7 or steps_passed > 4e9


class LeagueExploiter(Player):

    def __init__(self, model, payoff):
        self.agent = LuxAgent(model=model)
        self._initial_weights = model.state_dict()
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = "LeagueExploiter"

    def get_match(self):
        historical = [
            player for player in self._payoff.players 
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(

            historical, p=pfsp(win_rates, weighting="linear_capped")), True

    def checkpoint(self):
        if np.random.random() < 0.25:
            self.agent.model.load_state_dict(self._initial_weights)
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()

    def ready_to_checkpoint(self):
        steps_passed = self._agent.get_steps() - self._checkpoint_step
        if steps_passed < 2e9:
            return False
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > 4e9
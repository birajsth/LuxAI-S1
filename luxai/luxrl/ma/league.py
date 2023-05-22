" The code for the league"

# modified from AlphaStar pseudo-code

from .payoff import Payoff

from .player import MainPlayer as MP
from .player import MainExploiter as ME
from .player import LeagueExploiter as LE


debug = False


class League(object):

    def __init__(self,
                 initial_model,
                 main_players=1,
                 main_exploiters=1,
                 league_exploiters=2):
        self._payoff = Payoff()
        self._learning_players = []

        for _ in range(main_players):
            main_player = MP(initial_model, self._payoff)
            self._learning_players.append(main_player)

            # add Historcal (snapshot) player
            self._payoff.add_player(main_player.checkpoint())

        for _ in range(main_exploiters):
            self._learning_players.append(
                ME(initial_model, self._payoff))

        for _ in range(league_exploiters):
            self._learning_players.append(
                LE(initial_model, self._payoff))

        # add MP, ME, LE player
        for player in self._learning_players:
            self._payoff.add_player(player)

        self._learning_players_num = len(self._learning_players)

    def update(self, p1, p2, result):
        return self._payoff.update(p1, p2, result)

    def get_learning_player(self, idx):
        return self._learning_players[idx]

    def add_player(self, player):
        self._payoff.add_player(player)

    def get_learning_players_num(self):
        return self._learning_players_num

    def get_players_num(self):
        return self._payoff.get_players_num()
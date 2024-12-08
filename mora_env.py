from mora_params import *

class Mora:
    def __init__(self):
        self.num_players = NUM_PLAYERS
        self.num_rounds = NUM_ROUNDS
        self.players = []
        self.games_history = []

    def add_player(self, player):        
        if player.name in [player.name for player in self.players]:
            raise Exception("Игрок уже добавлен")
        
        if len(self.players) < self.num_players:
            self.players.append(player)
        else:
            raise Exception(f"Слишком много игроков, максимум {self.num_players}")
    
    def kick_player(self, player):        
        if player in self.players:
            self.players.remove(player)
        else:
            raise Exception("Игрок не найден")
        
    def kick_all(self):
        self.players = []
        
    def _play_round(self):
        round_data = {player.name: {'finger': None, 'amount': None, 'score': 0} for player in self.players}
        fingers_sum = 0
        for player in self.players:
            action = player.get_action(self.game_history)
            round_data[player.name]['finger'] = action['finger']
            round_data[player.name]['amount'] = action['amount']
            fingers_sum += action['finger']

        win_counter = 0
        for player in self.players:
            if round_data[player.name]['amount'] == fingers_sum:
                round_data[player.name]['score'] = 1
                win_counter += 1
                if win_counter > 1:
                    for player in self.players:
                        round_data[player.name]['score'] = 0
                    break
            else:
                round_data[player.name]['score'] = 0

        self.game_history.append(round_data)
        return round_data

    def _play_game(self):
        self.game_history = []
        if len(self.players) < self.num_players:
            raise Exception("Не выполнены условия начала игры")
        
        for _ in range(self.num_rounds):
            self._play_round()

        for player in self.players:
            player.set_buffer(self.game_history)

        self.games_history.append(self.game_history)
        return self.game_history

    def play_games(self, num_games):
        for _ in range(num_games):
            self._play_game()



if __name__ == "__main__":
    from mora_player import MoraPlayer, MoraOpponent
    game = Mora()

    p1 = MoraPlayer('p1', None, 1.0)
    p2 = MoraPlayer('p2', None, 1.0)
    p3 = MoraPlayer('p3', None, 1.0)
    p4 = MoraPlayer('p4', None, 1.0)

    game.add_player(p1)
    game.add_player(p2)
    game.add_player(p3)
    game.add_player(p4)

    game.play_games(2)




                


        

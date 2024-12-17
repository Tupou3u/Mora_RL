from mora_params import *

class Mora:
    def __init__(self):
        self.num_players = NUM_PLAYERS
        self.num_rounds = NUM_ROUNDS
        self.players = []
        self._reset()

    def _reset(self):
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

        # v1: За угадывание, каждый получает по очку
        # for player in self.players:
        #     if round_data[player.name]['amount'] == fingers_sum:
        #         round_data[player.name]['score'] = 1
        #     else:
        #         round_data[player.name]['score'] = 0

        # v2: Если угадал один игрок, то получает очко. Если угадали несколько - никто не получает
        # win_counter = 0
        # for player in self.players:
        #     if round_data[player.name]['amount'] == fingers_sum:
        #         round_data[player.name]['score'] = 1
        #         win_counter += 1
        #         if win_counter > 1:
        #             for player in self.players:
        #                 round_data[player.name]['score'] = 0
        #             break
        #     else:
        #         round_data[player.name]['score'] = 0

        # v3: Очки деляться на количество победителей
        win_counter = 0
        for player in self.players:
            if round_data[player.name]['amount'] == fingers_sum:
                round_data[player.name]['score'] = 1
                win_counter += 1
            else:
                round_data[player.name]['score'] = 0

        if win_counter > 0:
            for player in self.players:
                round_data[player.name]['score'] /= win_counter

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
        self._reset()
        for _ in range(num_games):
            self._play_game()

    def get_history(self):
        return self.games_history



if __name__ == "__main__":
    from mora_player import *
    from utils import *

    game = Mora()

    p1 = MoraRandom('p1')
    p2 = MoraRandom('p2')
    p3 = MoraRandom('p3')
    p4 = MoraRandom('p4')

    game.add_player(p1)
    game.add_player(p2)
    game.add_player(p3)
    game.add_player(p4)

    game.play_games(100_000)
    print(get_score(game.games_history))





                


        


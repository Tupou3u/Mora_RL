def get_score(games_history):
    score = {}
    for game_history in games_history:
        for round in game_history:
            for player in round.keys():
                if player not in score.keys():
                    score[player] = round[player]['score']
                else:
                    score[player] += round[player]['score']
    return score

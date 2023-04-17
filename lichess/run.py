# -*- coding: utf-8 -*-
"""
Start lichess playing loop.
"""

from time import sleep

import requests

from arek_chess.common.constants import Print
from arek_chess.main.controller import Controller

token = "lip_o2ZT2O8UxRCT4HRUFLNm"
headers = {"Authorization": f"Bearer {token}"}


def accept_first_challenge():
    challenges_json = requests.get("https://lichess.org/api/challenge", headers=headers).json()
    challenges = challenges_json["in"]
    print(challenges)
    if challenges:
        challenge_id = challenges[0]["id"]
        challenge_request = requests.post(f"https://lichess.org/api/challenge/{challenge_id}/accept", headers=headers)
        if challenge_request.status_code != 200:
            print("failed accepting challenge")
            print(challenge_request.status_code, challenge_request.json())
        else:
            print("accepted a challenge")


def play(game_data):
    my_turn = game_data["isMyTurn"]

    if my_turn:
        game_id = game_data["gameId"]
        fen = game_data["fen"]

        controller = Controller(Print.CANDIDATES, search_limit=15)
        controller.boot_up(fen)
        my_move = controller.search().uci()

        print(my_move)

        move_request = requests.post(f"https://lichess.org/api/bot/game/{game_id}/move/{my_move}", headers=headers)
        if move_request.status_code != 200:
            print("failed making a move")
            print(move_request.status_code, move_request.json())
        else:
            print(f"playing {my_move}")

        controller.tear_down()


while True:
    try:
        print("looping over?")
        going_games_json = requests.get("https://lichess.org/api/account/playing", headers=headers).json()
        going_games = going_games_json["nowPlaying"]
        if not going_games:
            accept_first_challenge()
            print("waiting for challenges...")
        else:
            print("got a game to play...")
            play(going_games[0])

        sleep(1)
    except:
        import traceback
        traceback.print_exc()
        break

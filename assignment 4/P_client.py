# client to Mancala server. Lab4, DVA340, MDU.
# For students: you only need to fill out function decide_move(boardIn, playerTurnIn)
# it currently selects a random available move.
# To test your client: start Mancala_server.pyc, then your program and one bot in that order (server first, then clients)
import copy
import socket
import numpy as np
import time
from multiprocessing.pool import ThreadPool
import os
from datetime import date

# My current implementation of minimax maximizes the score for player 1 and minimizes it for player 2.
# But since we can be player 1 or 2, we look to minimize the score for player 2 and maximize it for player 1.
# So it works as intended but the implementation is a bit confusing.

def decide_move(boardIn, playerTurnIn):
    #CHANGE THIS FILE TO CODE INTELLIGENCE IN YOUR CLIENT.
    # PLAYERMOVE IS '1'..'6'
    # BOARDIN CONSISTS OF 14 INTS. BOARDIN[0-5] ARE P1 HOLES, BOARDIN[6] IS P1 STORE
    # BOARDIN[7-12] ARE P2 HOLES, BOARDIN[13] IS P2 STORE
    best_move, _ = minimax(boardIn, playerTurnIn, depth=0, max_depth=3)

    if best_move is None:
        print("No valid move found, defaulting to first available move.")
        best_move = get_legal_moves(boardIn, playerTurnIn)[0]

    playerMove = str(best_move)

    return playerMove, "miniMaxMove"


def minimax(board, playerTurnIn, depth, max_depth):
    legal_moves = get_legal_moves(board, playerTurnIn)  # list of legal moves

    # Base case: if we reached max depth or there are no legal moves left
    if depth == max_depth or not legal_moves:
        return None, evaluate_board(board)

    # For player 1 we want to maximize the score,
    # for player 2 we want to minimize it
    best_score = float('-inf') if playerTurnIn == 1 else float('inf')
    best_move = None

    # For each legal move, simulate the move and evaluate recursively.
    for move in legal_moves:
        new_board, next_turn = simulate_move(board, playerTurnIn, move)
        # Recursively call minimax for the new board and the next player's turn,
        # incrementing the depth.
        _, score = minimax(new_board, next_turn, depth + 1, max_depth)

        # For Player 1, we update if the new score is higher (maximization).
        # For Player 2, we update if the new score is lower (minimization).
        if (playerTurnIn == 1 and score > best_score) or (playerTurnIn == 2 and score < best_score):
            best_score = score
            best_move = move

    return best_move, best_score


def get_legal_moves(board, playerTurnIn):
    if playerTurnIn == 1:
        return [i + 1 for i in range(6) if board[i] > 0]  # P1 side moves
    else:
        return [i - 6 for i in range(7, 13) if board[i] > 0]  # P2 side moves


def evaluate_board(board):
    """Evaluates the board state from a fixed perspective.

    Returns a positive score if Player 1 is ahead and a negative score if Player 2 is ahead.
    """
    return board[6] - board[13]  # P1 store minus P2 store


def simulate_move(board, playerTurn, playerMove):
    """Simulates a move without modifying the real board.
    Calls `play()` on a board copy and returns the new board + next turn.
    """
    new_board = copy.deepcopy(board)  # Creates a copy of the board
    # TODO: Concept of utility function report, improvements
    result = play(playerTurn, playerMove, new_board)  # Simulate move using play()

    if result is None:
        return None, None

    new_board, next_turn = result  # Unpack result

    return new_board, next_turn  # Return simulated board state


# Dont touch: -----------------------------------
def play(playerTurn: int, playerMove: int, boardGame):
    #playerTurn ar 1 eller 2
    #playerMove ar 1..6
    #boardGame ar en 1x14 vektor
    if not correctPlay(playerMove, boardGame, playerTurn):
        print("Illegal move! break")
        return

    # Determine starting index based on playerTurn and playerMove
    idx = playerMove - 1 + (playerTurn - 1) * 7  #-1 for p1, +6 for p2
    # grab stones from hole
    numStones: int = boardGame[idx]
    boardGame[idx] = 0
    hand: int = numStones
    while hand > 0:
        #idx next hole
        idx = (idx + 1) % 14
        # Skip opponent's store
        if idx == 13 - 7 * (playerTurn - 1):  #13 for p1, 6 for p2
            continue
        # add stone in hole,
        boardGame[idx] += 1
        hand -= 1

    # end in store? get another turn. otherwise other players turn
    nextTurn = 3 - playerTurn
    if idx == 6 + 7 * (playerTurn - 1):
        nextTurn = playerTurn

    #end on own empty hole? score stone and opposite hole
    if boardGame[idx] == 1 and idx in range((playerTurn - 1) * 7, 6 + (playerTurn - 1) * 7):
        boardGame[idx] -= 1  #score stone in last hole
        boardGame[6 + (playerTurn - 1) * 7] += 1  #and remove it from the hole
        boardGame[6 + (playerTurn - 1) * 7] += boardGame[12 - idx]  #also score stones from opposite hole
        boardGame[12 - idx] = 0  #and remove them from the hole
    return (boardGame, nextTurn)


def correctPlay(playerMove: int, board, playerTurn):
    correct = 0
    if playerMove in range(1, 7) and board[playerMove - 1 + (playerTurn - 1) * 7] > 0:
        correct = 1
    return correct


def countScorePlayer1(boardGame):
    (p1s, p2s) = countPoints(boardGame)
    return int(p1s - p2s)


def countPoints(boardGame):
    return (boardGame[6], boardGame[13])


def receive(socket):
    msg = ''.encode()

    try:
        data = socket.recv(1024)
        msg += data
    except:
        pass

    return msg.decode()


def send(socket, msg):
    socket.sendall(msg.encode())


# LET THE MAIN BEGIN


startTime = date(2020, 11, 9)
playerName = 'Lukas_Bonkowski'
host = '127.0.0.1'
port = 30000
s = socket.socket()
pool = ThreadPool(processes=1)
gameEnd = False
MAX_RESPONSE_TIME = 20
print('The player: ' + playerName + ' starts!')
s.connect((host, port))
print('The player: ' + playerName + ' connected!')
while not gameEnd:
    asyncRetult = pool.apply_async(receive, (s,))
    startTime = time.time()
    currentTime = 0
    received = 0
    data = []
    while received == 0 and currentTime < MAX_RESPONSE_TIME:
        time.sleep(0.01)
        if asyncRetult.ready():
            data = asyncRetult.get()
            received = 1
        currentTime = time.time() - startTime
    if received == 0:
        print('No response in ' + str(MAX_RESPONSE_TIME) + ' sec')
        gameEnd = 1
    if data == 'N':
        send(s, playerName)
    if data == 'E':
        gameEnd = 1
    if len(data) > 1:
        board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        playerTurn = int(data[0])
        i = 0
        j = 1
        while i <= 13:
            board[i] = int(data[j]) * 10 + int(data[j + 1])
            i += 1
            j += 2
        (move, botname) = decide_move(board, playerTurn)
        #    print('sending ', move)
        send(s, move)

#wait = input('Press ENTER to close the program.')

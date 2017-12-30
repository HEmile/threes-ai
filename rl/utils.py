import numpy as np
from threes import do_move

def to_nn_input(board, tileset):
    # assume 16 inputs.
    one_hot = 16
    board = board.flatten()
    sz_board = len(board)
    in_nn = np.zeros((sz_board + 1, one_hot))
    for i in range(sz_board):
        in_nn[i, board[i]] = 1
    for tile in tileset:
        in_nn[-1, tile] = 1
    return in_nn.flatten()

def valid_moves(board):
    valid = []
    for i in range(4):
        if do_move(board, i):
            valid.append(i)
    return valid

def best_move(activations, valid):
    activations = np.array(activations)[valid]
    move = np.argmax(activations)
    return valid[move]
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random

__author__ = 'Robert Xiao <nneonneo@gmail.com>'


def to_val(x):
    x = np.asarray(x)
    return np.where(x < 3, x, 3*np.power(2., (x-3.)))


def to_score(x):
    x = np.asarray(x)
    return np.where(x < 3, 0, np.power(3., (x-2)))


def find_fold(line):
    ''' find the position where the line folds, assuming it folds towards the left. '''
    for i in range(3):
        if line[i] == 0:
            return i
        elif (line[i], line[i+1]) in ((1, 2), (2, 1)):
            return i
        elif line[i] == line[i+1] and line[i] >= 3:
            return i
    return -1


def do_fold(line, pos):
    if line[pos] == 0:
        line[pos] = line[pos+1]
    elif line[pos] < 3:
        line[pos] = 3
    else:
        line[pos] += 1
    line[pos+1:-1] = line[pos+2:]
    line[-1] = 0
    return line


def get_lines(m, dir):
    if dir == 0: # up
        return [m[:, i] for i in range(4)]
    elif dir == 1: # down
        return [m[::-1, i] for i in range(4)]
    elif dir == 2: # left
        return [m[i, :] for i in range(4)]
    elif dir == 3: # right
        return [m[i, ::-1] for i in range(4)]


# Creates a deck of 4 3s, 4 2s and 4 1s. These are emptied randomly
def make_deck():
    import random
    deck = [1]*4 + [2]*4 + [3]*4
    random.shuffle(deck)
    return deck

def do_move(m, move):
    lines = get_lines(m, move)
    folds = [find_fold(l) for l in lines]
    changelines = []
    for i in range(4):
        if folds[i] >= 0:
            do_fold(lines[i], folds[i])
            changelines.append(lines[i])
    return changelines

def to_nn_input(board, tileset):
    # assume 16 inputs.
    one_hot = 16
    board = board.flatten()
    sz_board = len(board)
    in_nn = np.zeros(sz_board + one_hot)
    for i in range(sz_board):
        in_nn[i] = board[i]
    for tile in tileset:
        in_nn[sz_board + tile] = 1
    return in_nn.flatten()

class ThreesEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self):
        self.deck = None
        self.board = None
        self.tileset = None
        self.valid = []
        # Observation space is 16 places on the board with any of the 16 tiles, ([16]*16)
        # then for the next tile it's a sequence of binaries ([2]*16) as sometimes multiple options are possible
        self.observation_space = spaces.MultiDiscrete([16]*16 + [2]*16)
        self.action_space = spaces.Discrete(4)

    def _prepare_move(self):
        lineset = [get_lines(self.board, i) for i in range(4)]
        foldset = [[find_fold(l) for l in lineset[i]] for i in range(4)]
        valid = [i for i in range(4) if any(f >= 0 for f in foldset[i])]

        # TODO: Update random tile generation to account for new pick-three implementation
        # If the highest tile is >= 48 (?), then with 1/24 probability create a tileset of 3 options.
        maxval = self.board.max()
        if maxval >= 7 and random.random() < 1 / 24.:
            if maxval <= 9:
                tileset = range(4, maxval - 2)
            else:
                top = random.choice(range(6, maxval - 2))
                tileset = range(top - 2, top + 1)
        # Otherwise, pop from the deck
        else:
            if not self.deck:
                self.deck = make_deck()
            tileset = [self.deck.pop()]
        return valid, tileset

    def step(self, action):
        # Make sure the action is valid. If it's not, exit and return a reward of 0
        if action not in self.valid:
            print(action, self.valid, self.board, self.tileset)
            if not self.valid:
                return to_nn_input(self.board, self.tileset), 0.0, True, {}
            return to_nn_input(self.board, self.tileset), 0.0, False, {}
        prev_score = to_score(self.board).sum()
        # Apply the action
        changelines = do_move(self.board, action)
        # Add a random number from the tileset to the right
        random.choice(changelines)[-1] = random.choice(self.tileset)

        new_score = to_score(self.board).sum()

        self.valid, self.tileset = self._prepare_move()
        if not self.valid:
            return to_nn_input(self.board, self.tileset), new_score - prev_score, True, {}
        return to_nn_input(self.board, self.tileset), new_score - prev_score, False, {}

    def reset(self):
        # Returns the initial numbers that have to be distributed over the board
        self.deck = make_deck()

        # Create an empty board and distribute the numbers in the deck randomly
        pos = random.sample(range(16), 9)
        self.board = np.zeros((16,), dtype=int)
        self.board[pos] = self.deck[:len(pos)]
        self.board = self.board.reshape((4, 4))

        # What does this do? I think it empties the deck.
        self.deck = self.deck[len(pos):]

        self.valid, self.tileset = self._prepare_move()

        return to_nn_input(self.board, self.tileset)

    def render(self, mode='human'):
        print(to_val(self.board))

    def close(self):
        pass


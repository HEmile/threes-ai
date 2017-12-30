import sys
import numpy as np


def play_with_search(verbose=True):
    from threes import play_game, to_val, to_score
    from collections import Counter

    import random
    import time
    seed = hash(str(time.time()))
    print("seed=%d" % seed)
    random.seed(seed)

    initial_deck = Counter([1,2,3]*4)
    deck = None
    game = play_game()
    move = None

    moveno = 0
    while True:
        m, tileset, valid = game.send(move)
        if verbose:
            print(to_val(m))
        if deck is None:
            deck = initial_deck.copy() - Counter(m.flatten())

        if not valid:
            break

        if verbose:
            print('next tile:', list(to_val(tileset)))

        move = find_best_move(m, deck, tileset)
        moveno += 1
        if verbose:
            print("Move %d: %s" % (moveno, ['up', 'down', 'left', 'right'][move]))
        else:
            sys.stdout.write('UDLR'[move])
            sys.stdout.flush()

        if tileset[0] <= 3:
            deck[tileset[0]] -= 1
        if all(deck[i] == 0 for i in (1,2,3)):
            deck = initial_deck.copy()

    print()
    print("Game over. Your score is", to_score(m).sum())
    return to_score(m).sum()
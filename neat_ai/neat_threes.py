import neat
import random
import rl.parameters as params
import sys
import rl.utils as utils
import numpy as np
import pickle
import os
import neat_ai.hyperneat as hn

USE_HYPERNEAT = True

class NEATThreesEvaluator:

    def __init__(self, len_input=params.LEN_INPUT, len_output=params.LEN_OUTPUT, verbose=False):
        self.len_inputs = len_input
        self.len_outputs = len_output
        self.verbose = verbose
        if USE_HYPERNEAT:
            substrate = hn.Substrate()
            def d3_layer(layer, len_x, len_y, len_z):
                lin_input = []
                for x in np.linspace(-1, 1, len_x):
                    for y in np.linspace(-1, 1, len_y):
                        for z in np.linspace(-1, 1, len_z):
                            lin_input.append((layer, x, y, z))
                return lin_input
            lin_input = d3_layer(0, 4, 4, 16)
            for z in np.linspace(-1, 1, 16):
                lin_input.append((0, 0, 0, z))
            substrate.add_nodes(lin_input, layer_id='input')

            lin_hidden_1 = d3_layer(1, 6, 6, 3)
            substrate.add_nodes(lin_hidden_1, layer_id='hidden1')
            lin_hidden_2 = d3_layer(2, 6, 6, 3)
            substrate.add_nodes(lin_hidden_2, layer_id='hidden2')
            lin_output = [(3, 0, -1, 0), (3, 0, 1, 0), (3, -1, 0, 0), (3, 1, 0, 0)]
            substrate.add_nodes(lin_output, layer_id='output')
            substrate.add_connections('input', 'hidden1')
            substrate.add_connections('hidden1', 'hidden2')
            substrate.add_connections('hidden2', 'output')
            self.hn_conv = hn.HyperNEATDeveloper(substrate, feedforward=True)

    def eval_genomes(self, population, config):
        from threes import play_game, to_val, to_score, do_move
        # For every network in the population, evaluate the fitness function
        for genome_id, genome in population:
            genome.track_fitness = {}
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            if USE_HYPERNEAT:
                net = self.hn_conv.convert(net)

            game = play_game()
            move = None

            moveno = 0
            while True:
                m, tileset, valid = game.send(move)
                if self.verbose:
                    print(to_val(m))

                if not valid:
                    break

                if self.verbose:
                    print('next tile:', list(to_val(tileset)))

                nn_input = utils.to_nn_input(m, tileset)
                if USE_HYPERNEAT:
                    activations = net.feed(nn_input)[-4:]
                else:
                    activations = net.activate(nn_input)
                move = utils.best_move(activations, valid)
                moveno += 1
                if self.verbose:
                    print("Move %d: %s" % (moveno, ['up', 'down', 'left', 'right'][move]))
                # else:
                    # sys.stdout.write('UDLR'[move])
                    # sys.stdout.flush()
            fitness = to_score(m).sum()
            print("Game over. Genomes score is", fitness)
            genome.fitness = fitness

def run(config_file, evaluator):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.genome_config.num_inputs = evaluator.len_inputs
    config.genome_config.num_outputs = evaluator.len_outputs

    # Create the population, which is the top-level object for a NEAT run.
    if len(sys.argv) > 1:
        p = neat.Checkpointer.restore_checkpoint(sys.argv[-1])
    else:
        p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(3))

    # Run for up to 300 generations.
    winner = p.run(lambda p, c: evaluator.eval_genomes(p, c), 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner_net, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'config-hyperneat' if USE_HYPERNEAT else 'config-feedforward')
    run(config_path, NEATThreesEvaluator())
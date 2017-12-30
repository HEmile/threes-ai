import neat
import os
from android_assistant import main
import rl.parameters as params
import sys
import rl.utils as utils

# Implements a driver that uses an RNN to drive around.
# The given network needs to have parameters.LEN_INPUTS inputs
# and 12 outputs. The outputs need to be in [0, 1].
def test_checkpoint(checkpoint):
    local_dir = os.path.dirname(__file__)
    p = neat.Checkpointer.restore_checkpoint(checkpoint).population.values()
    config_file = os.path.join(local_dir, 'config-feedforward')
    genome = max(p, key=lambda x: x.fitness if x.fitness else 0)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.genome_config.num_inputs = params.LEN_INPUT
    config.genome_config.num_outputs = params.LEN_OUTPUT
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    def ai_func(board, deck, tileset):
        nn_input = utils.to_nn_input(board, tileset)
        activations = network.activate(nn_input)
        valid = utils.valid_moves(board)
        return utils.best_move(activations, valid)
    main(sys.argv[1:], ai_func)


if __name__ == '__main__':
    test_checkpoint("neat_ai/neat_checkpoints/pop100/neat-checkpoint-164")
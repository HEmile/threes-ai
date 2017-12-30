""" Implements HyperNEAT's conversion
    from genotype to phenotype.
"""

### IMPORTS ###
from itertools import product
from neat.nn import FeedForwardNetwork

# Libs
import numpy as np

# Local
from functools import partial

# Shortcuts
inf = float('inf')

""" Package with some classes to simulate neural nets.
"""

### IMPORTS ###

import sys
np.seterr(over='ignore', divide='raise')

sqrt_two_pi = np.sqrt(np.pi * 2)

### FUNCTIONS ###

# Node functions
def ident(x):
    return x

def bound(x, clip=(-1.0, 1.0)):
    return np.clip(x, *clip)

def gauss(x):
    """ Returns the pdf of a gaussian.
    """
    return np.exp(-x ** 2 / 2.0) / sqrt_two_pi

def sigmoid(x):
    """ Sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid2(x):
    """ Sigmoid function.
    """
    return 1 / (1 + np.exp(-4.9*x))

def abs(x):
    return np.abs(x)

def sin(x):
    return np.sin(x)

def tanh(x):
    return np.tanh(x)

def summed(fn):
    return lambda x: fn(sum(x))

### CONSTANTS ###

SIMPLE_NODE_FUNCS = {
    'sin': np.sin,
    'abs': np.abs,
    'ident': ident,
    'linear': ident,
    'bound': bound,
    'gauss': gauss,
    'sigmoid': sigmoid,
    'sigmoid2': sigmoid2,
    'exp': sigmoid,
    'tanh': tanh,
    None : ident
}

def rbfgauss(x):
    return np.exp(-(x ** 2).sum() / 2.0) / sqrt_two_pi

def rbfwavelet(x):
    return np.exp(-(x ** 2).sum() / ( 2* 0.5**2 )) * np.sin(2 * np.pi * x[0])

COMPLEX_NODE_FUNCS = {
    'rbfgauss': rbfgauss,
    'rbfwavelet': rbfwavelet
}

class NeuralNetwork(object):
    """ A neural network. Can have recursive connections.
    """

    def from_matrix(self, matrix, node_types=['sigmoid']):
        """ Constructs a network from a weight matrix.
        """
        # Initialize net
        self.original_shape = matrix.shape[:matrix.ndim//2]
        # If the connectivity matrix is given as a hypercube, squash it down to 2D
        n_nodes = np.prod(self.original_shape)
        self.cm  = matrix.reshape((n_nodes,n_nodes))
        self.node_types = node_types
        if len(self.node_types) == 1:
            self.node_types *= n_nodes
        self.act = np.zeros(self.cm.shape[0])
        self.optimize()
        return self

    def optimize(self):
        # If all nodes are simple nodes
        if all(fn in SIMPLE_NODE_FUNCS for fn in self.node_types):
            # Simply always sum the node inputs, this is faster
            self.sum_all_node_inputs = True
            self.cm = np.nan_to_num(self.cm)
            # If all nodes are identical types
            if all(fn == self.node_types[0] for fn in self.node_types):
                self.all_nodes_same_function = True
            self.node_types = [SIMPLE_NODE_FUNCS[fn] for fn in self.node_types]
        else:
            nt = []
            for fn in self.node_types:
                if fn in SIMPLE_NODE_FUNCS:
                    # Substitute the function(x) for function(sum(x))
                    nt.append(summed(SIMPLE_NODE_FUNCS[fn]))
                else:
                    nt.append(COMPLEX_NODE_FUNCS[fn])
            self.node_types = nt


    def __init__(self, source=None):
        # Set instance vars
        self.feedforward    = False
        self.sandwich       = False
        self.cm             = None
        self.node_types     = None
        self.original_shape = None
        self.sum_all_node_inputs = False
        self.all_nodes_same_function = False

        if source is not None:
            try:
                self.from_matrix(*source.get_network_data())
                if hasattr(source, 'feedforward') and source.feedforward:
                    self.make_feedforward()
            except AttributeError:
                raise Exception("Cannot convert from %s to %s" % (source.__class__, self.__class__))

    def make_sandwich(self):
        """ Turns the network into a sandwich network,
            a network with no hidden nodes and 2 layers.
        """
        self.sandwich = True
        self.cm = np.hstack((self.cm, np.zeros(self.cm.shape)))
        self.cm = np.vstack((np.zeros(self.cm.shape), self.cm))
        self.act = np.zeros(self.cm.shape[0])
        return self

    def num_nodes(self):
        return self.cm.shape[0]

    def make_feedforward(self):
        """ Zeros out all recursive connections.
        """
        if np.triu(np.nan_to_num(self.cm)).any():
            raise Exception("Connection Matrix does not describe feedforward network. \n %s" % np.sign(self.cm))
        self.feedforward = True
        self.cm[np.triu_indices(self.cm.shape[0])] = 0

    def flush(self):
        """ Reset activation values. """
        self.act = np.zeros(self.cm.shape[0])

    def feed(self, input_activation, add_bias=True, propagate=1):
        """ Feed an input to the network, returns the entire
            activation state, you need to extract the output nodes
            manually.

            :param add_bias: Add a bias input automatically, before other inputs.
        """
        if propagate != 1 and (self.feedforward or self.sandwich):
            raise Exception("Feedforward and sandwich network have a fixed number of propagation steps.")
        act = self.act
        node_types = self.node_types
        cm = self.cm
        input_shape = input_activation.shape

        if add_bias:
            input_activation = np.hstack((1.0, input_activation))

        if input_activation.size >= act.size:
            raise Exception("More input values (%s) than nodes (%s)." % (input_activation.shape, act.shape))

        input_size = min(act.size - 1, input_activation.size)
        node_count = act.size

        # Feed forward nets reset the activation, and activate as many
        # times as there are nodes
        if self.feedforward:
            act = np.zeros(cm.shape[0])
            propagate = len(node_types)
        # Sandwich networks only need to activate a single time
        if self.sandwich:
            propagate = 1
        for _ in range(propagate):
            act[:input_size] = input_activation.flat[:input_size]

            if self.sum_all_node_inputs:
                nodeinputs = np.dot(self.cm, act)
            else:
                nodeinputs = self.cm * act
                nodeinputs = [ni[-np.isnan(ni)] for ni in nodeinputs]

            if self.all_nodes_same_function:
                act = node_types[0](nodeinputs)
            else:
                for i in range(len(node_types)):
                    act[i] = node_types[i](nodeinputs[i])
        self.act = act

        # Reshape the output to 2D if it was 2D
        if self.sandwich:
            return act[act.size//2:].reshape(input_shape)
        else:
            return act.reshape(self.original_shape)

    def cm_string(self):
        print ("Connectivity matrix: %s" % (self.cm.shape,))
        cp = self.cm.copy()
        s = np.empty(cp.shape, dtype='a1')
        s[cp == 0] = ' '
        s[cp > 0] = '+'
        s[cp < 0] = '-'
        return '\n'.join([''.join(l) + '|' for l in s])

    def __str__(self):
        return 'Neuralnet with %d nodes.' % (self.act.shape[0])

class Substrate(object):
    """ Represents a substrate, that is a configuration
        of nodes without connection weights. Connectivity
        is defined, and connection weights are later
        determined by HyperNEAT or another method.
    """

    def __init__(self, nodes_or_shape=None):
        """ Constructor, pass either a shape (as in numpy.array.shape)
            or a list of node positions. Dimensionality is determined from
            the length of the shape, or from the length of the node position
            vectors.
        """
        self.nodes = None
        self.is_input = []
        self.num_nodes = 0
        self.layers = {}
        self.connections = []
        self.connection_ids = []
        self.linkexpression_ids = []
        # If a shape is passed, create a mesh grid of nodes.
        if nodes_or_shape is not None:
            self.add_nodes(nodes_or_shape, 'a')
            self.add_connections('a', 'a')

    def add_nodes(self, nodes_or_shape, layer_id='a', is_input=False):
        """ Add the given nodes (list) or shape (tuple)
            and assign the given id/name.
        """
        if type(nodes_or_shape) == list:
            newnodes = np.array(nodes_or_shape)

        elif type(nodes_or_shape) == tuple:
            # Create coordinate grids
            newnodes = np.mgrid[[slice(-1, 1, s * 1j) for s in nodes_or_shape]]
            # Move coordinates to last dimension
            newnodes = newnodes.transpose(range(1, len(nodes_or_shape) + 1) + [0])
            # Reshape to a N x nD list.
            newnodes = newnodes.reshape(-1, len(nodes_or_shape))
            self.dimensions = len(nodes_or_shape)

        elif type(nodes_or_shape) == np.ndarray:
            pass  # all is good

        else:
            raise Exception("nodes_or_shape must be a list of nodes or a shape tuple.")

        if self.nodes is None:
            self.dimensions = newnodes.shape[1]
            self.nodes = np.zeros((0, self.dimensions))

        # keep a dictionary with the set of node IDs for each layer_id
        ids = self.layers.get(layer_id, set())
        ids |= set(range(len(self.nodes), len(self.nodes) + len(newnodes)))
        self.layers[layer_id] = ids

        # append the new nodes
        self.nodes = np.vstack((self.nodes, newnodes))
        self.num_nodes += len(newnodes)

    def add_connections(self, from_layer='a', to_layer='a', connection_id=-1, max_length=inf, link_expression_id=None):
        """ Connect all nodes in the from_layer to all nodes in the to_layer.
            A maximum connection length can be given to limit the number of connections,
            manhattan distance is used.
            HyperNEAT uses the connection_id to determine which CPPN output node
            to use for the weight.
        """
        conns = product(self.layers[from_layer], self.layers[to_layer])
        conns = list(filter(lambda t: np.all(np.abs(self.nodes[t[0]] - self.nodes[t[1]]) <= max_length), conns))
        self.connections.extend(conns)
        self.connection_ids.extend([connection_id] * len(conns))
        self.linkexpression_ids.extend([link_expression_id] * len(conns))

    def get_connection_list(self, add_deltas):
        """ Builds the connection list only once.
            Storing this is a bit of time/memory tradeoff.
        """
        if not hasattr(self, '_connection_list'):
            self._connection_list = []
            for ((i, j), conn_id, expr_id) in zip(self.connections, self.connection_ids, self.linkexpression_ids):
                fr = self.nodes[i]
                to = self.nodes[j]
                if add_deltas:
                    conn = np.hstack((fr, to, to - fr))
                else:
                    conn = np.hstack((fr, to))
                self._connection_list.append(((i, j), conn, conn_id, expr_id))

        return self._connection_list


class HyperNEATDeveloper(object):
    """ HyperNEAT developer object."""

    def __init__(self, substrate,
                 sandwich=False,
                 feedforward=False,
                 add_deltas=False,
                 weight_range=0.1,
                 min_weight=0.4,
                 activation_steps=10,
                 node_type='tanh'):
        """ Constructor

            :param substrate:      A substrate object
            :param weight_range:   (min, max) of substrate weights
            :param min_weight:     The minimum CPPN output value that will lead to an expressed connection.
            :param sandwich:       Whether to turn the output net into a sandwich network.
            :param feedforward:       Whether to turn the output net into a feedforward network.
            :param node_type:      What node type to assign to the output nodes.
        """
        self.substrate = substrate
        self.sandwich = sandwich
        self.feedforward = feedforward
        self.add_deltas = add_deltas
        self.weight_range = weight_range
        self.min_weight = min_weight
        self.node_type = node_type

    def convert(self, network):
        """ Performs conversion.

            :param network: Any object that is convertible to a :class:`~peas.networks.NeuralNetwork`.
        """

        # Since Stanley mentions to "fully activate" the CPPN,
        # I assume this means it's a feedforward net, since otherwise
        # there is no clear definition of "full activation".
        # In an FF network, activating each node once leads to a stable condition.

        # Check if the network has enough inputs.
        required_inputs = 2 * self.substrate.dimensions
        if self.add_deltas:
            required_inputs += self.substrate.dimensions
        if len(network.input_nodes) < required_inputs:
            raise Exception("Network does not have enough inputs. Has %d, needs %d" %
                            (len(network.input_nodes), required_inputs))

        # Initialize connectivity matrix
        cm = np.zeros((self.substrate.num_nodes, self.substrate.num_nodes))

        for (i, j), coords, conn_id, expr_id in self.substrate.get_connection_list(self.add_deltas):
            weight = network.activate(coords)[0]
            cm[j, i] = weight

        # Remove connections with a low absolute weight
        cm[np.abs(cm) < self.min_weight] = 0
        # Rescale the CM
        cm -= (np.sign(cm) * self.min_weight)
        cm *= (self.weight_range / (1 - self.min_weight))

        # Clip highest weights
        cm = np.clip(cm, -self.weight_range, self.weight_range)
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])

        if self.sandwich:
            net.make_sandwich()

        if self.feedforward:
            net.make_feedforward()

        if not np.all(np.isfinite(net.cm)):
            raise Exception("Network contains NaN/inf weights.")

        return net


def create_converter(substr, sandwich=False, add_deltas=False, node_type='tanh'):
    developer = HyperNEATDeveloper(substr, sandwich=sandwich, add_deltas=add_deltas, node_type=node_type)
    return developer.convert
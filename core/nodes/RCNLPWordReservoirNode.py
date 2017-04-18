import Oger
import mdp.utils
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import cudamat as cm
except:
    pass


class RCNLPWordReservoirNode(Oger.nodes.LeakyReservoirNode):
    """Reservoir node with leaky integrator neurons to handle distributed representation of words.
    """

    def __init__(self, word_sparsity=1.0, *args, **kwargs):
        """Initializes and constructs a random reservoir with leaky-integrator neurons.
           Parameters are:
                - input_dim: input dimensionality
                - output_dim: output_dimensionality, i.e. reservoir size
                - nonlin_func: string representing the non-linearity to be applied, default: 'tanh'
                - bias_scaling: scaling of the bias, a constant input to each neuron, default: 0 (no bias)
                - input_scaling: scaling of the input weight matrix, default: 1
                - spectral_radius: scaling of the reservoir weight matrix, default value: 0.9
                - leak_rate: if 1 it is a standard neuron, lower values give slower dynamics

           Weight matrices are either generated randomly or passed at construction time.  If w, w_in or w_bias are not given in the constructor, they are created randomly:
                - input matrix : input_scaling * uniform weights in [-1, 1]
                - bias matrix :  bias_scaling * uniform weights in [-1, 1]
                - reservoir matrix: gaussian weights rescaled to the desired spectral radius

           If w, w_in or w_bias were given as a numpy array or a function, these will be used as initialization instead.
        """
        super(RCNLPWordReservoirNode, self).__init__(*args, **kwargs)

        # Number if neurons linked to each inputs
        if self.input_dim >= self.output_dim:
            nb_neurons = 1
        else:
            nb_neurons = int(self.output_dim / self.input_dim * word_sparsity)
        # end if
        print("Number of neurons per inputs : " + str(nb_neurons))

        # Permutations
        permuts = np.random.choice(self.output_dim, nb_neurons * self.input_dim, replace=False)
        permuts.shape = (self.input_dim, nb_neurons)

        # Init
        self.w_in = mdp.numx.zeros((self.output_dim, self.input_dim))

        # For each inputs
        n_neur = 0
        for inp in permuts:
            self.w_in[inp, n_neur] = mdp.numx.random.randint(0, 2, (nb_neurons)) * 2 - 1
            n_neur += 1
        # end for
        self.w_in *= self.input_scaling
    # end __init__

# end RCNLPWordReservoirNode
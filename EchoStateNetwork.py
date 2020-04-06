import numpy as np
from numpy.core._multiarray_umath import dot, square, tanh
# from sklearn.metrics import mean_squared_error
from math import sqrt

# TODO: added matplotlib
import matplotlib.pyplot as plt


###########
# CLASSES #
###########

class EchoStateNetwork:
    def __init__(self, input, res, outputsize):
        self.input = input
        self.res = res
        self.outputsize = outputsize

        # TODO: changed weights initialization to half standard normal and
        #       removed bias

        self.inputweights = np.random.randn(input, res) * 0.25
        self.reservoirweights = np.random.randn(res, res) * 0.25
        self.outputweights = np.random.randn(res, outputsize) * 0.25
        # self.outputweights = np.random.randn(res+1, output) * 0.25




        # TODO: changed output initialization and input size of activities
        self.output = np.zeros((1, outputsize))
        self.activ = np.zeros((outputsize, res))
        # self.activ = np.zeros((1, res+1))

    def forwardPass(self, reservoir_input):

        # TODO: changed forward pass computation

        # Compute the input that is fed into the reservoir
        external_input = reservoir_input * self.inputweights

        # Get the recurrent input to the reservoir
        recurrent_input = self.activ[0, :self.res]


        # Compute the reservoir activity based on the reservoir input and the
        # recurrent reservoir activity (from the previous time step)
        res_act = np.tanh(
                external_input + np.matmul(recurrent_input, self.reservoirweights)
        )
        
        # bias
        # one = np.array([[1]])
        # res_act = np.concatenate((res_act, one), axis=1)

        # Update the ESN's reservoir activity
        self.activ = res_act


        # Compute the output of the ESN
        network_output = np.matmul(res_act, self.outputweights)

        # Update the ESN's output
        self.output = network_output


    def reset(self):
        # TODO: changed the dimensionality of the activity
        self.activ = np.zeros((self.outputsize, self.res))

    def oscillator(self):
        self.forwardPass(self.output)

    def teacherForcing(self, target):
        self.output = target
    
    def train(self, seq, washout, training, test):

        self.reset

        # Washout
        for i in range(washout):

            # TODO: changed order of teacher forcing and oscillator call

            self.oscillator()
            self.teacherForcing(seq[i])

        # TODO: renamed a to activities and removed b  # to training_sequence

        activities = np.zeros((training, self.res))
        # activities = np.zeros((training, self.res+1))

        training_sequence = np.zeros((training, self.outputsize))


        # TODO: renamed c to net_out
        net_out = np.zeros((training, self.outputsize))

        # Training
        for i in range(washout, washout + training):
            self.oscillator()

            # TODO: since output and activity dimensionalities have been
            #       changed (see __init__ and forwardPass), these line were
            #       adapted accordingly

            # net_out[i-washout] = self.output

            net_out[i - washout, 0] = self.output

            activities[i - washout] = self.activ

            self.teacherForcing(seq[i])
            training_sequence[i-washout] = seq[i]

        # TODO: changed rms
        rms = my_rmse(net_out=net_out[:, 0],
                      target=seq[washout:training + washout])

        print('RMSE1 = ' + str(rms))

        # Perform pseudo matrix inversion of the recorded reservoir activities
        # to calculate output weights according to
        # W_out * activities = net_out
        #              W_out = net_out * pinv(activities)

        inv_activities = np.linalg.pinv(activities)


        # TODO: changed output weight calculation slightly
        # self.outputweights = np.dot(np.linalg.pinv(activities), training_sequence)
        # Compute the output weights

        self.outputweights = np.matmul(inv_activities, training_sequence)
        

        # Test
        for i in range(washout+test, washout + training+test):


            self.oscillator()

            # TODO: again, output size was adapted
            # net_out[i-washout-200] = self.output
            net_out[i-washout-test, 0] = self.output

            # TODO: actually, testing should work without teacher forcing...
            self.teacherForcing(seq[i])

        # TODO: changed rms
        rms = my_rmse(net_out=net_out[:, 0],
                      target=seq[washout + test:training + washout + test])

        print('RMSE2 = ' + str( rms ))

        # self.outputweights = dot( dot(np.ravel(b),a), np.linalg.inv( dot(a,a_T) + \
        #                     reg*np.eye(200) ) )

        # self.outputweights = np.dot(np.dot(np.linalg.inv(np.dot(a.T, a) + reg * np.eye(self.res+1)), a.T), b)

        self.reset()

        # Washout
        for i in range(washout):

            # TODO: again changed order of teacher forcing and oscillator call

            self.oscillator()
            self.teacherForcing(seq[i])

        # TODO: as above, renamed c to net_out
        net_out = np.zeros((test, self.outputsize))


        # test
        for i in range(washout, washout + test):
            self.oscillator()
            # TODO: again net_output dimension was changed
            # net_out[i-washout] = self.output
            net_out[i - washout, 0] = self.output

            # TODO: actually, testing should work without teacher forcing...
            # self.teacherForcing(seq[i])

        # Briefly visualize performance
        plt.plot(range(test), seq[washout: washout + test], label="Target")
        plt.plot(range(test), net_out[:, 0], linestyle="dashed", color="red",
                 label="Network output")
        plt.legend()
        plt.show()


        # TODO: changed rms
        rms = my_rmse(net_out=net_out[:, 0],
                      target=seq[washout:test + washout])

        print('RMSE = ' + str(rms))


#############
# FUNCTIONS #
#############

def my_rmse(net_out, target):
    """
    This function calculates the root mean squared error for a given network
    output and target
    """

    # Get the length of the sequence
    seq_len = len(net_out)

    # Root mean square error calculation
    rmse = np.sqrt((1 / seq_len) * np.sum(np.square(net_out - target)))

    return rmse


##########
# SCRIPT #
##########

seq = np.array
seq = np.loadtxt("sequence.txt")




fs = 700 # sample rate 
f = 20 # the frequency of the signal

x = np.arange(fs) # the points on the x axis for plotting
# compute the value (amplitude) of the sin wave at the for each sample
y = np.sin(2*np.pi*f * (x/fs)) 


sin = np.sin(y)


esn = EchoStateNetwork(1, 10, 1)
esn.train(sin, 100, 400, 200)

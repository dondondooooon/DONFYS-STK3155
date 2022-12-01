import autograd.numpy as np
from autograd import elementwise_grad

# Set rng seed
np.random.seed(2022)

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            xtest,
            ytest,
            n_hidden_neurons=20,
            n_hidden_layers=1,
            batch_size=5,
            eta=0.1,
            lmbd=0.0,
            cost="mse",
            activation="sigmoid",
            score='mse',
            output_activation=None):

        # test data
        self.xtest = xtest
        self.testtarget = ytest

        # data set
        self.X_data_full = X_data 
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.target = Y_data 
        self.init_target = self.target 

        # network parameters
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = n_hidden_layers
        self.Ltot = self.n_hidden_layers + 2 
        if self.target.ndim == 1:
            self.n_categories = 1
        else:
            self.n_categories = self.target.shape[1]
        self.n_output_neurons = self.n_categories
        self.w_scaled = [1,1,1]
        
        # SGD parameters
        self.batch_size = batch_size
        self.minibatch_size = self.n_inputs // self.batch_size
        self.init_eta = eta
        self.t0 = 1.0
        self.t1 = self.t0 / self.init_eta
        self.lmbd = lmbd
        self.tol = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999
        self.delta = 1e-5

        # dictionaries
        Cost_func = {"mse": self.Cost_MSE, "ce": self.Cross_Entropy}
        Activation_func = {"sigmoid": self.Sigmoid, "relu": self.RELU,\
            "leaky": self.Leaky_RELU, "softmax": self.Soft_Max}
        Score_func = {"mse": self.Score_MSE, "r2": self.R2,\
            "prob": self.Prob_Score, "accuracy": self.Acc_Score}

        # create hidden layers, weights, and biases
        self.init_layers_and_error()
        self.init_biases_and_weights()

        # initialize important mathematical functions
        self.cost = Cost_func[cost] 
        self.der_cost = elementwise_grad(self.cost)
        self.activs = [Activation_func[activation] for i in range(self.Ltot)]
        self.der_act = elementwise_grad(self.activs[0])
        if output_activation != None:
            self.activs[-1] = Activation_func(output_activation)
        self.output_der_act = elementwise_grad(self.activs[-1])
        
        # initialize scaled weights if any
        if activation == "relu" or activation == "leaky":
            ri, ci = np.shape(self.weights[0])
            rh, ch = np.shape(self.weights[1])
            ro, co = np.shape(self.weights[-1])
            self.w_scaled = [ri * ci, rh * ch, ro * co]

        # initialize score params
        self.score_name = score
        self.score = Score_func[self.score_name]
        self.score_shape = 1
        if score == "accuracy":
            self.score_shape = self.n_categories

    '''
    Class Functions
    '''
    def init_layers_and_error(self): 
        # z_l: weighted sum / unactivated values of all nodes 
        self.z_l = [np.zeros((self.n_inputs,self.n_hidden_neurons))\
            for i in range(self.n_hidden_layers)]
        # insert input & output layer
        self.z_l.insert(0,self.X_data_full.copy())
        self.z_l.append(np.zeros(np.shape(self.target)))
        # a_l: activated values of all nodes in z 
        self.a_l = self.z_l.copy()
        # d_l: error for a given layer (delta)
        self.d_l = self.z_l.copy()
        self.d_l[0] = np.nan # does not include input layer

    def init_biases_and_weights(self): 
        # weights between hidden layers
        self.weights = [np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)/\
            (self.w_scaled[1]) for i in range(self.n_hidden_layers-1)]
        # weight between input layer and first hidden layer
        self.weights.insert(0,np.random.randn(self.n_features,self.n_hidden_neurons)/\
            (self.w_scaled[0]))
        # weight between last hidden layer and output layer
        self.weights.append(np.random.randn(self.n_hidden_neurons,self.n_categories)/\
            (self.w_scaled[2]))
        
        # bias between input layer and 1st hidden layers & between hidden layers
        self.bias = [np.zeros(self.n_hidden_neurons)+0.01\
            for i in range(self.n_hidden_layers)]
        # bias between last hidden layer and output layer 
        self.bias.append(np.zeros(self.n_categories)+0.01)

    def init_moments(self):  # initialize moments for ADAM
        self.first_moment_weights, self.first_moment_bias = [],[]
        for i in range(self.Ltot-1):
            self.first_moment_weights.append(np.zeros(np.shape(self.weights[i]),float))
            self.first_moment_bias.append(np.zeros(np.shape(self.bias[i]),float))
        self.second_moment_weights = self.first_moment_weights.copy()
        self.second_moment_bias = self.first_moment_bias.copy()

    def feed_forward(self): # feed-forward step
        for i in range(0,self.Ltot-1):
            # weighted sum of inputs to the hidden layer
            new_z = np.matmul(self.a_l[i],self.weights[i]) + self.bias[i]
            self.z_l[i+1] = new_z # update z_l
            self.a_l[i+1] = self.activs[i](new_z) # activation in z_l

    def backpropagation(self): # back-prop step
        # error in the output layer
        self.d_l[-1] = self.output_der_act(self.z_l[-1]) * self.der_cost(self.a_l[-1])
        # back propagate error for other layers
        for i in reversed(range(1,self.Ltot-1)):
            self.d_l[i] = np.matmul(self.d_l[i+1],self.weights[i].T) * self.der_act(self.z_l[i])
        # check for vanish / exploding gradients 
        self.grad_prob = np.linalg.norm(self.d_l[-1] * self.eta)
        if self.grad_prob > self.tol and np.isfinite(self.grad_prob):
            # update parameters
            for i in range(0,self.Ltot-1):
                self.weights_grad = np.matmul(self.d_l[i+1].T,self.a_l[i]).T
                if self.lmbd > 0.0: # regularization term gradients
                    self.weights_grad += self.lmbd * self.weights[i]
                self.bias_grad = np.mean(self.d_l[i+1], axis=0)
                # computing moments
                self.first_moment_weights[i] = self.b1 * self.first_moment_weights[i] +\
                    (1-self.b1) * self.weights_grad
                self.second_moment_weights[i] = self.b2 * self.second_moment_weights[i] +\
                    (1-self.b2) * self.weights_grad * self.weights_grad
                # self.first_moment_bias[i] = self.b1 * self.first_moment_bias[i] +\
                #     (1-self.b1) * self.bias_grad
                # self.second_moment_bias[i] = self.b2 * self.second_moment_bias[i] +\
                #     (1-self.b2) * self.bias_grad
                first_term_w = self.first_moment_weights[i] / (1.0-self.b1**self.iter) 
                second_term_w = self.second_moment_weights[i] / (1.0-self.b2**self.iter)
                # first_term_b = self.first_moment_bias[i] / (1.0-self.b1**self.iter) 
                # second_term_b = self.second_moment_bias[i] / (1.0-self.b2**self.iter) # i get nans here when i sqrt
                # print(second_term_b)
                # print(np.sqrt(second_term_b))
                # update weights and biases
                self.weights[i] -= self.eta * first_term_w / (np.sqrt(second_term_w) + self.delta)
                self.bias[i] -= self.eta * self.bias_grad # self.eta * first_term_b / (np.sqrt(second_term_b) + self.delta)

    def predict(self, X):
        self.a_l[0] = X
        self.feed_forward()
        return self.activs[-1](self.z_l[-1])

    def SGD_train(self,epochs):
        epoch = 0
        self.iter = 0
        self.grad_prob = 1.0
        self.escore = np.zeros((epochs+1,self.score_shape))

        self.testscore = self.escore.copy()
        self.testscore[0] = self.score(self.xtest,self.testtarget)

        self.escore[0] = self.score(self.X_data_full,self.init_target)
        self.epoch_evo = [self.escore.copy() for e in range(epochs+1)]
        self.epoch_evo[0] = self.epoch_print(epoch, self.escore[0])

        while (epoch < epochs and self.grad_prob > self.tol and np.isfinite(self.grad_prob)):
            # Initialize first and second moments
            self.init_moments()
            self.iter = self.iter + 1
            for i in range(self.minibatch_size):
                # set learning rate
                self.eta_schedule(epoch,i)
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(self.n_inputs, size=self.batch_size, replace=False)
                # minibatch training data
                self.a_l[0] = self.X_data_full[chosen_datapoints]
                self.target = self.init_target[chosen_datapoints]
                self.feed_forward()
                self.backpropagation()
            epoch += 1
            self.escore[epoch] = self.score(self.X_data_full,self.init_target)
            self.epoch_evo[epoch] = self.epoch_print(epoch, self.escore[epoch])
            
            self.testscore[epoch] = self.score(self.xtest,self.testtarget)

    '''
    Activation Functions
    '''
    def Sigmoid(self, val):
        vexp = np.exp(-val)
        return 1.0/(1.0 + vexp)
    
    def RELU(self, val):
        return np.where(val > 0, val, 0)

    def Leaky_RELU(self, val):
        return np.where(val > 0, val, 0.01 * val)

    def Soft_Max(self, val):
        vexp = np.exp(val)
        return vexp / np.sum(vexp, axis=1, keepdims=True)

    '''
    Cost Functions
    '''
    def Cost_MSE(self, ytilde):
        return (ytilde - self.target) ** 2

    def Cross_Entropy(self, ytilde):
        return -(self.target * np.log(ytilde) +\
            (1 - self.target) * np.log(1 - ytilde))

    '''
    Score Functions
    '''
    def Score_MSE(self, X, target):
        predict = self.predict(X)
        return np.mean( (predict.ravel() - target.ravel()) ** 2)
    
    def R2(self, X, target):
        ypredict = self.predict(X)
        return 1 - np.sum((target - ypredict) ** 2)\
            / np.sum((target - np.mean(target, axis=0)) ** 2)
    
    def Prob_Score(self, X, target):
        self.a_l[0] = X
        self.feedforward()
        predict = self.soft_max_(self.z_l[-1])
        guess = np.argmax(predict, axis=1)
        target = np.argmax(target, axis=1)
        return np.sum(guess == target) / len(target)
    
    def Acc_Score(self, X, target):
        predict = self.predict(X)
        hits = np.sum(np.around(predict) == target, axis=0)
        total = target.shape[0]
        return hits/total

    '''
    Learning Rate Scheduler
    '''
    def eta_schedule(self, epoch, i_batch):
        t = epoch * self.minibatch_size * i_batch
        self.eta = self.t0 / (t + self.t1)

    '''
    Miscellaneous
    '''
    def epoch_print(self, epoch, escore):
        return (f"Epoch: {epoch}; {self.score_name.upper()}_Score = {escore}")
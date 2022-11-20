import numpy as np
from autograd import elementwise_grad

# Set rng seed
np.random.seed(2022)

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=20,
            n_hidden_layers=1,
            batch_size=5,
            eta=0.1,
            lmbd=0.0,
            gamma=0.0,
            cost="mse",
            activation="sigmoid",
            score='mse',
            output_activation=None):

        self.X_data_full = X_data # shape: N x features
        self.target = Y_data # shape: N x 1
        self.init_target = self.target
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]

        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = n_hidden_layers
        self.Ltot = self.n_hidden_layers + 2 
        if self.target.ndim == 1:
            self.n_categories = 1
        else:
            self.n_categories = self.target.shape[1]
        self.n_output_neurons = self.n_categories
        
        self.batch_size = batch_size
        self.minibatch_size = self.n_inputs // self.batch_size
        self.init_eta = eta
        self.lmbd = lmbd
        self.gamma = gamma
        self.tol = 1e-8

        # Dictionaries
        Cost_func = {"mse": self.Cost_MSE, "ce": self.Cross_Entropy}
        Activation_func = {"sigmoid": self.Sigmoid, "softmax": self.Soft_Max}
        Score_func = {"mse": self.Score_MSE, "r2": self.R2,\
            "prob": self.Prob_Score, "accuracy": self.Acc_Score}

        # Create Hidden Layers, Weights, and Biases
        self.init_layers_and_error()
        self.create_biases_and_weights()
        self.init_memory_and_ADAM_momentum()

        # Initialize important mathematical functions
        self.cost = Cost_func[cost]
        self.activs = [Activation_func[activation] for i in range(self.Ltot)]
        self.der_cost = elementwise_grad(self.cost)
        self.der_act = elementwise_grad(self.activs[0])
        if output_activation != None:
            self.activs[-1] = Activation_func(output_activation)
        self.output_der_act = elementwise_grad(self.activs[-1])
        self.score_name = score
        self.score = Score_func[self.score_name]
        self.score_shape = 1
        if score == "accuracy":
            self.score_shape = self.n_categories




    def init_layers_and_error(self): 
        # z_l: weighted sum / unactivated values of all nodes 
        self.z_l = [np.zeros((self.n_inputs,self.n_hidden_neurons))\
            for i in range(self.hidden_layers)]
        # insert input & output layer
        self.z_l.insert(0,self.X_data_full.copy())
        self.z_l.append(np.zeros(np.shape(self.target)))
        # a_l: activated values of all nodes in z 
        self.a_l = self.z_l.copy()
        # d_l: error for a given layer (delta)
        self.d_l = self.z_l.copy()
        self.d_l[0] = np.nan # does not include input layer

    def init_biases_and_weights(self):
        '''
        might have to fix indexing by putting nan on first index in the list
        '''
        # weights between hidden layers
        self.weights = [np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)\\
            for i in range(self.n_hidden_layers-1)]
        # weight between input layer and first hidden layer
        self.weights.insert(0,np.random.randn(self.n_features,self.n_hidden_neurons))
        # weight between last hidden layer and output layer
        self.weights.append(np.random.randn(self.n_hidden_neurons,self.n_categories))
        
        # bias between input layer and 1st hidden layers & between hidden layers
        self.bias = [np.zeros(self.n_hidden_neurons)+0.01\\
            for i in range(self.n_hidden_layers)]
        # bias between last hidden layer and output layer 
        self.bias.append(np.zeros(self.n_categories)+0.01)

    def init_memory_and_ADAM_moment(self): 
        # previous step for momentum/memory  
        self.prev_weights, self.prev_bias = [],[]
        # initialize first and second moment for ADAM optimizer
        self.m_weights, self.v_weights = [], []
        self.m_bias, self.v_bias = [], []
        for i in range(self.Ltot):
            self.prev_weights.append(np.zeros(np.shape(self.weights[i])))
            self.prev_bias.append(np.zeros(np.shape(self.bias[i])))
            self.m_weights.append(np.zeros(np.shape(self.weights[i])))
            self.v_weights.append(np.zeros(np.shape(self.weights[i])))
            self.m_bias.append(np.zeros(np.shape(self.bias[i])))
            self.v_bias.append(np.zeros(np.shape(self.bias[i])))

    def feed_forward(self): # feed-forward step
        for i in range(0,self.Ltot-1):
            # weighted sum of inputs to the hidden layer
            new_z = np.matmul(self.a_l[i],self.weights[i]) + self.bias[i]
            self.z_l[i] = new_z # update z_l
            self.a_l[i] = self.activs[i](new_z) # activation in z_l

    def backpropagation(self): # back-prop step
        # error in the output layer
        self.d_l[-1] = self.output_der_act(self.z_l[-1]) * self.der_cost(self.a_l[-1])
        # back propagate error for other layers
        for i in reversed(range(1,self.Ltot-1)):
            self.d_l[i] = np.matmul(self.d_l[i+1][:,np.newaxis],self.weights[i].T) * self.der_act(self.z_l[i])
        
    def param_update(self):
        self.backpropagation()
        # convergence check 
        self.grad_conv = np.linalg.norm(self.d_l[-1] * self.eta)
        if self.grad_conv > self.tol and np.isfinite(self.grad_conv):
            # update parameters
            for i in range(0,self.Ltot-1):
                self.memo_weights = self.gamma * self.prev_weights[i]
                self.weights_grad = self.eta * np.matmul(self.d_l[i+1].T,self.a_l[i]).T
                # regularization term gradients
                if self.lmbd > 0.0:
                    self.weights_grad += self.eta * self.lmbd * self.weights[i]

                self.memo_bias = self.gamma * self.prev_bias[i]
                self.bias_grad = self.eta * np.mean(self.d_l[i+1], axis=0)

                # Compute Moments
                self.m0 = beta1 * self.m0 + (1 - beta1) * self.weights_grad

                # update weights and biases
                self.weights[i] -= self.memo_weights + self.weights_grad
                self.bias[i] -= self.memo_bias + self.bias_grad

    def predict(self, X):
        self.a_l[0] = X
        self.feed_forward()
        return self.activs[-1](self.z_l(-1))

    def SGD_train(self,epochs):
        epoch = 0
        itera = 0
        # Exponential Decay rates for the Moment Estimates
        self.beta1 = 0.9
        self.beta2 = 0.999
        # Gradient Conversion Initialization
        self.grad_conv = 1
        # Initialize Learning Rate
        self.eta = self.init_eta
        # Initialize First and Second Moment Vector

        np.zeros()
        m_0, v_0 = 0.0, 0.0 

        self.escore = np.zeros((epochs+1,self.score_shape))
        self.escore[0] = self.score(self.X_data_full,self.init_target)
        self.epoch_print(epoch, self.escore[0])
        while (epoch < epochs and self.grad_conv > self.tol and np.isfinite(self.grad_conv)):
            fst_mom, sec_mom = 0.0, 0.0 # first and second moment
            itera += 1
            for k in range(self.minibatch_size):
            # pick datapoints with replacement
                chosen_datapoints = np.random.choice(self.n_inputs, size=self.batch_size, replace=False)
                # minibatch training data
                self.a_l[0] = self.X_data_full[chosen_datapoints]
                self.t = self.init_target[chosen_datapoints]
                self.feed_forward()
                self.param_update()

    # Eta function / Scheduler
    def eta_schedule(self, epoch):
        self.eta = 

    def epoch_print(self, epoch, escore):
        print(f"Epoch: {epoch}; {self.score_name.upper()} = {escore}")

    '''
    Activation Functions
    '''
    def Sigmoid(self, val):
        return 1.0/(1.0 + np.exp(-val))

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
        return np.mean((predict - target) ** 2)
    
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
import numpy as np

class Neuron:
    def __init__(self, weights_vec, bias):
        self.w = weights_vec
        self.b = bias

    def forward(self, inputs):
        y = 0
        for i in range(len(inputs)):
            y += inputs[i]*self.w[i]
        y += self.b
        return self.__sigmoid(y)
        
    
    def backward(self, error):
        return self.__node_delta(error)


    def __sigmoid(self, total_net_input):
        return 1. / (1 + np.exp(-total_net_input))

    def __node_delta(self, error):
        return -(error[0]-error[1])*error[1]*(1-error[1])

class NeuronLayer:
    def __init__(self, weights_arr, bias_vec):
        self.w = weights_arr
        self.b = bias_vec
        self.neuron = []
        self.out = [] 

        for i in range(len(self.w)):
            self.neuron.append(Neuron(self.w[i],self.b[i]))
            # print(self.w[i],self.b[i])
            # print(i)
    def inspect(self):
        print("  Neurons {}".format(len(self.w)))
        
        for i in range(len(self.w)):
            print("    Neuron: {}".format(i))
            print("      Weight: {}".format(self.neuron[i].w))
            print("      Bias: {}".format(self.neuron[i].b))

    def feed_forward(self,inputs):
        for i in range(len(self.w)):
            y = self.neuron[i].forward(inputs)
            self.out.append(y)
        return self.out 
        


class NeuronNetwork:
    def __init__(self, weights_arrs, bias_arr):
        self.w = weights_arrs
        self.b = bias_arr
        self.layer = []

        for i in range(len(self.w)):
            self.layer.append(NeuronLayer(self.w[i],self.b[i]))
             
    
    def inspect(self):
        print("Layers: {}".format(len(self.w)))
        for i in range(len(self.w)):
            print("NeuronLayer {}".format(i))
            self.layer[i].inspect()
            # print(self.layer[i].out)
    
    def feed_forward(self, inputs):
        
        y = self.layer[0].feed_forward(inputs)
        y = self.layer[1].feed_forward(y)
        y = self.layer[2].feed_forward(y)
            
        return y


weights_arrs = [[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]], # init weights of the 1st neuron layer
                [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]], # init weights of the 2nd neuron layer
                [[2.1,2.2,2.3,2.4],[2.5,2.6,2.7,2.8]]] # init weights of the 3rd neuron layer

bias_arr = [[0.1,0.2,0.3], # init biases of the 1st neuron layer
            [1.1,1.2,1.3,1.4], # init biases of the 2nd neuron layer
            [2.1,2.2]] 


nn = NeuronNetwork(weights_arrs, bias_arr)
# nn.inspect()
y = nn.feed_forward([1,1,1])
nn.inspect()
print(y)




# h1 = Neuron([0.15,0.20],0.35)
# h2 = Neuron([0.25,0.30],0.35)
# o1 = Neuron([0.40,0.45],0.60)
# o2 = Neuron([0.50,0.55],0.60)

# x = [0.05, 0.1]
# out_h1 = h1.forward(x)
# out_h2 = h2.forward(x)

# out_o1 = o1.forward([out_h1,out_h2])
# out_o2 = o2.forward([out_h1,out_h2])

# # print(out_h1,out_h2)
# # print(out_o1,out_o2)

# loss_1 = ((0.01-out_o1)**2) / 2
# loss_2 = ((0.99-out_o2)**2) / 2
# total_loss = loss_1 + loss_2

# print(o1.backward([0.01,0.7513650695523157])*out_h1)



# print(total_loss)


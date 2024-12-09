import torch as t
class NeuralNetwork(t.nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        # Define layers
        t.set_num_threads(10)
        self.layer1 = t.nn.Linear(8, 16)
        self.layer2 = t.nn.Linear(16, 8)
        self.layer3 = t.nn.Linear(8, 2)

        self.weights = weights
        self.biases = biases

        #Manually set weights and biases
        with t.no_grad():
            self.layer1.weight.data = t.tensor(self.weights[:16*8], dtype=t.float32).view(16, 8)  # [16, 8]
            self.layer1.bias.data = t.tensor(self.biases[:16], dtype=t.float32)  # [16]
            
            self.layer2.weight.data = t.tensor(self.weights[16*8:2*(16*8)], dtype=t.float32).view(8, 16)  # [8, 16]
            self.layer2.bias.data = t.tensor(self.biases[16:24], dtype=t.float32)  # [8]
            
            self.layer3.weight.data = t.tensor(self.weights[2*(16*8):2*(16*8)+2*8], dtype=t.float32).view(2, 8)  # [2, 8]
            self.layer3.bias.data = t.tensor(self.biases[24:26], dtype=t.float32)  # [2]

    def forward(self, x):
        with t.no_grad():
            x = t.tanh(self.layer1(x)) # tanh instead of relu because we need negative values in order to be able to turn instead of just moving straight or backwards
            x = t.tanh(self.layer2(x))
            output = self.layer3(x)
            return output[0].item(),output[1].item()
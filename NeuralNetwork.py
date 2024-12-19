# Change the input to be only the robot's y position
import torch as t
class NeuralNetwork(t.nn.Module):
    def __init__(self, weights, biases):
        super().__init__()
        # Define layers
        t.set_num_threads(10)
        neurons1 = 1
        neurons2 = 10
        neurons3 = 5
        neurons4 = 2
        self.layer1 = t.nn.Linear(neurons1, neurons2)
        self.layer2 = t.nn.Linear(neurons2, neurons3)
        self.layer3 = t.nn.Linear(neurons3, neurons4)

        self.weights = weights
        self.biases = biases

        #Manually set weights and biases
        with t.no_grad():
            self.layer1.weight.data = t.tensor(self.weights[:(neurons1*neurons2)], dtype=t.float32).view(neurons2, neurons1)  # [6, 2]
            self.layer1.bias.data = t.tensor(self.biases[:neurons2], dtype=t.float32)  # [16]
            
            self.layer2.weight.data = t.tensor(self.weights[(neurons1*neurons2):(neurons1*neurons2)+(neurons2*neurons3)], dtype=t.float32).view(neurons3, neurons2)  # [4, 6]
            self.layer2.bias.data = t.tensor(self.biases[neurons2:neurons2+neurons3], dtype=t.float32)  # [8]
            
            self.layer3.weight.data = t.tensor(self.weights[(neurons1*neurons2)+(neurons2*neurons3):(neurons1*neurons2)+(neurons2*neurons3)+(neurons3*neurons4)], dtype=t.float32).view(neurons4, neurons3)  # [2, 4]
            self.layer3.bias.data = t.tensor(self.biases[neurons2+neurons3:neurons2+neurons3+neurons4], dtype=t.float32)  # [2]
    def forward(self, x):
        with t.no_grad():
            x = t.tanh(self.layer1(x))
            #print("Layer 1 output: " + str(x))
            x = t.tanh(self.layer2(x))
            #print("Layer 2 output: " + str(x))
            output = t.tanh(self.layer3(x))
            output = t.flatten(output)
            #print("Layer 3 final output: " + str(output))
            #outputClass = t.argmax(output)
            return output[0].item(), output[1].item()
            # Treated as a classification problem. Turn right, left, drive forward, back, or do nothing
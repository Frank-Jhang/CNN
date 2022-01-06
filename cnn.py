from PIL import Image
import numpy as np

class CNN:
    def __init__(self):
        self.image_size = 32
        self.batch_size = 64
        self.filter_size = 3
        # create a 2d array to initialize filter
        self.filter_1 = np.asarray([[1 for x in range(self.filter_size)] for y in range(self.filter_size)])
        self.filter_2 = np.asarray([[1 for x in range(self.filter_size)] for y in range(self.filter_size)])
        self.data = []
        self.params = {
            'W1': np.random.randn(10, 36) / np.sqrt(10/2),   # FC layer weight initialization
            'W2': np.random.randn(3, 10) / np.sqrt(3/2)
        }    # a dictionary to store parameters

    def read_data(self, type, index):
        # load the image of specific type and index
        if(type == 0):
            image = Image.open('./Data/Data_train/Carambula/Carambula_train_' + str(index) + '.png')
        if(type == 1):
            image = Image.open('./Data/Data_train/Lychee/Lychee_train_' + str(index) + '.png')
        if(type == 2):
            image = Image.open('./Data/Data_train/Pear/Pear_train_' + str(index) + '.png')
        raw_data = np.asarray(image)    # convert image to numpy array
        self.data = np.delete(raw_data, 1, 2) / 255 - 0.5   # delete axis=2 , 2 is same as the index of shape?
        self.data = np.squeeze(self.data, axis=2)

    def one_hot_label(self, type):
        arr = np.zeros(3)
        arr[type] = 1
        return arr
    
    def convolution(self, input, filter):
        h, w = input.shape
        output = np.zeros((h - 2, w - 2))
        for i in range(h - 2):
            for j in range(w - 2):
                temp_region = input[i:(i + 3), j:(j + 3)]   # index i+3 is not including, i.e. i~i+2
                output[i][j] = np.sum(temp_region * filter)
        return output
    
    def maxpool(self, input):
        h, w = input.shape
        output = np.zeros((h//2, w//2)) # delete the decimal point
        for i in range(h//2):
            for j in range(w//2):
                temp_region = input[(2*i):(2*i+2), (2*j):(2*j+2)]
                output[i][j] = np.amax(temp_region)
        return output

    def forward_pass(self, input):
        params = self.params

        params['A0'] = input

        # first convolution
        params['Z1'] = self.convolution(input, self.filter_1)
        params['A1'] = self.relu_2d(params['Z1'])

        # first maxpooling
        params['P1'] = self.maxpool(params['A1'])

        # second convolution
        params['Z2'] = self.convolution(params['P1'], self.filter_2)
        params['A2'] = self.relu_2d(params['Z2'])

        # second maxpooling
        params['P2'] = self.maxpool(params['A2'])

        # flatten (fully connected layer's input)
        params['FC_A0'] = params['P2'].flatten()

        # hidden layer
        params['FC_Z1'] = np.dot(params['W1'], params['FC_A0'])
        params['FC_A1'] = self.relu(params['FC_Z1'])

        # output layer
        params['FC_Z2'] = np.dot(params['W2'], params['FC_A1'])
        params['FC_A2'] = self.softmax(params['FC_Z2'])

        return params['FC_A2']

    def backpropagation(self, initial_grad):
        params = self.params
        change_w = {}   # a dictionary to store weight gradient

        # Calculate W2 update
        change_w['W2'] = np.outer(initial_grad, params['FC_A1'])  # partial_cw = partial_cz * partial_zw
        # propagate gradient (gradient of FC_Z1)
        prop_grad = np.dot(params['W2'].T, initial_grad) * self.relu(params['FC_Z1'], derivative=True)

        # Calculate W1 update
        change_w['W1'] = np.outer(prop_grad, params['FC_A0'])
        # propagate gradient (gradient of FC_A0)
        prop_grad = np.dot(params['W1'].T, prop_grad) * 1   # 1, cuz FC_A0 doesn't have activation function (just flatten the pooling)
        # change the prop_grad into pooling layer size (i.e. a 2d gradient array)
        prop_grad = prop_grad.reshape(params['P2'].shape)   # 6*6
        print(prop_grad)

        return change_w
    
    def relu_2d(self, x, derivative=False):
        row, col = x.shape
        if derivative:
            for i in range(row):
                for j in range(col):
                    if x[i][j] < 0: x[i][j] = 0   # call by reference? call by value?
                    else: x[i][j] = 1
        for i in range(row):
            for j in range(col):
                if x[i][j] < 0: x[i][j] = 0
        return x
    
    def relu(self, x, derivative=False):
        if derivative:
            for i in range(x.size):
                if x[i] < 0: x[i] = 0   # call by reference? call by value?
                else: x[i] = 1
        for i in range(x.size):
            if x[i] < 0: x[i] = 0

        return x
    
    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    # gradient of softmax + cross entropy
    def softmax_CE_grad(self, label, softmax_z):
        grad_list = [(softmax_z[i] * np.sum(label) - label[i]) for i in range(3)]
        grad = np.asarray(grad_list).reshape(3)
        return grad

if __name__ == '__main__':
    cnn = CNN()
    type = 0
    idx = 7
    cnn.read_data(type, idx)
    label = cnn.one_hot_label(type)
    output = cnn.forward_pass(cnn.data)
    initial_grad = cnn.softmax_CE_grad(label, output)
    # print(output)
    # print(initial_grad)
    cnn.backpropagation(initial_grad)
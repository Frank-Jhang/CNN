from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self):
        self.image_size = 32    # 32*32
        self.batch_size = 49
        self.filter_size = 3    # 3*3
        self.image_num = 490    # 490 pictures in total (each fruit)
        self.training_num = int(self.image_num * 0.7)
        self.validation_num = int(self.image_num * 0.3)
        self.testing_num = 166
        self.fc_l_rate = 0.002  # FC layer learning rate
        self.conv_l_rate = 0.002  # convolutional layer learning rate
        self.epochs = 10
        
        # create a 2d array to initialize filter
        self.filter_1 = np.asarray([[np.random.uniform(-1.0,1.0) for x in range(self.filter_size)] for y in range(self.filter_size)])
        # self.filter_2 = np.arange(9).reshape((3,3))
        self.filter_2 = np.asarray([[np.random.uniform(-1.0,1.0) for x in range(self.filter_size)] for y in range(self.filter_size)])
        self.data = []
        self.params = {
            'W1': np.random.randn(10, 36) / np.sqrt(10/2),   # FC layer weight initialization
            'W2': np.random.randn(3, 10) / np.sqrt(3/2)
        }    # a dictionary to store parameters

        self.sum_del_w = {'W1': 0, 'W2': 0}
        self.sum_del_filter_2 = np.asarray([[0.0 for x in range(self.filter_size)] for y in range(self.filter_size)])
        self.sum_del_filter_1 = np.asarray([[0.0 for x in range(self.filter_size)] for y in range(self.filter_size)])
        
        self.training_loss = 0
        self.validation_loss = 0

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
    
    def read_testing_data(self, type, index):
        # load the image of specific type and index
        if(type == 0):
            image = Image.open('./Data/Data_test/Carambula/Carambula_test_' + str(index) + '.png')
        if(type == 1):
            image = Image.open('./Data/Data_test/Lychee/Lychee_test_' + str(index) + '.png')
        if(type == 2):
            image = Image.open('./Data/Data_test/Pear/Pear_test_' + str(index) + '.png')
        raw_data = np.asarray(image)    # convert image to numpy array
        self.data = np.delete(raw_data, 1, 2) / 255 - 0.5   # delete axis=2 , 2 is same as the index of shape?
        self.data = np.squeeze(self.data, axis=2)

    def one_hot_label(self, type):
        arr = np.zeros(3)
        arr[type] = 1
        return arr
    
    def convolution(self, in_1, in_2):
        h1, w1 = in_1.shape
        h2, w2 = in_2.shape
        output = np.zeros((h1-h2+1, w1-w2+1))   # the size after doing convolution
        for i in range(h1-h2+1):
            for j in range(w1-w2+1):
                temp_region = in_1[i:(i + h2), j:(j + w2)]   # index i+3 is not including, i.e. i~i+2
                output[i][j] = np.sum(temp_region * in_2)
        return output

    def full_convolution(self, in_1_, in_2):
        h1_, w1_ = in_1_.shape
        in_1 = np.zeros((h1_+4, w1_+4)) # initialize the padding array
        for i in range(h1_):
            for j in range(w1_):
                in_1[i+2][j+2] = in_1_[i][j]    # padding 0 with 2 dim of each side (cuz filter size is 3)
        # below are same as normal convolution
        h1, w1 = in_1.shape
        h2, w2 = in_2.shape
        output = np.zeros((h1-h2+1, w1-w2+1))   # the size after doing convolution
        for i in range(h1-h2+1):
            for j in range(w1-w2+1):
                temp_region = in_1[i:(i + h2), j:(j + w2)]   # index i+3 is not including, i.e. i~i+2
                output[i][j] = np.sum(temp_region * in_2)
        return output
    
    def maxpool(self, input):
        h, w = input.shape
        output = np.zeros((h//2, w//2)) # delete the decimal point
        for i in range(h//2):
            for j in range(w//2):
                temp_region = input[(2*i):(2*i+2), (2*j):(2*j+2)]
                output[i][j] = np.amax(temp_region)
        return output

    def maxpool_backprop(self, input, prop_grad):
        h, w = input.shape
        grad_arr = np.zeros((h, w))
        for i in range(h//2):
            for j in range(w//2):
                temp_region = input[(2*i):(2*i+2), (2*j):(2*j+2)]
                for k in range(2):  # check which element is the max, and repalce it with prop_grad
                    for m in range(2):
                        if(temp_region[k][m] == np.amax(temp_region)):
                            grad_arr[2*i+k][2*j+m] = prop_grad[i][j]
        return grad_arr

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
        del_w = {}   # a dictionary to store weight gradient

        # Calculate W2 update
        del_w['W2'] = np.outer(initial_grad, params['FC_A1'])  # partial_cw = partial_cz * partial_zw
        # propagate gradient (gradient of FC_Z1)
        prop_grad = np.dot(params['W2'].T, initial_grad) * self.relu(params['FC_Z1'], derivative=True)

        # Calculate W1 update
        del_w['W1'] = np.outer(prop_grad, params['FC_A0'])
        # propagate gradient (gradient of FC_A0)
        prop_grad = np.dot(params['W1'].T, prop_grad) * 1   # 1, cuz FC_A0 doesn't have activation function (just flatten the pooling)
        # change the prop_grad into pooling layer size (i.e. a 2d gradient array)
        prop_grad = prop_grad.reshape(params['P2'].shape)   # 6*6 (gradient of P2)

# -------------------------------------------
        # propagate gradient to the layer before maxpool (gradient of Z2)
        prop_grad = self.maxpool_backprop(params['A2'], prop_grad)  # the size before doing maxpool (by fill in 0 to the right position)

        # Calculate filter 2 update
        del_filter_2 = self.convolution(params['P1'], prop_grad)
        # print('filter 2 gradient:')
        # print(del_filter_2)
        # propagate gradient (gradient of P1)
        prop_grad = self.full_convolution(prop_grad, np.flip(self.filter_2))    # 15*15

# -------------------------------------------
        # propagate gradient to the layer before maxpool (gradient of Z1)
        prop_grad = self.maxpool_backprop(params['A1'], prop_grad)

        # Calculate filter 1 update
        del_filter_1 = self.convolution(params['A0'], prop_grad)
        # print('filter 1 gradient:')
        # print(del_filter_1)
        # propagate gradient (gradient of A0)
        prop_grad = self.full_convolution(prop_grad, np.flip(self.filter_1))    # 32*32 (end of the back propagation)

        return del_w, del_filter_2, del_filter_1
    
    # update the weight
    def update_weight(self, del_w, del_filter_2, del_filter_1):
        # FC layer weight
        for key, value in del_w.items():
            self.params[key] -= self.fc_l_rate * value
        # filter weight
        self.filter_2 -= self.conv_l_rate * del_filter_2
        self.filter_1 -= self.conv_l_rate * del_filter_1
    
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
    
    # cross entropy loss
    def CE_loss(self, output, label):
        loss = 0.0
        for i in range(3):
            loss += label[i] * np.log(output[i])
        return - loss    # need to calculate mean?

    # gradient of softmax + cross entropy
    def softmax_CE_grad(self, label, softmax_z):
        grad_list = [(softmax_z[i] * np.sum(label) - label[i]) for i in range(3)]
        grad = np.asarray(grad_list).reshape(3)
        return grad
    
    def sum_grad(self, del_w, del_filter_2, del_filter_1):
        self.sum_del_w['W1'] += del_w['W1']
        self.sum_del_w['W2'] += del_w['W2']
        self.sum_del_filter_2 += del_filter_2
        self.sum_del_filter_1 += del_filter_1
    
    def avg_grad(self, sum_del_w, sum_del_filter_2, sum_del_filter_1):
        avg_del_w = {'W1': 0, 'W2': 0}
        avg_del_w['W1'] = sum_del_w['W1'] / self.batch_size
        avg_del_w['W2'] = sum_del_w['W2'] / self.batch_size
        avg_del_filter_2 = sum_del_filter_2 / self.batch_size
        avg_del_filter_1 = sum_del_filter_1 / self.batch_size
        return avg_del_w, avg_del_filter_2, avg_del_filter_1
    
    def reset_sum_grad(self):
        self.sum_del_w['W1'] = 0
        self.sum_del_w['W2'] = 0
        self.sum_del_filter_2 = np.asarray([[0.0 for x in range(self.filter_size)] for y in range(self.filter_size)])
        self.sum_del_filter_1 = np.asarray([[0.0 for x in range(self.filter_size)] for y in range(self.filter_size)])

    def training(self):
        print('start training !')
        temp_idx = 0
        plot_train_x = []
        plot_train_y = []
        plot_valid_x = []
        plot_valid_y = []
        for i in range(self.epochs):
            suffle_idx = np.random.permutation(self.image_num)
            training_idx = suffle_idx[0 : self.training_num]
            validation_index = suffle_idx[self.training_num : :]
            for j in range(self.training_num//self.batch_size):
                idx = training_idx[temp_idx : temp_idx+self.batch_size]   # choose a segment of the suffle index
                for k in range(self.batch_size):
                    for m in range(3):  # 3 types
                        idx = k
                        type = m

                        self.read_data(type, idx)
                        label = self.one_hot_label(type)

                        output = self.forward_pass(self.data)
                        # print('forward pass output:')
                        # print(output)

                        loss = self.CE_loss(output, label)
                        self.training_loss += loss

                        initial_grad = self.softmax_CE_grad(label, output)
                        del_w, del_filter_2, del_filter_1 = self.backpropagation(initial_grad)
                        self.sum_grad(del_w, del_filter_2, del_filter_1)

                # end of a batch
                # update weight
                avg_del_w, avg_del_filter_2, avg_del_filter_1 = self.avg_grad(self.sum_del_w, self.sum_del_filter_2, self.sum_del_filter_1)
                # print(avg_del_w['W1'])
                # print(avg_del_w['W2'])
                # print(avg_del_filter_2)
                # print(avg_del_filter_1)
                self.update_weight(avg_del_w, avg_del_filter_2, avg_del_filter_1)
                # reset the sum
                self.reset_sum_grad()
                # move to the next index segment
                temp_idx += self.batch_size

            # end of an epoch
            self.validation(validation_index)
            print('epoch {}: training loss {} , validation loss {}'.format(i+1, round(self.training_loss,2), round(self.validation_loss,2)))
            # print('epoch {}: validation loss {}'.format(i+1, self.validation_loss))
            plot_train_x.append(i+1)
            plot_train_y.append(round(self.training_loss,2))
            plot_valid_x.append(i+1)
            plot_valid_y.append(round(self.validation_loss,2))
            # reset
            self.training_loss = 0
            self.validation_loss = 0

        print('end of training !')
        plots = plt.plot(plot_train_x, plot_train_y, plot_valid_x, plot_valid_y)
        plt.legend(plots, ('training loss', 'validation loss'))
        plt.show()

    def validation(self, index_array):
        for j in range(self.validation_num):
            for m in range(3):  # 3 types
                idx = index_array[j]
                type = m

                self.read_data(type, idx)
                label = self.one_hot_label(type)

                output = self.forward_pass(self.data)

                loss = self.CE_loss(output, label)
                self.validation_loss += loss

    def testing(self):
        print('-')
        print('start testing !')
        num_correct = 0
        for j in range(self.testing_num):
            for m in range(3):  # 3 types
                idx = j
                type = m

                self.read_testing_data(type, idx)
                label = self.one_hot_label(type)

                output = self.forward_pass(self.data)

                if(np.argmax(output) == np.argmax(label)):
                    num_correct += 1

        print('test_accuracy: {}'.format(num_correct / (self.testing_num*3)))
        print('end of testing !')


if __name__ == '__main__':
    cnn = CNN()
    cnn.training()
    cnn.testing()
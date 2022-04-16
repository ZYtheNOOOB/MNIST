import json
import random
import sys

from PIL import Image
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

epsilon = 1e-8


class MLP:
    def __init__(self, hidden_dims, lr, beta1=0.9, beta2=0.999):
        self.hidden = []
        self.trainable = {}
        # hidden layers
        input_dim = 784
        for i, n in enumerate(hidden_dims):
            name = 'l' + str(i)
            layer = FCLayer(input_dim=input_dim, output_dim=n, name=name)
            activation = Tanh()
            self.hidden.append(layer)
            self.hidden.append(activation)
            self.trainable[name] = layer
            input_dim = n
        # softmax layer
        self.softmax = SoftmaxCrossEntropy()
        # optimizer
        self.optimizer = AdamOptimizer(self.trainable, lr, beta1, beta2)

    def __call__(self, input):
        h_out = input
        for h in self.hidden:
            h_out = h(h_out)
        probs = self.softmax.forward_prob(h_out)
        return probs

    def compute_loss(self, value, target):
        loss = self.softmax.compute_loss(value, target)
        grad = self.softmax.bp()
        for i in range(len(self.hidden)-1, -1, -1):
            grad = self.hidden[i].bp(grad)
        return loss

    def update(self):
        for layer in self.trainable.values():
            self.optimizer.update(layer)

    def save(self):
        params = {}
        for name, layer in self.trainable.items():
            params[name] = {'w': layer.W, 'b': layer.b}
        with open('./weights.json', 'w') as f:
            json.dump(params, f, cls=NumpyEncoder, indent=2)

    def zero_grad(self):
        for layer in self.hidden:
            layer.clear()
        self.softmax.clear()


class FCLayer:
    def __init__(self, input_dim, output_dim, name=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.W = np.random.normal(size=(self.input_dim, self.output_dim), loc=0., scale=.01)
        # self.b = np.zeros(shape=(1, self.output_dim))
        # Xavier
        # std = np.sqrt(2 / (input_dim + output_dim))
        # self.W = np.random.normal(size=(self.input_dim, self.output_dim), loc=0., scale=std)
        # self.b = np.random.normal(size=(1, self.output_dim), loc=0., scale=std)
        self.W = np.random.normal(size=(self.input_dim, self.output_dim), loc=0., scale=1.) / np.sqrt(input_dim)
        self.b = np.random.normal(size=(1, self.output_dim), loc=0., scale=1.) / np.sqrt(input_dim)
        self.input = None
        self.output = None
        self.W_grad = None
        self.b_grad = None

        self.name = name

    def __call__(self, input):
        self.input = input
        self.output = np.dot(self.input, self.W) + self.b
        return self.output

    def bp(self, grad):
        self.W_grad = np.dot(np.transpose(self.input), grad)
        self.b_grad = np.sum(grad, axis=0)
        return np.dot(grad, np.transpose(self.W))

    def update(self, lr):
        if self.W_grad is not None and self.b_grad is not None:
            self.W = self.W - lr * self.W_grad
            self.b = self.b - lr * self.b_grad
        else:
            print('no grad available, run backward first.')

    def clear(self):
        self.input = None
        self.output = None
        self.W_grad = None
        self.b_grad = None


class Tanh:
    def __init__(self):
        self.output = None

    def __call__(self, input):
        self.output = np.tanh(input)
        return self.output

    def bp(self, grad):
        tanh_grad = grad * (1 - np.square(self.output))
        return tanh_grad

    def clear(self):
        self.output = None


class SoftmaxCrossEntropy:
    def __init__(self):
        self.input = None
        self.output = None
        self.target = None

    def forward_prob(self, input):
        self.input = input - np.max(input)
        self.output = np.exp(input) / np.exp(input).sum(axis=1)[:, np.newaxis]
        return self.output

    def compute_loss(self, probs, target):
        self.target = target
        return -np.mean(self.target * np.log(probs))

    def bp(self):
        batch_size = self.input.shape[0]
        grad = -(self.target - self.output) / batch_size
        return grad

    def clear(self):
        self.input = None
        self.output = None
        self.target = None


class AdamOptimizer:
    def __init__(self, trainable, lr, beta1, beta2):
        self.step = 1
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.gW_weighted_avg = {}
        self.gb_weighted_avg = {}
        self.gW_squared_weighted_avg = {}
        self.gb_squared_weighted_avg = {}
        for k, v in trainable.items():
            self.gW_weighted_avg[k], self.gW_squared_weighted_avg[k] = np.zeros_like(v.W), np.zeros_like(v.W)
            self.gb_weighted_avg[k], self.gb_squared_weighted_avg[k] = np.zeros_like(v.b), np.zeros_like(v.b)

    def update(self, layer):
        # momentum for weighted avg
        self.gW_weighted_avg[layer.name] = self.beta1 * self.gW_weighted_avg[layer.name] + (
                1 - self.beta1) * layer.W_grad
        self.gb_weighted_avg[layer.name] = self.beta1 * self.gb_weighted_avg[layer.name] + (
                1 - self.beta1) * layer.b_grad
        # momentum for squared weighted avg
        self.gW_squared_weighted_avg[layer.name] = self.beta2 * self.gW_squared_weighted_avg[layer.name] + (
                1 - self.beta2) * np.square(layer.W_grad)
        self.gb_squared_weighted_avg[layer.name] = self.beta2 * self.gb_squared_weighted_avg[layer.name] + (
                1 - self.beta2) * np.square(layer.b_grad)
        # bias correction 1
        self.gW_weighted_avg_corr = self.gW_weighted_avg[layer.name] / (1 - self.beta1 ** self.step)
        self.gb_weighted_avg_corr = self.gb_weighted_avg[layer.name] / (1 - self.beta1 ** self.step)
        # bias correction 2
        self.gW_squared_weighted_avg_corr = self.gW_squared_weighted_avg[layer.name] / (
                1 - self.beta2 ** self.step)
        self.gb_squared_weighted_avg_corr = self.gb_squared_weighted_avg[layer.name] / (
                1 - self.beta2 ** self.step)
        # update params
        layer.W = layer.W - lr * (
                self.gW_weighted_avg_corr / np.sqrt(self.gW_squared_weighted_avg_corr + epsilon))
        layer.b = layer.b - lr * (
                self.gb_weighted_avg_corr / np.sqrt(self.gb_squared_weighted_avg_corr + epsilon))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def augment_data(imgs, labels):
    aug_imgs = []
    aug_labels = []
    for i, img in enumerate(imgs):
        img = img.reshape((28, 28))
        img_s = np.zeros_like(img)

        # original
        img = Image.fromarray(img)
        # scale
        img_s[3:-3, 3:-3] = np.array(img.resize((22, 22)))
        img_s = Image.fromarray(img_s)
        # rotate
        img_r1 = img.rotate(random.randint(-20, 20))
        # img_r2 = img.rotate(-random.randint(20, 30))
        # transform
        x_t1, y_t1 = random.randint(-5, 5), random.randint(-5, 5)
        img_t1 = img_s.transform(img.size, Image.AFFINE, (1, 0, x_t1, 0, 1, y_t1))
        # to numpy
        img = np.array(img).reshape((784,))
        img_s = np.array(img_s).reshape((784,))
        img_r1 = np.array(img_r1).reshape((784,))
        # img_r2 = np.array(img_r2).reshape((784,))
        img_t1 = np.array(img_t1).reshape((784,))

        aug_imgs.extend([img, img_s, img_r1, img_t1])
        aug_labels.extend([[labels[i]] for _ in range(4)])

    # shuffle
    aug_set = np.concatenate((np.array(aug_imgs), np.array(aug_labels)), axis=1)
    np.random.shuffle(aug_set)
    return aug_set[:, :-1], aug_set[:, -1]


def make_one_hot(labels):
    one_hot_labels = np.zeros((len(labels), 10))
    for n in range(len(labels)):
        one_hot_labels[n, int(labels[n])] = 1
    return one_hot_labels


def run_test(model, test_img):
    preds = [np.argmax(model(img)) for img in test_img]
    return preds


def acc_score(pred, label):
    return sum([1 if p == l else 0 for p, l in zip(pred, label)]) / len(label)


if __name__ == '__main__':
    batch_size = 128
    lr = 0.001
    max_epochs = 20

    train_img = np.array(pd.read_csv('train_image.csv', header=None), dtype=np.float32)
    train_label = np.array(pd.read_csv('train_label.csv', header=None), dtype=np.float32)
    test_img = np.array(pd.read_csv('test_image.csv', header=None), dtype=np.float32)
    test_label = np.array(pd.read_csv('test_label.csv', header=None), dtype=np.float32)

    # normalize
    train_img /= 255
    test_img /= 255

    # shuffle training set and label
    train_set = np.concatenate((train_img, train_label), axis=1)
    np.random.shuffle(train_set)

    # split train val
    split = int(len(train_set) * 0.9)
    train_set, val_set = train_set[:split], train_set[split:]
    train_img, train_label = augment_data(train_set[:, :-1], train_set[:, -1])
    val_img, val_label = val_set[:, :-1], val_set[:, -1]

    # create batch
    train_data = []
    for i in range(0, len(train_label), batch_size):
        imgs = train_img[i:i + batch_size]
        labels = train_label[i:i + batch_size]
        one_hot_labels = make_one_hot(labels)
        train_data.append((imgs, one_hot_labels))

    # train
    model = MLP(hidden_dims=[1024, 1024, 10], lr=lr, beta1=0.9, beta2=0.999)
    best_val, best_model = 0, deepcopy(model)
    for e in range(max_epochs):
        random.shuffle(train_data)
        epoch_loss = []
        for inputs, targets in tqdm(train_data):
            outputs = model(inputs)
            loss = model.compute_loss(outputs, targets)
            model.update()
            epoch_loss.append(loss)
            model.optimizer.step += 1
            model.zero_grad()

        # validation
        val_pred = run_test(model=model, test_img=val_img)
        val_acc = acc_score(val_pred, val_label)
        print('Epoch %d, avg loss %f, val acc %f' % (e, np.average(epoch_loss), val_acc))
        # early termination
        if val_acc > best_val:
            best_val = val_acc
            best_model = deepcopy(model)
        elif val_acc < best_val - 0.005:
            print('Early termination at epoch', e)
            break

    pred_label = run_test(model=best_model, test_img=test_img)
    # test_acc = acc_score(pred_label, np.squeeze(test_label))
    # print('Test acc:', test_acc)
    # output
    with open('test_predictions.csv', 'w') as f:
        res = ""
        for n in pred_label:
            res += str(int(n)) + '\n'
        f.write(res[:-1])

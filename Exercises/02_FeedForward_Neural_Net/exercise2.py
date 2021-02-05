"""
Implementation of Scalar Backpropagation.(For Binary Classification)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class FFNMulticlass:
    def __init__(self):
        np.random.seed(0)
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.w7 = np.random.randn()
        self.w8 = np.random.randn()
        self.w9 = np.random.randn()
        self.w10 = np.random.randn()
        self.w11 = np.random.randn()
        self.w12 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0

    def sigmoid(self, y):
        return (1 / (1 + np.exp(-y)))

    def forward_pass(self, x):
        self.x1, self.x2 = self.x

        # First Layer
        self.a1 = (self.w1 * self.x1 + self.w2 * self.x2) + self.b1
        self.h1 = self.sigmoid(self.a1)

        self.a2 = (self.w3 * self.x1 + self.w4 * self.x2) + self.b2
        self.h2 = self.sigmoid(self.a2)

        # 2nd Layer
        self.a3 = (self.w5 * self.h1 + self.w6 * self.h2) + self.b3
        self.a4 = (self.w7 * self.h1 + self.w8 * self.h2) + self.b4
        self.a5 = (self.w9 * self.h1 + self.w10 * self.h2) + self.b5
        self.a6 = (self.w11 * self.h1 + self.w12 * self.h2) + self.b6

        # softmax Function
        sum_exps = np.sum([np.exp(self.a3), np.exp(self.a4), np.exp(self.a5), np.exp(self.a6)])
        self.h3 = np.exp(self.a3) / sum_exps
        self.h4 = np.exp(self.a4) / sum_exps
        self.h5 = np.exp(self.a5) / sum_exps
        self.h6 = np.exp(self.a6) / sum_exps

        return np.array([self.h3, self.h4, self.h5, self.h6])

    def backward_pass(self, x, y):
        self.forward_pass(x)
        self.y1, self.y2, self.y3, self.y4 = y
        





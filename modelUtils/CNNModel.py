import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class CNNModel(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        self.conv1 = kl.Conv2D(128, 3, activation='selu')
        self.mp1 = kl.MaxPool2D((5, 5))
        self.conv2 = kl.Conv2D(128, 3, activation='selu')
        self.mp2 = kl.MaxPool2D((3, 3))
        self.flatten = kl.Flatten()
        self.hidden1 = kl.Dense(128, activation='selu')
        self.hidden2 = kl.Dense(128, activation='selu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, activation='softmax', name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

# Created by yingwen at 2019-03-10

from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
# from malib.core import Serializable
from bilevel_pg.bilevelpg.core import Serializable
from bilevel_pg.bilevelpg.networks.mlp import MLP
import tensorflow as tf

class Policy(Serializable):
    def reset(self):
        """Reset and clean the policy."""
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, *args, **kwargs):
        raise NotImplementedError

    def get_action(self, condition):
        raise NotImplementedError

    def get_actions(self, condition):
        raise NotImplementedError

    def get_action_np(self, condition):
        return self.get_action(condition).numpy()

    def get_actions_np(self, conditions):
        return self.get_actions(conditions).numpy()

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Arguments:
            conditions: Observations to run the diagnostics for.
        Returns:
            diagnostics: OrderedDict of diagnostic information.
        """
        diagnostics = OrderedDict({})
        return diagnostics

    def terminate(self):
        """
        Clean up operation
        """
        pass

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def __getstate__(self):
        state = Serializable.__getstate__(self)
        state['pickled_weights'] = self.get_weights()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state)
        self.set_weights(state['pickled_weights'])


class StochasticPolicy(Policy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 preprocessor=None,
                 rnn_size=10,
                 name='DeterministicPolicy',
                 *args,
                 **kwargs):
        # print(input_shapes)
        self._Serializable__initialize(locals())
        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor
        super(StochasticPolicy, self).__init__(*args, **kwargs)

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        # print(conditions.shape)

        if preprocessor is not None:
            # if preprocessor == 'LSTM':
            #     conditions = tf.keras.layers.LSTM(rnn_size)(tf.reshape(conditions, shape=[tf.shape(conditions)[0],tf.shape(conditions)[1],1]))
            # else:
            conditions = preprocessor(conditions)

        # print(output_shape)
        # print(conditions.shape)

        raw_policies = self._policy_net(
            input_shapes=(conditions.shape[1:],),
            output_size=output_shape[0],
            # output_activation = tf.nn.softmax,
        )(conditions)

        policies = raw_policies
        # actions = raw_actions if self._squash else tf.nn.tanh(raw_actions)
        # actions = np.random.choice()
        # self.actions_model = tf.keras.Model(self.condition_inputs, actions)
        self.diagnostics_model = tf.keras.Model(self.condition_inputs, (raw_policies, policies))
        self.policy_model = tf.keras.Model(self.condition_inputs, policies)

    def log_pis(self, conditions, actions):
        """Compute log probs for given observations and actions."""
        raise NotImplementedError

    def log_pis_np(self, conditions, actions):
        """Compute numpy log probs for given observations and actions."""
        return self.log_pis(conditions, actions).numpy()

    def get_actions(self, conditions):

        raise NotImplementedError
        # return self.actions_model(conditions)

    def get_action(self, condition, extend_dim=True):

        raise NotImplementedError
        # if extend_dim:
        #     condition = condition[None]
        # return self.get_actions(condition)[0]

    def get_action_np(self, condition, extend_dim=True):

        raise NotImplementedError
        # if type(condition) is list:
        #     extend_dim = False
        # if extend_dim:
        #     condition = condition[None]
        # return self.get_actions_np(condition)[0]

    def get_actions_np(self, conditions):

        raise NotImplementedError
        # return self.actions_model.predict(conditions)

    def _policy_net(self, input_shapes, output_size):
        raise NotImplementedError

    def reset(self):
        pass

    def get_weights(self):
        return self.policy_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.policy_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.policy_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(StochasticPolicy, self).non_trainable_weights))

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.
        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            '{}/raw-actions-mean'.format(self._name): np.mean(raw_actions_np),
            '{}/raw-actions-std'.format(self._name): np.std(raw_actions_np),

            '{}/actions-mean'.format(self._name): np.mean(actions_np),
            '{}/actions-std'.format(self._name): np.std(actions_np),
        })


class LatentSpacePolicy(StochasticPolicy):
    def __init__(self, *args, smoothing_coefficient=None, **kwargs):
        super(LatentSpacePolicy, self).__init__(*args, **kwargs)

        assert smoothing_coefficient is None or 0 <= smoothing_coefficient <= 1
        self._smoothing_alpha = smoothing_coefficient or 0
        self._smoothing_beta = (
            np.sqrt(1.0 - np.power(self._smoothing_alpha, 2.0))
            / (1.0 - self._smoothing_alpha))
        self._reset_smoothing_x()
        self._smooth_latents = False

    def _reset_smoothing_x(self):
        self._smoothing_x = np.zeros((1, *self._output_shape))

    def actions_np(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model.predict(conditions)
        elif self._smoothing_alpha == 0:
            return self.actions_model.predict(conditions)
        else:
            alpha, beta = self._smoothing_alpha, self._smoothing_beta
            raw_latents = self.latents_model.predict(conditions)
            self._smoothing_x = (
                    alpha * self._smoothing_x + (1.0 - alpha) * raw_latents)
            latents = beta * self._smoothing_x

            return self.actions_model_for_fixed_latents.predict(
                [*conditions, latents])

    def reset(self):
        self._reset_smoothing_x()



class StochasticMLPPolicy(StochasticPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='relu',
                 output_activation='softmax',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(StochasticMLPPolicy, self).__init__(*args, **kwargs)

    def _policy_net(self, input_shapes, output_size):
        raw_actions = MLP(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation,
            name='{}/StochasticMLPPolicy'.format(self._name)
        )
        return raw_actions

    def get_policy(self, condition, extend_dim = True):
        if extend_dim:
            condition = condition[None]
        return self.get_policies(condition)[0]

    def get_policies(self, conditions):
        return self.policy_model(conditions)

    def get_actions(self, conditions):
        policy = self.get_policies(conditions)
        # print(type(policy))
        actions = []

        for i in range(policy.shape[0]):
            # print(policy[i])
            # norm_prob = policy[i] / policy[i].size()
            actions.append(np.random.choice(2, 1, p=policy[i].numpy()))
        return np.array(actions)

    def get_action_np(self, condition, extend_dim=True):
        # print('in get action np:')
        # print(type(condition), condition)
        if extend_dim:
            # print('has extend dim!!!')
            condition = condition[None]
        return self.get_actions_np(condition)

    def get_actions_np(self, conditions):
        # print(conditions)
        # print('action shape:')
        # print(self._input_shapes)
        # print(tf.transpose(conditions).shape)
        # print(tf.shape(conditions)[0])
        # long = self._input_shapes[0]
        # cur_pol = self.policy_model.predict(tf.reshape(conditions, [-1, long]))
        # conditions = conditions[None]
        # print(np.array(conditions).shape)
        cur_pol = self.policy_model.predict(conditions)
        # print(cur_pol.shape)
        # return np.random.choice(2, 1, p=cur_pol[0])
        actions = []

        for i in range(cur_pol.shape[0]):
            # actions.append(np.argmax(cur_pol[i]))
            actions.append(np.random.choice(cur_pol.shape[1], 1, p=cur_pol[i])[0])
        return np.array(actions)

    def get_policy_np(self, conditions):
        # long = self._input_shapes[0]
        cur_pol = self.policy_model.predict(conditions)
        return cur_pol
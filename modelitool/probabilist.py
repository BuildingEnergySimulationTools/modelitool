from modelitool.simulate import run_batch

import pandas as pd

import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class DCGenerator:
    def __init__(self,
                 simulator,
                 params,
                 sample_size,
                 observable_inputs=None,
                 random_seed=42):

        self.simulator = simulator
        self.params = params
        self.sample_size = sample_size
        self.sample = self.draw_sample(seed=random_seed)
        self.simulation_results = np.array([])
        self.eta_scaler = StandardScaler()
        self.remain_scaler = MinMaxScaler()

        if isinstance(observable_inputs, str) and \
                observable_inputs == "from_simulator":
            if simulator.boundaries is not None:
                self.observable_inputs = simulator.boundaries
            else:
                raise ValueError("Simulator doesn't have a "
                                 "boundary df")

        elif observable_inputs is not None:
            self.observable_inputs = observable_inputs

        else:
            warnings.warn("No observable input were specified")

    def get_dc(self, indicator, scaled=True):
        if self.simulation_results.size == 0:
            raise ValueError("Empty simulation results")

        if self.observable_inputs is not None:
            eta = self.simulation_results[
                      ...,
                      self.simulator.output_list.index(indicator)
                  ].flatten()[:, np.newaxis]

            if isinstance(self.observable_inputs, pd.DataFrame):
                np_observable = self.observable_inputs.to_numpy()
            elif isinstance(self.observable_inputs, np.array):
                np_observable = self.observable_inputs
            else:
                raise ValueError("Please provide pd.DataFrame Object"
                                 "for observable input")
            observable = np.concatenate(
                [np_observable] * self.sample_size)

            parameters = np.concatenate([
                np.concatenate(
                    [sim[:, np.newaxis].T] * self.simulation_results.shape[1])
                for sim in self.sample
            ])

            obs_param = np.concatenate([observable, parameters], axis=1)

            self.eta_scaler.fit(eta)
            self.remain_scaler.fit(obs_param)

            if scaled:
                scaled_eta = self.eta_scaler.transform(eta)
                scaled_obs_param = self.remain_scaler.transform(obs_param)
                return np.concatenate([scaled_eta, scaled_obs_param], axis=1)
            else:
                return np.concatenate([eta, obs_param], axis=1)

    def draw_sample(self, seed=None):
        gather_list = []
        for val in self.params.values():
            if seed is not None:
                rs = RandomState(MT19937(SeedSequence(seed)))
                uniform_pdf = rs.uniform
            else:
                uniform_pdf = np.random.uniform

            gather_list.append(uniform_pdf(low=val[0], high=val[1],
                                           size=self.sample_size))

        return np.concatenate([gather_list], axis=1).T

    def run_simulations(self, verbose_step=10):
        self.simulation_results = run_batch(
            simulator=self.simulator,
            param_name_list=list(self.params.keys()),
            sample=self.sample,
            verbose_step=verbose_step
        )

    def plot_param_distribution(self):

        for i, par in enumerate(self.params.keys()):
            sns.displot(self.sample[:, i])

        plt.show()


class GPTrainer:
    def __init__(self,
                 x_observed,
                 y_observation,
                 kernel=None,
                 prior_dict=None,
                 optimizer=None):

        self.x_observed = x_observed
        self.y_observation = y_observation
        self.training_log_likelihood = np.empty(0, dtype=np.float64)

        if optimizer is None:
            self.optimizer = tf.optimizers.Adam(learning_rate=.01)

        if kernel is None:
            self.kernel = tfk.ExponentiatedQuadratic
        else:
            self.kernel = kernel

        if prior_dict is None:
            self.prior_dict = {
                "amplitude": tfd.LogNormal(loc=0., scale=np.float64(1.)),
                "length_scale": tfd.LogNormal(loc=0., scale=np.float64(1.)),
                "x_observed_noise": tfd.LogNormal(loc=0., scale=np.float64(1.))
            }
        else:
            self.prior_dict = prior_dict

        # We'll put priors on the kernel hyperparameters,
        # and write the joint distribution
        # of the hyperparameters and observed data
        # using tfd.JointDistributionNamed

        # The covariance kernel is instanced here.
        # He will be shared between the prior (which we
        # use for maximum likelihood training) and the posterior
        # (which we use for posterior predictive sampling)

        # Create the GP prior distribution,
        # which we will use to train the model parameters.

        self.gp_joint_model = tfd.JointDistributionNamed({
            **self.prior_dict,
            **dict(
                observations=lambda amplitude, length_scale, x_observed_noise:
                tfd.GaussianProcess(
                    kernel=self.kernel(amplitude, length_scale),
                    index_points=self.x_observed,
                    observation_noise_variance=x_observed_noise
                ))
        })

        # Create the trainable model parameters,
        # which we'll subsequently optimize.
        # Note that we constrain them to be strictly positive

        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

        self.amplitude_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='amplitude',
            dtype=np.float64)

        self.length_scale_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='length_scale',
            dtype=np.float64)

        self.observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='observation_noise_variance_var',
            dtype=np.float64)

        self.trainable_variables = [
            v.trainable_variables[0] for v in [
                self.amplitude_var,
                self.length_scale_var,
                self.observation_noise_variance_var
            ]]

    def target_log_prob(self, amplitude, length_scale, obs_noise_variance):
        return self.gp_joint_model.log_prob({
            'amplitude': amplitude,
            'length_scale': length_scale,
            'x_observed_noise': obs_noise_variance,
            'observations': self.y_observation
        })

    # Use `tf.function` to trace the loss for more efficient evaluation.
    @tf.function(autograph=False, jit_compile=False)
    def train_model(self):
        with tf.GradientTape() as tape:
            loss = -self.target_log_prob(
                self.amplitude_var,
                self.length_scale_var,
                self.observation_noise_variance_var)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def fit(self, num_iterations=1000, verbose_step=10):
        if self.training_log_likelihood.shape[0] == 0:
            self.training_log_likelihood = np.zeros(num_iterations, np.float64)
            idx_base = 0
            loss_nm1 = 0
        else:
            idx_base = self.training_log_likelihood.shape[0]
            loss_nm1 = self.training_log_likelihood[-1]
            self.training_log_likelihood = np.concatenate([
                self.training_log_likelihood,
                np.zeros(num_iterations, np.float64)
            ])

        for idx in range(num_iterations):
            loss = self.train_model()
            self.training_log_likelihood[idx_base + idx] = loss
            if not idx % verbose_step:
                print(f"Iteration : {idx} / {num_iterations}\n"
                      f"Actual Loss = {loss}\n"
                      f"Loss evolution = {loss_nm1 - loss}")
                loss_nm1 = loss

    def predict(self, x_index, num_sample):
        optimized_kernel = self.kernel(
            self.amplitude_var, self.length_scale_var)

        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=x_index,
            observation_index_points=self.x_observed,
            observations=self.y_observation,
            observation_noise_variance=self.observation_noise_variance_var,
            predictive_noise_variance=0.)

        return gprm.sample(num_sample)

    def plot_training_log_likelihood(self):
        if self.training_log_likelihood.shape[0] == 0:
            raise ValueError("Please train the model before"
                             " trying to plot log likelihood")

        # Plot the loss evolution
        plt.figure()
        plt.plot(self.training_log_likelihood)
        plt.xlabel("Training iteration")
        plt.ylabel("Log marginal likelihood")
        plt.show()

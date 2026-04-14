"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""

'''
This module implements Epsilon-Greedy and Thompson Sampling algorithms
for a multi-armed bandit problem with four advertisement options.
 
Bandit: Abstract base class for bandit algorithms.
Visualization: Handles plotting of learning curves and comparisons.
EpsilonGreedy: Epsilon-greedy bandit with 1/t decay.
ThompsonSampling: Thompson sampling with known precision.
'''

############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.
 
    Attributes:
        p (float): The true reward mean of the bandit.
        p_estimate (float): The estimated reward mean.
        N (int): Number of times this bandit has been pulled.
        r_estimate (float): Running reward estimate.
    """
    
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """
        Initialize a bandit with true mean reward p.
 
        Parameters:
            p (float): True mean reward of the bandit.
        """
        self.p = p
        self.p_estimate = 0.0
        self.N = 0
        self.r_estimate = 0.0

    @abstractmethod
    def __repr__(self):
        """Returns string representation of the bandit."""
        return f"Bandit(p={self.p}, estimate={self.p_estimate:.4f}, N={self.N})"

    @abstractmethod
    def pull(self):
        """Pulls the bandit arm and returns a reward."""
        return np.random.randn() + self.p

    @abstractmethod
    def update(self, x):
        """Updates the bandit's estimate based on observed reward x.
        Parameters:
            x (float): The observed reward.
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        
    @abstractmethod
    def experiment(self):
        '''Run the bandit experiment'''
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self, rewards, num_trials, algorithm_name):
        # Visualize the performance of each bandit: linear and log
        
        cumulative_avg = np.cumsum(rewards) / (np.arange(num_trials) + 1)
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
        # Linear scale
        axes[0].plot(cumulative_avg, label=algorithm_name)
        axes[0].set_xlabel("Trials")
        axes[0].set_ylabel("Cumulative Average Reward")
        axes[0].set_title(f"{algorithm_name} - Learning Curve (Linear)")
        axes[0].legend()
 
        # Log scale
        axes[1].plot(cumulative_avg, label=algorithm_name)
        axes[1].set_xscale('log')
        axes[1].set_xlabel("Trials (log scale)")
        axes[1].set_ylabel("Cumulative Average Reward")
        axes[1].set_title(f"{algorithm_name} - Learning Curve (Log)")
        axes[1].legend()
 
        plt.tight_layout()
        plt.savefig(f"{algorithm_name.replace(' ', '_').replace('-', '_')}_learning_curve.png")
        plt.show()
        
    def plot2(self, rewards_eg, rewards_ts, num_trials):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        cum_rewards_eg = np.cumsum(rewards_eg)
        cum_rewards_ts = np.cumsum(rewards_ts)
 
        best_reward = max(Bandit_Reward)
        cum_regret_eg = np.cumsum([best_reward - r for r in rewards_eg])
        cum_regret_ts = np.cumsum([best_reward - r for r in rewards_ts])
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
        # Cumulative rewards
        axes[0].plot(cum_rewards_eg, label="Epsilon-Greedy")
        axes[0].plot(cum_rewards_ts, label="Thompson Sampling")
        axes[0].set_xlabel("Trials")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].set_title("Cumulative Rewards Comparison")
        axes[0].legend()
 
        # Cumulative regret
        axes[1].plot(cum_regret_eg, label="Epsilon-Greedy")
        axes[1].plot(cum_regret_ts, label="Thompson Sampling")
        axes[1].set_xlabel("Trials")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].set_title("Cumulative Regret Comparison")
        axes[1].legend()
 
        plt.tight_layout()
        plt.savefig("comparison.png")
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm with 1/t epsilon decay.
 
    Attributes:
        p (float): True mean reward.
        p_estimate (float): Estimated mean reward.
        N (int): Number of pulls.
        epsilon (float): Initial epsilon value.
    """
    def __init__(self, p):
        super().__init__(p)
        self.p = p
        self.p_estimate = 0.0
        self.N = 0
    
    def __repr__(self):
        """Return string representation."""
        return f"EpsilonGreedy(p={self.p}, estimate={self.p_estimate:.4f}, N={self.N})"

    def pull(self):
        """
        Pull the arm and return a reward sampled from N(p, 1).
 
        Returns:
            float: Sampled reward.
        """
        return np.random.randn() + self.p
    
    def update(self, x):
        """
        Update estimated mean 
 
        Parameters:
            x (float): Observed reward.
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        
    def experiment(self, bandit_rewards, num_trials, initial_epsilon=1.0):
        """
        Run the Epsilon-Greedy experiment across all bandits.
 
        Parameters:
            bandit_rewards (list): List of true mean rewards for each bandit.
            num_trials (int): Number of trials to run.
            initial_epsilon (float): Starting epsilon value.
 
        Returns:
            tuple: (list of rewards, list of bandit indices chosen)
        """
        bandits = [EpsilonGreedy(p) for p in bandit_rewards]
        rewards = []
        chosen_bandits = []
 
        for t in range(1, num_trials + 1):
            epsilon = 1 / t  # decay epsilon by 1/t
 
            if np.random.random() < epsilon:
                # Pick random bandit (explore)
                idx = np.random.randint(len(bandits))
                logger.debug(f"Trial {t}: Exploring bandit {idx} (epsilon={epsilon:.4f})")
            else:
                # Pick bandit with highest estimate (exploit)
                idx = np.argmax([b.p_estimate for b in bandits])
                logger.debug(f"Trial {t}: Exploiting bandit {idx} (epsilon={epsilon:.4f})")
 
            reward = bandits[idx].pull()
            bandits[idx].update(reward)
 
            rewards.append(reward)
            chosen_bandits.append(idx)
 
        self._rewards = rewards
        self._bandits = bandits
        self._chosen = chosen_bandits
        self._num_trials = num_trials
 
        return rewards, chosen_bandits

    def report(self):
        """
        Report results: store in CSV, print cumulative reward and regret.
 
        Returns:
            pd.DataFrame: DataFrame with columns [Bandit, Reward, Algorithm].
        """
        best_reward = max([b.p for b in self._bandits])
        cumulative_reward = sum(self._rewards)
        cumulative_regret = self._num_trials * best_reward - cumulative_reward
 
        logger.info(f"Epsilon-Greedy - Cumulative Reward: {cumulative_reward:.2f}")
        logger.info(f"Epsilon-Greedy - Cumulative Regret: {cumulative_regret:.2f}")
 
        df = pd.DataFrame({
            'Bandit': self._chosen,
            'Reward': self._rewards,
            'Algorithm': 'EpsilonGreedy'
        })
 
        return df

 
#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit algorithm with known precision.
 
    Using Gaussian prior and updating with known precision
 
    Attributes:
        p (float): True mean reward.
        p_estimate (float): Estimated mean reward.
        N (int): Number of pulls.
        tau (float): Known precision (1/variance).
        mu (float): Prior/posterior mean.
        lambda_ (float): Posterior precision.
    """
    
    def __init__(self, p):
        """
        Initialize Thompson Sampling bandit.
 
        Parameters:
            p (float): True mean reward of the bandit.
        """
        super().__init__(p)
        self.p = p
        self.p_estimate = 0.0
        self.N = 0
        self.tau = 1.0       # known precision
        self.mu = 0.0        # prior mean
        self.lambda_ = 1.0   # prior precision
        
    def __repr__(self):
        """Return string representation."""
        return f"ThompsonSampling(p={self.p}, mu={self.mu:.4f}, N={self.N})"
    
    def pull(self):
        """
        Pull the arm and return a reward sampled from N(p, 1).
 
        Returns:
            float: Sampled reward.
        """
        return np.random.randn() + self.p
    
    def update(self, x):
        """
        Update posterior parameters using Bayesian update.
 
        Parameters:
            x (float): Observed reward.
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        # Bayesian update for Gaussian with known precision
        new_lambda = self.lambda_ + self.tau
        self.mu = (self.lambda_ * self.mu + self.tau * x) / new_lambda
        self.lambda_ = new_lambda

    def sample(self):
        """
        Sample from the posterior distribution.
 
        Returns:
            float: A sample from N(mu, 1/lambda).
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.mu

    def experiment(self, bandit_rewards, num_trials):
        """
        Run the Thompson Sampling experiment across all bandits.
 
        Parameters:
            bandit_rewards (list): List of true mean rewards for each bandit.
            num_trials (int): Number of trials to run.
 
        Returns:
            tuple: (list of rewards, list of bandit indices chosen)
        """
        bandits = [ThompsonSampling(p) for p in bandit_rewards]
        rewards = []
        chosen_bandits = []
 
        for t in range(1, num_trials + 1):
            # Sample from each bandit's posterior and picks the highest
            samples = [b.sample() for b in bandits]
            idx = np.argmax(samples)
            logger.debug(f"Trial {t}: Chose bandit {idx} (samples={[f'{s:.3f}' for s in samples]})")
 
            reward = bandits[idx].pull()
            bandits[idx].update(reward)
 
            rewards.append(reward)
            chosen_bandits.append(idx)
 
        self._rewards = rewards
        self._bandits = bandits
        self._chosen = chosen_bandits
        self._num_trials = num_trials
 
        return rewards, chosen_bandits

    def report(self):
        """
        Report results: store in CSV, print cumulative reward and regret.
 
        Returns:
            pd.DataFrame: DataFrame with columns [Bandit, Reward, Algorithm].
        """
        best_reward = max([b.p for b in self._bandits])
        cumulative_reward = sum(self._rewards)
        cumulative_regret = self._num_trials * best_reward - cumulative_reward
 
        logger.info(f"Thompson Sampling - Cumulative Reward: {cumulative_reward:.2f}")
        logger.info(f"Thompson Sampling - Cumulative Regret: {cumulative_regret:.2f}")
 
        df = pd.DataFrame({
            'Bandit': self._chosen,
            'Reward': self._rewards,
            'Algorithm': 'ThompsonSampling'
        })
 
        return df
    
def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    """
    Compare Epsilon-Greedy and Thompson Sampling visually and numerically.
 
    Runs both algorithms on the same bandit rewards, generates plots,
    stores results in a CSV file and prints cumulative metrics.
    """
    viz = Visualization()
 
    # Epsilon Greedy
    logger.info("Starting Epsilon-Greedy experiment")
    eg = EpsilonGreedy(0)
    rewards_eg, chosen_eg = eg.experiment(Bandit_Reward, NumberOfTrials)
    df_eg = eg.report()
    viz.plot1(rewards_eg, NumberOfTrials, "Epsilon-Greedy")
 
    # Thompson Sampling
    logger.info("Starting Thompson Sampling experiment")
    ts = ThompsonSampling(0)
    rewards_ts, chosen_ts = ts.experiment(Bandit_Reward, NumberOfTrials)
    df_ts = ts.report()
    viz.plot1(rewards_ts, NumberOfTrials, "Thompson Sampling")
 
    # Comparison
    viz.plot2(rewards_eg, rewards_ts, NumberOfTrials)
 
    # Save to CSV 
    df = pd.concat([df_eg, df_ts], ignore_index=True)
    df.to_csv("bandit_rewards.csv", index=False)
    logger.info("Results saved to bandit_rewards.csv")

# Global Parameters
Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
    
    comparison()

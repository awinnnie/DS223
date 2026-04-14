# A/B Testing with Multi-Armed Bandits

## Overview

This project compares two multi-armed bandit algorithms, **Epsilon-Greedy** (with 1/t epsilon decay) and **Thompson Sampling** (with known precision) on a simulated A/B testing scenario with four advertisement options.

Each advertisement (bandit) has a true mean reward of [1, 2, 3, 4], and the algorithms compete over 20,000 trials to identify and exploit the best-performing ad.

## Project Structure

```
A_B_Testing/
├── Bandit.py           # Main script with all classes and experiment logic
├── requirements.txt    # Python dependencies
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python Bandit.py
```

## Output

Running the script produces:

- **Epsilon_Greedy_learning_curve.png** - cumulative average reward over trials (linear and log scale)
- **Thompson_Sampling_learning_curve.png** - cumulative average reward over trials (linear and log scale)
- **comparison.png** - side-by-side cumulative reward and cumulative regret for both algorithms
- **bandit_rewards.csv** - all trial data with columns: Bandit, Reward, Algorithm

Cumulative reward and regret for each algorithm are printed to the console.

## Algorithms

### Epsilon-Greedy
- Explores randomly with probability epsilon, exploits the best-known bandit otherwise
- Epsilon decays as 1/t, so exploration decreases over time
- Simple but suboptimal — still wastes some trials exploring even when confident

### Thompson Sampling
- Maintains a Gaussian posterior (mean and precision) for each bandit
- Each trial, samples from each posterior and picks the highest sample
- Exploration happens naturally through posterior uncertainty, no tuning needed
- Generally converges to the optimal bandit faster than Epsilon-Greedy

## Bonus: UCB1 (Upper Confidence Bound)

As an improved implementation, UCB1 is included as a third algorithm. UCB1 selects the arm that maximizes:

score = estimated_mean + sqrt(2 * log(N) / n_j)

where N is the total number of trials and n_j is the number of times arm j has been pulled. This formula is derived from Hoeffding's inequality, providing a mathematically justified upper bound on each arm's true mean.

**Why UCB1 is better:**
- No hyperparameters to tune (unlike Epsilon-Greedy's epsilon or decay schedule)
- Deterministic selection rule (unlike Thompson Sampling's random sampling)
- Exploration is driven by uncertainty, not randomness — arms with fewer pulls get higher scores automatically
- Achieves the lowest cumulative regret in our experiments

## Dependencies

- numpy
- matplotlib
- pandas
- loguru
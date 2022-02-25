import argparse
import torch
import numpy as np
import random
import os
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="whitegrid")
import envs as bandits

# from envs.linear import LinearContinuous
from algorithms.greedy_agent import GreedyAgent


def parse_args():
    parser = argparse.ArgumentParser("Linear Bandit")
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save-dir', type=str, default='save_dir')
    parser.add_argument('--horizon', type=int, default=4000)
    # env options
    # env options
    parser.add_argument('--dim', type=int, default=20, metavar='Context dimension')
    parser.add_argument('--narms', type=int, default=5, metavar='Number of actions')
    # parser.add_argument('--horizon', type=int, default=20000, metavar='Horizon of the bandit problem')
    parser.add_argument('--seed', type=int, default=0, metavar='Seed used for the generation of the bandit problem')
    parser.add_argument('--bandittype', default="expanded", help="None, expanded, onehot")
    parser.add_argument('--contextgeneration', default="uniform", help="uniform, gaussian, bernoulli")
    # algo options
    # parser.add_argument('--context-dim', type=int, default=10)
    # parser.add_argument('--num-actions', type=int, default=2)
    # parser.add_argument('--noise', type=float, default=0.05, help='the level of reward noise')
    # network options
    parser.add_argument('--layer-sizes', type=list, default=[200])
    parser.add_argument('--batch-size', type=int, default=128)
    # optimizer options
    parser.add_argument('--initial-lr', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=5.0)

    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--initial-pulls', type=int, default=2)
    parser.add_argument('--training-freq', type=int, default=100)
    parser.add_argument('--training-steps', type=int, default=200)

    return parser.parse_args()


def main():
    args = parse_args()
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # don't throw data
    args.replay_buffer = args.horizon

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create Env
    # env = LinearContinuous(context_dim=args.context_dim, num_actions=args.num_actions, noise=args.noise)
    noise_std = 0.5

    env = bandits.LinearContinuous(
        context_dim=args.dim, num_actions=args.narms, context_generation=args.contextgeneration,
        feature_expansion=args.bandittype, seed=args.seed, noise="gaussian", noise_param=noise_std
    )
    args.context_dim = args.dim
    args.num_actions = args.narms
    agent = GreedyAgent('greedy agent', args)

    h_actions = []
    h_rewards = []
    h_regrets = []
    h_loss = []
    for h in range(args.horizon):
        context = env.sample_context() # expanded feature num_actions x (context_dim x num_actions)
        features = env.features()
        action = agent.action(features)
        reward = env.step(action)
        loss = agent.update(features, action, reward)  # update replay and train model
        best_reward, best_action = env.best_reward_and_action()
        if loss is not None:
            h_loss.append(loss)
            # import pdb
            # pdb.set_trace()
            print('loss: {:10.4f}'.format(loss))
        h_actions.append(action)
        h_rewards.append(reward)
        h_regrets.append(best_reward - reward)


    total_regret = np.cumsum(h_regrets, axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(x=np.arange(total_regret.shape[0]),
                 y=total_regret,
                 legend='brief',
                 label=agent.name)
    ax.legend()
    plt.savefig('regret.png')

if __name__=='__main__':
    main()







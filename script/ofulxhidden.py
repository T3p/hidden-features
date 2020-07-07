from hiddenfeat.bandits.finite_linear_bandits import make_random_hflb
from hiddenfeat.utils import seed_all
from hiddenfeat.algos.hidden_linear import oful, hoful, incr_hoful, offset_approach, ucb1
import argparse


parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algo', type=str, default='ihoful')
parser.add_argument('--S', type=int, default=100)
parser.add_argument('--A', type=int, default=10)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--hidden', type=int, default=2)
parser.add_argument('--it', type=int, default=10000)
parser.add_argument('--reg1', type=float, default=0.1)
parser.add_argument('--reg2', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--pb', type=float, default=1)
parser.add_argument('--fb', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

seed_all(args.seed)

bandit = make_random_hflb(n_contexts=args.S, n_arms=args.A, dim=args.dim, 
                          noise=args.noise, hidden=args.hidden)

if args.algo == 'oful':
    oful(bandit, args.it, reg=args.reg1, noise=args.noise, delta=args.delta, 
         param_bound=args.pb, seed=args.seed)
elif args.algo == 'offset':
    offset_approach(bandit, args.it, reg=args.reg1, noise=args.noise, 
                    delta=args.delta, param_bound=args.pb, seed=args.seed)
elif args.algo == 'ucb':
    ucb1(bandit, args.it, seed=args.seed)
else:
    if args.algo == 'hoful':
        algo = hoful
    elif args.algo == 'ihoful':
        algo = incr_hoful
    else:
        raise NotImplementedError
    algo(bandit, args.it, reg1=args.reg1, reg2=args.reg2, noise=args.noise, 
         delta=args.delta, param_bound=args.pb, feat_bound=args.fb, 
         seed=args.seed)

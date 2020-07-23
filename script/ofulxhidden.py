from hiddenfeat.bandits.finite_linear_bandits import make_random_hflb
from hiddenfeat.utils import seed_all
from hiddenfeat.algos.hidden_linear import oful, hoful, span_hoful
import argparse


parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--algo', type=str, default='hoful')
parser.add_argument('--S', type=int, default=10)
parser.add_argument('--A', type=int, default=3)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--hidden', type=int, default=1)
parser.add_argument('--it', type=int, default=10000)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--pb', type=float, default=1)
parser.add_argument('--ob', type=float, default=1)
parser.add_argument('--fb', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fulldim', type=int, default=None)
args = parser.parse_args()

seed_all(args.seed)

bandit = make_random_hflb(n_contexts=args.S, n_arms=args.A, dim=args.dim, 
                          noise=args.noise, hidden=args.hidden)

if args.algo == 'oful':
    oful(bandit, args.it, reg=args.reg, noise=args.noise, delta=args.delta, 
         param_bound=args.pb, seed=args.seed)
elif args.algo == 'hoful':
    hoful(bandit, args.it, reg=args.reg, noise=args.noise, delta=args.delta, 
          off_scale=args.scale, param_bound=args.pb, off_bound=args.ob, seed=args.seed)
elif args.algo == 'span':
    span_hoful(bandit, args.it, reg=args.reg, noise=args.noise, delta=args.delta, 
                   param_bound=args.pb, seed=args.seed, fulldim=args.fulldim)
else:
    raise NotImplementedError
    
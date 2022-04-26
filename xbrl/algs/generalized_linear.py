import numpy as np
from typing import Optional, Any, Callable
from xbrl.algs.batched.templates import XBModule
from xbrl.algs.linear import inv_sherman_morrison
from scipy.optimize import root
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
import warnings
import cvxpy as cp

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_nonlinearity(param_bound, features_bound):
    z = param_bound * features_bound + 1
    return 2 * (np.cosh(z) + 1)

#optimized for logistic bandits
class UCBGLM(XBModule):

    def __init__(
        self, env: Any,
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        bonus_scale: Optional[float]=1.,
        opt_tolerance=1e-8,
        param_bound = 1.,
        features_bound = 1.,
        true_param=None, #for testing purposes!
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.bonus_scale = bonus_scale
        self.link_function = sigmoid
        self.nonlinearity_function = sigmoid_nonlinearity
        self.link_derivative = sigmoid_derivative
        self.solver = LogisticRegression(fit_intercept=False,
                                         random_state=np.random.RandomState(seed=seed),
                                         solver='lbfgs',
                                         warm_start=True,
                                         tol=opt_tolerance,
                                         max_iter=1000)
        self.param_bound = param_bound
        self.features_bound = features_bound
        self.true_param = true_param
        self.rng = np.random.RandomState(seed=seed)        
        self.dim = self.env.dim
        self.update_every_n_steps = update_every_n_steps
        self.nonlinearity_coeff = self.nonlinearity_function(self.param_bound, self.features_bound)
        self.mineig_threshold = 16 * self.noise_std**2 * self.nonlinearity_coeff**2 + (self.dim + np.log(1 / self.delta))

        
    def reset(self) -> None:
        super().reset()
        dim = self.dim
        self.A = np.zeros(shape=(dim, dim))
        self.A_logdet = None
        self.inv_A = None
        self.theta = np.zeros(dim) if self.true_param is None else self.true_param
        self.new_inv_A = None
        self.features_history = [np.zeros(dim), np.zeros(dim)]
        self.reward_history = [0., 1.]
        self.exp_phase = True

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        
        mineig = np.amin(np.linalg.eig(self.A)[0])
        
        if mineig < self.mineig_threshold:
            action = self.rng.choice(self.env.action_space.n)
        else:
            if self.exp_phase==True:
                print('Completed exploratory phase in %d rounds' % (self.t - 1))
                self.inv_A = np.linalg.inv(self.A)
                self.new_inv_A = self.inv_A.copy()
                self.A_logdet = np.log(np.linalg.det(self.A))
                self.exp_phase = False
            #beta = nonlinearity_coeff * self.noise_std * np.sqrt(self.A_logdet - 2*np.log(self.ucb_regularizer**(dim / 2) * self.delta )) + np.sqrt(self.ucb_regularizer) * self.param_bound
            #beta = nonlinearity_coeff * self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2*self.t/self.ucb_regularizer)/self.delta)) + self.param_bound * np.sqrt(self.ucb_regularizer)
            #beta = np.sqrt(np.log(self.t+1))
            beta =  self.nonlinearity_coeff * self.noise_std * np.sqrt(self.dim / 2 * np.log(
                    (1 + 2 * self.features_bound**2 * self.t / self.dim) / self.delta))
            # get features for each action and make it tensor
            bonus = ((features @ self.inv_A)*features).sum(axis=1)
            bonus = self.bonus_scale * beta * np.sqrt(bonus)
            ucb = features @ self.theta + bonus
            action = np.argmax(ucb).item()
            #print(bonus[action].item())
            self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
            
        assert 0 <= action < self.env.action_space.n, ucb
        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        assert reward==0 or reward==1

        self.features_history.append(features)
        self.reward_history.append(reward)
        #self.features_bound = max(self.features_bound, np.linalg.norm(v, 2).item())
        self.writer.add_scalar('features_bound', self.features_bound, self.t)
        self.A = self.A + np.outer(features, features)
        if self.exp_phase==False:
            self.new_inv_A, den = inv_sherman_morrison(features, self.new_inv_A)
            self.A_logdet += np.log(den)
        
    def train(self) -> float:
        if self.t % self.update_every_n_steps == 0:
            if self.true_param is None:
                X = np.stack(self.features_history, axis=0) #txd
                y = np.array(self.reward_history)
                self.solver.fit(X, y)
                
                
                
                self.theta = self.solver.coef_.ravel()
                if np.linalg.norm(self.theta) >= self.param_bound:
                    self.theta = self.theta / np.linalg.norm(self.theta) * self.param_bound
            self.inv_A = self.new_inv_A
        
        return 0
    
"""
Efficient GLM for logistic bandits
https://arxiv.org/abs/1509.07728
Assumptions: feature norm <= 1, parameter bound known
"""
class OL2M(XBModule):

    def __init__(
        self, env: Any,
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        param_bound: float=1.,
        delta: Optional[float]=0.01,
        step_size: float=0.01,
        ucb_regularizer: Optional[float]=1.,
        bonus_scale: Optional[float]=1.
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.param_bound = param_bound
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale
        self.nl_coeff = 2 * (1 + np.exp(self.param_bound)) #nonlinearity coefficient
        self.step_size = step_size
        self.features_bound = 1.
        self.bonus_const = max(self.ucb_regularizer, 
                               self.step_size/(2*self.nl_coeff)) * self.param_bound

    def reset(self) -> None:
        super().reset()
        self.dim = dim = self.env.feature_dim
        self.A = np.eye(dim) * self.ucb_regularizer
        self.inv_A = np.eye(dim) / self.ucb_regularizer
        self.theta = np.zeros(dim)
        self.new_A = np.eye(dim) * self.ucb_regularizer
        self.new_inv_A = np.eye(dim) / self.ucb_regularizer
        self.new_theta = np.zeros(dim)
        self.det_ratio = 1.
        self.new_det_ratio = 1.

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        tau = np.log(2*np.ceil(2*np.log2(max(2,self.t))) * 
                     max(1,self.t)**2 / self.delta)
        logdet = np.log(self.det_ratio)
        #logdet = self.dim * np.log(1 + self.step_size/(2*self.nl_coeff*self.dim*self.ucb_regularizer)*self.t)
        #beta = np.sqrt(2*self.step_size*(4*self.param_bound + (4*self.nl_coeff+8./3*self.param_bound)*tau + self.nl_coeff*logdet) + self.bonus_const)
        beta = np.sqrt(np.log(self.t+1))
        # get features for each action and make it tensor
        bonus = ((features @ self.inv_A)*features).sum(axis=1)
        bonus = self.bonus_scale * beta * np.sqrt(bonus)
        ucb = features @ self.theta + bonus
        action = np.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        assert 0 <= action < self.env.action_space.n, ucb

        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        # exp = (features, reward)
        # self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        v = features
        assert reward == 1 or reward == 0
        #self.features_bound = max(self.features_bound, np.linalg.norm(v, 2).item())
        #self.writer.add_scalar('features_bound', self.features_bound, self.t)

        self.new_A = self.new_A + self.step_size/(2*self.nl_coeff) * np.outer(v, v)
        self.new_inv_A, self.new_det_ratio = inv_sherman_morrison(
            v*np.sqrt(self.step_size/(2*self.nl_coeff)), self.new_inv_A)
        
        #Compute new parameter by solving convex problem
        #note that it is a trust region subproblem so it could be solved even more efficiently with line search
        x = cp.Variable(self.dim)
        b = self.step_size * v * (sigmoid(self.new_theta.T @ v) - reward)
        A = cp.atoms.affine.wraps.psd_wrap(self.new_A)
        I = cp.atoms.affine.wraps.psd_wrap(np.eye(self.dim))

        objective = cp.quad_form(x, A) / 2 + b.T @ x
        constr = (cp.quad_form(x, I) + 2*self.new_theta.T @ x <= 
                  self.param_bound**2 - np.dot(self.new_theta,self.new_theta))
        problem = cp.Problem(cp.Minimize(objective), [constr])
        problem.solve()
        assert(problem.status=="optimal")
        self.new_theta = self.new_theta + x.value 

    def train(self) -> float:
        if self.t % self.update_every_n_steps == 0:
            self.inv_A = self.new_inv_A
            self.theta = self.new_theta
            self.det_ratio = self.new_det_ratio
        
        return 0

"""
"""
class UCBGLM_general(XBModule):

    def __init__(
        self, env: Any,
        seed: Optional[int] = 0, 
        update_every_n_steps: Optional[int] = 1,
        noise_std: float=1,
        delta: Optional[float]=0.01,
        ucb_regularizer: Optional[float]=1,
        bonus_scale: Optional[float]=1.,
        link_function: Callable[[float], float] = sigmoid,
        nonlinearity_function: Callable[[float, float], float] = sigmoid_nonlinearity,
        link_derivative: Optional[Callable[[float], float]] = sigmoid_derivative,
    ) -> None:
        super().__init__(env, None, None, None, None, None, None, 0, seed, None, update_every_n_steps)
        self.np_random = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.delta = delta
        self.ucb_regularizer = ucb_regularizer
        self.bonus_scale = bonus_scale
        self.link_function = link_function
        self.nonlinearity_function = nonlinearity_function
        self.link_derivative = link_derivative
        self.retry_root_finding = False
        
    def reset(self) -> None:
        super().reset()
        dim = self.env.feature_dim
        self.b_vec = np.zeros(dim)
        self.inv_A = np.eye(dim) / self.ucb_regularizer
        self.theta = np.zeros(dim)
        self.new_b_vec = np.zeros(dim)
        self.new_inv_A = np.eye(dim) / self.ucb_regularizer
        #self.new_theta = np.zeros(dim)
        self.param_bound = 1
        self.features_bound = 1
        self.features_history = []

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.env.action_space.n
        dim = features.shape[1]
        nonlinearity_coeff = self.nonlinearity_function(1. + self.param_bound, self.features_bound)
        #beta = nonlinearity_coeff * self.noise_std * np.sqrt(dim * np.log((1+self.features_bound**2*self.t/self.ucb_regularizer)/self.delta)) + self.param_bound * np.sqrt(self.ucb_regularizer)
        beta = np.sqrt(np.log(self.t+1))
        # get features for each action and make it tensor
        bonus = ((features @ self.inv_A)*features).sum(axis=1)
        bonus = self.bonus_scale * beta * np.sqrt(bonus)
        ucb = features @ self.theta + bonus
        action = np.argmax(ucb).item()
        self.writer.add_scalar('bonus selected action', bonus[action].item(), self.t)
        #print(bonus[action].item())
        assert 0 <= action < self.env.action_space.n, ucb

        return action

    def add_sample(self, context: np.ndarray, action: int, reward: float, features: np.ndarray) -> None:
        # exp = (features, reward)
        # self.buffer.append(exp)

        # estimate linear component on the embedding + UCB part
        v = features
        self.features_history.append(features)
        #self.features_bound = max(self.features_bound, np.linalg.norm(v, 2).item())
        self.writer.add_scalar('features_bound', self.features_bound, self.t)

        self.new_b_vec = self.new_b_vec + v * reward
        self.new_inv_A, den = inv_sherman_morrison(v, self.new_inv_A)
        # self.A_logdet += np.log(den)
        
        #self.new_theta = self.inv_A @ self.b_vec

    def train(self) -> float:
        if self.t % self.update_every_n_steps == 0 or self.retry_root_finding:
            feature_matrix = np.stack(self.features_history, axis=0) #txd
            
            def mle_equation(theta):
                predictions = self.link_function(feature_matrix @ theta) #t
                return predictions @ feature_matrix - self.new_b_vec # d
            
            def mle_jacobian(theta):
                derivatives = self.link_derivative(feature_matrix @ theta) #t
                jacobian = np.einsum('i,ij,ik->kj', derivatives, feature_matrix, feature_matrix)  #dxd (pre-transposed for efficiency)
                return jacobian + np.eye(jacobian.shape[0]) * 1e-12
  
            theta_0 = self.new_inv_A @ self.new_b_vec
            sol = root(mle_equation, theta_0, jac=mle_jacobian, method='lm',
                       options={'col_deriv': True,
                                        'xtol': 1e-6, #1.49012e-08, 
                                        'ftol': 1e-6, #1.49012e-08, 
                                        'gtol': 1e-12, #0.0, 
                                        'maxiter': 100, #0 means 100*(d+1) 
                                        'eps': 0.0, 
                                        'factor': 100, 
                                        'diag': None})
            theta = sol.x
            success = sol.success
            if success:
                self.retry_root_finding = False
                self.theta = theta
                #self.param_bound = max(self.param_bound, np.linalg.norm(self.theta, 2).item())
                #self.writer.add_scalar('param_bound', self.param_bound, self.t)
                self.inv_A = self.new_inv_A
                self.b_vec = self.new_b_vec
            else:
                self.retry_root_finding = True
                print("Root not found, will try again next round")
            
        
        return 0
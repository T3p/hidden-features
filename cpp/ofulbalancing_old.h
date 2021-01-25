#ifndef OFULBAL_H
#define OFULBAL_H

#include <Eigen/Dense>
#include "abstractclasses.h"
#include <algorithm>

using namespace std;
using namespace Eigen;

/**
 * Implementation of bandit model selection by regret balancing from
 * 
 * Yasin Abbasi-Yadkori, Aldo Pacchiano, My Phan:
 * Regret Balancing for Bandit and RL Model Selection
 * https://arxiv.org/abs/2006.05491
 * 
 * This is a specific implementation for selecting the representation of
 * OFUL algorithm.
 */
template<typename X>
class OFULBalance : public Algo<X>
{
public:

    OFULBalance(
        std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep, 
        double reg_val, double noise_std, double bonus_scale=1., 
        double delta=0.01, bool adaptive_ci=true
    )
        : Algo<X>("OFULBalance"), linrep(linrep), 
        reg_val(reg_val), noise_std(noise_std), 
        bonus_scale(bonus_scale), delta(delta), adaptive_ci(adaptive_ci)
    {
        reset();
    }
    ~OFULBalance() {}

    void reset()
    {

        int nreps = linrep.size();
        assert(nreps > 0);
        int n_arms = linrep[0]->n_arms();
        inv_A.resize(nreps);
        b_vec.resize(nreps);
        features_bound.resize(nreps);
        param_bound.resize(nreps);
        num_selection.resize(nreps);
        cum_rewards.resize(nreps);
        for (int i = 0, ii = linrep.size(); i < ii; ++i)
        {
            int dim = linrep[i]->features_dim();
            inv_A[i] = MatrixXd::Identity(dim, dim) / reg_val;
            b_vec[i] = VectorXd::Zero(dim);
            features_bound[i] = linrep[i]->features_bound();
            param_bound[i] = linrep[i]->param_bound();
            cum_rewards[i] = 0;
            num_selection[i] = 0;
            assert(n_arms == linrep[i]->n_arms());
        }
        t = 1;
    }

    double _upper_bound(int i) {
        return -log(this->inv_A[i].determinant() * this->delta) * sqrt(this->num_selection[i]);
    }

    int _optimistic_action(int base_alg, int context) {
        int n_arms = linrep[base_alg]->n_arms();
        int dim = linrep[base_alg]->features_dim();
        VectorXd theta = inv_A[base_alg] * b_vec[base_alg];
        VectorXd ucb = VectorXd::Zero(n_arms);// + 1e-15 * VectorXd::Random(n_arms);
        double max_ucb, beta;
        int action;
        if (adaptive_ci)
        {
                double val = log(sqrt(inv_A[base_alg].determinant()) * pow(reg_val, dim/2) * delta);
                beta = noise_std * sqrt(-2 * val) + param_bound[base_alg] * sqrt(reg_val);
        }
        else
        {
                beta = noise_std * sqrt(dim * log((1+features_bound[base_alg]*features_bound[base_alg]*t/reg_val)/delta)) + param_bound[base_alg] * sqrt(reg_val);
        }
        for (int a = 0; a < n_arms; ++a)
        {
            VectorXd v = linrep[base_alg]->get_features(context, a);
            ucb[a] = v.dot(theta) +  bonus_scale * beta * sqrt(v.dot(inv_A[base_alg] * v));

            if (a ==0 || max_ucb < ucb[a])
            {
                action = a;
                max_ucb = ucb[a];
            }
        }
        return action;
    }

    int action(const X& context)
    {
        int n_rep = linrep.size();

        // compute empirical regret
        int opt_base = 0;
        double opt_value = 0;
        for (int i = 0; i < n_rep; ++i)
        {
            double ub = _upper_bound(i);
            double u = (cum_rewards[i] + ub) / num_selection[i];
            if (i == 0 || u > opt_value) {
                opt_value = u;
                opt_base = i;
            }
        }
        
        double min_empreg;
        for (int i = 0; i < n_rep; ++i)
        {
            double emp_regret = num_selection[i] * opt_value - cum_rewards[i];
            if (i == 0 || emp_regret < min_empreg) {
                last_selected_rep = i;
                min_empreg = emp_regret;
            }
        }

        double action = _optimistic_action(last_selected_rep, context);

        t++;
        return action;
    }

    void update(const X& context, int action, double reward)
    {
        VectorXd v = linrep[last_selected_rep]->get_features(context, action);
        // update b
        b_vec[last_selected_rep] += v * reward;
        // Sherman–Morrison formula
        double den = 1. + v.dot(inv_A[last_selected_rep]*v);
        MatrixXd m = (inv_A[last_selected_rep]*v*v.transpose()*inv_A[last_selected_rep]) / den;
        inv_A[last_selected_rep] -= m;

        num_selection[last_selected_rep]++;
        cum_rewards[last_selected_rep] += reward;
    }

public:
    std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep;
    double reg_val, noise_std, bonus_scale, delta;
    bool adaptive_ci;
    std::vector<MatrixXd> inv_A;
    std::vector<VectorXd> b_vec;
    std::vector<double> features_bound;
    std::vector<double> param_bound;
    std::vector<double> cum_rewards;
    std::vector<int> num_selection;
    double t;
    int last_selected_rep;
};

/**
 * Implementation of bandit model selection by regret balancing
 * and elimination from
 * 
 * Aldo Pacchiano, Christoph Dann, Claudio Gentile, Peter Bartlett:
 * Regret Bound Balancing and Elimination for Model Selection in Bandits and RL
 * https://arxiv.org/abs/2012.13045
 * 
 * This is a specific implementation for selecting the representation of
 * OFUL algorithm.
 */
template<typename X>
class OFULBalanceAndEliminate : public OFULBalance<X>
{
public:
    OFULBalanceAndEliminate(
        std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep, 
        double reg_val, double noise_std, double bonus_scale=1., double delta=0.01, bool adaptive_ci=true
    )
    : OFULBalance<X>(linrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci)
    {
        reset();
        this->name = "OFULBalanceAndEliminate";
    }
    ~OFULBalanceAndEliminate() {}

    void reset()
    {
        OFULBalance<X>::reset();
        // all the representations are active [0, M]
        active_reps.resize(this->linrep.size());
        std::iota(active_reps.begin(), active_reps.end(), 0);
    }

    int action(const X& context)
    {
        //select base learner
        double max_ub;
        for(int i : active_reps)
        {
            double ub = this->_upper_bound(i);
            if ((i == 0) || (ub < max_ub)) {
                this->last_selected_rep = i;
                max_ub = ub;
            }
        }
        double action = this->_optimistic_action(this->last_selected_rep, context);
        this->t++;
        return action;
    }

    void update(const X& context, int action, double reward)
    {
        OFULBalance<X>::update(context, action, reward);
        double delta = this->delta;

        double XXX = 2, M = this->linrep.size();
        //eliminate representation
        double max_value = -1;
        for(int i : active_reps)
        {
            double N = max(1, this->num_selection[i]);
            double value = this->cum_rewards[i] / N - XXX * sqrt(log(M * log(N/delta))/ N);
            if (max_value < value) {
                max_value = value;
            }
        }
        std::vector<int> new_active_reps;
        for(int i : active_reps)
        {
            double N = max(1, this->num_selection[i]);
            double ub = this->_upper_bound(i);
            double lhs = this->cum_rewards[i] / N + XXX * sqrt(log(M * log(N/delta))/ N) + ub / N;
            if (lhs >= max_value) {
                new_active_reps.push_back(i);
            } else {
                cout << "eliminated " << i << endl;
            }
        }
        active_reps = new_active_reps;
    }

    public:
    std::vector<int> active_reps;
};
#endif

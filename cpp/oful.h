#ifndef OFUL_H
#define OFUL_H

#include <Eigen/Dense>
#include "abstractclasses.h"

using namespace std;
using namespace Eigen;

template<typename X>
class OFUL : public BaseAlgo<X>
{
public:

    OFUL(
        ContRepresentation<X>& linrep,
        double reg_val,
        double noise_std,
        double bonus_scale=1.,
        double delta=0.01, bool adaptive_ci=true
    )
        : BaseAlgo<X>("OFUL"), linrep(linrep),
          reg_val(reg_val), noise_std(noise_std),
          bonus_scale(bonus_scale), delta(delta), adaptive_ci(adaptive_ci)
    {
        reset();
        features_bound = linrep.features_bound();
        param_bound = linrep.param_bound();
    }

    OFUL(const OFUL& other)
        : OFUL(
              other.linrep, other.reg_val, other.noise_std,
              other.bonus_scale, other.delta, other.adaptive_ci
          ) {}

    ~OFUL() {}

    void reset()
    {
        int dim = linrep.features_dim();
        inv_A = MatrixXd::Identity(dim, dim) / reg_val;
        b_vec = VectorXd::Zero(dim);
        A_logdet = log(reg_val) * dim;
        UCBindex.resize(linrep.n_arms());
        t = 1;
    }

    int action(const X& context)
    {
        int n_arms = linrep.n_arms();
        int dim = linrep.features_dim();

        VectorXd theta = inv_A * b_vec;
        // VectorXd ucb = VectorXd::Zero(n_arms) + 1e-15 * VectorXd::Random(n_arms);
        // std::fill(UCBindex.begin(), UCBindex.end(), 0.);
        double max_ucb, beta;
        int action;
        if (adaptive_ci)
        {
            double val = A_logdet - dim * log(reg_val) - 2 * log(delta);
            beta = noise_std * sqrt(val) + param_bound * sqrt(reg_val);
        }
        else
        {
            beta = noise_std * sqrt(dim * log((1+features_bound*features_bound*t/reg_val)/delta)) + param_bound * sqrt(reg_val);
        }
        for (int a = 0; a < n_arms; ++a)
        {
            VectorXd v = linrep.get_features(context, a);
            UCBindex[a] = v.dot(theta) + bonus_scale * beta * sqrt(v.dot(inv_A * v));

            if (a ==0 || max_ucb < UCBindex[a])
            {
                action = a;
                max_ucb = UCBindex[a];
            }
        }
        t++;
        // cout << endl << ucb << endl;
        return action;
    }

    std::vector<double> action_distribution(const X& context) {
        int n_arms = linrep.n_arms();
        std::vector<double> proba(n_arms);
        proba[action(context)] = 1;
        return proba;
    }

    void update(const X& context, int action, double reward)
    {
        VectorXd v = linrep.get_features(context, action);
        // update b
        b_vec.noalias() += v * reward;
        // Sherman–Morrison formula
        double den = 1. + v.dot(inv_A*v);
        inv_A.noalias() -= (inv_A*(v*(v.transpose()*inv_A))) / den;
        A_logdet += log(den);
    }

    double upper_bound()
    {
        return -log(inv_A.determinant() * delta) * sqrt(t);
    }

    std::unique_ptr<BaseAlgo<X>> clone() const
    {
        auto copied = new OFUL<X>(*this);
        copied->inv_A = inv_A;
        copied->b_vec = b_vec;
        return std::unique_ptr<OFUL<X>>(copied);
    }

    double corral_lr(const int horizon, const int nbases) const
    {
        int dim=linrep.features_dim();
        return sqrt(1.*nbases / horizon) / dim;
    }

    double exp3_rate(const int horizon, const int nbases) const
    {
        int dim=linrep.features_dim();
        return pow(horizon, -1./3.) * pow(nbases, -2./3.) * pow(dim, 2./3.);
    }

public:
    ContRepresentation<X>& linrep;
    double reg_val, noise_std, bonus_scale, delta;
    bool adaptive_ci;
    MatrixXd inv_A;
    VectorXd b_vec;
    double features_bound, param_bound, t;
    std::vector<double> UCBindex;
    double A_logdet;
};
#endif

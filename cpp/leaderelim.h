#ifndef LEADERELIM_H
#define LEADERELIM_H

#include <Eigen/Dense>
#include "leader.h"

using namespace std;
using namespace Eigen;

template<typename X>
class LEADER_MSEELIM : public LEADER<X>
{
public:

    LEADER_MSEELIM(
        std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep,
        double reg_val, double noise_std,
        double bonus_scale=1., double delta=0.01, bool adaptive_ci=true
    )
        : LEADER<X>(linrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci)
    {
        reset();
    }
    ~LEADER_MSEELIM() {}

    void reset()
    {
        LEADER<X>::reset();

        context_history.reserve(10000);
        action_history.reserve(10000);
        reward_history.reserve(10000);
        active_reps.resize(this->oful_algs.size());
        std::iota(active_reps.begin(), active_reps.end(), 0);
        std::cout << "active_set: \n";
                for (auto& i : active_reps) {
                    std::cout << i << " ";
                }
                std::cout << std::endl << "--------\n";
    }

    int action(const X& context)
    {
        // cout << t << ", " << context << endl;
        int n_rep = this->linrep.size();
        int n_arms = this->linrep[0]->n_arms();

        double max_avalue;
        int action;
        for (int a = 0; a < n_arms; ++a)
        {
            // VectorXd ucb = 1e-15 * VectorXd::Zero(n_rep);
            double min_ucb_rep = std::numeric_limits<double>::max();
            for (int i : active_reps)
            {
                if (a==0) {this->oful_algs[i].action(context);} // to compute UCB index
                min_ucb_rep = min(this->oful_algs[i].UCBindex[a], min_ucb_rep);
            }

            if (a == 0 || max_avalue < min_ucb_rep)
            {
                action = a;
                max_avalue = min_ucb_rep;
            }
        }
        this->t++;
        return action;
    }

    void update(const X& context, int action, double reward)
    {
        // #pragma parallel for
        for (int i = 0, ii = this->linrep.size(); i < ii; ++i)
        {
            this->oful_algs[i].update(context, action, reward);
        }

        context_history.push_back(context);
        action_history.push_back(action);
        reward_history.push_back(reward);


        if (this->t % 10 == 0) {
            int M = this->oful_algs.size();
            std::vector<double> mse(M, -9);
            double min_mse = std::numeric_limits<double>::max();
            for(int i : active_reps)
            {
                OFUL<X>& model = this->oful_algs[i];
                VectorXd theta = model.inv_A * model.b_vec;
                mse[i] = 0.;
                int hlen = context_history.size();
                for (int k=0; k < hlen; ++k) {
                    auto v = this->linrep[i]->get_features(context_history[k], action_history[k]);
                    double yhat = theta.dot(v);
                    mse[i] += (yhat - reward_history[k])*(yhat - reward_history[k]);
                }
                mse[i] /= hlen;
                min_mse = min(min_mse, mse[i]);
            }           

            std::vector<int> new_active_reps;
            for(int i : active_reps)
            {
                double L = this->oful_algs[i].features_bound;
                double S = this->oful_algs[i].param_bound;
                double d = this->oful_algs[i].linrep.features_dim();
                double offset = (d *log((M * S * this->t)/(this->delta)) + 4 * L + this->reg_val * S *S) / this->t;
                // if (this->t % 100 == 0) {
                //     std::cout << i << " "<<mse[i] << " " << min_mse  << " " <<  offset << std::endl;
                // }
                if(mse[i] <= min_mse + offset) {
                    new_active_reps.push_back(i);
                }
            }
            if (active_reps.size() != new_active_reps.size()){
                for (auto& i : active_reps) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
            }
            active_reps = new_active_reps;
            // std::cout << std::endl;
        }
    }

public:
    std::vector<X> context_history;
    std::vector<int> action_history;
    std::vector<double> reward_history;
    std::vector<int> active_reps;
};
#endif

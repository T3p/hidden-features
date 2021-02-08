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

        int M = this->oful_algs.size();
        // context_history.reserve(10000);
        // action_history.reserve(10000);
        // reward_history.reserve(10000);
        active_reps.resize(M);
        std::iota(active_reps.begin(), active_reps.end(), 0);

        std::cout << "active_set: \n";
        for (auto& i : active_reps)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl << "--------\n";

        SSE_A.resize(M);
        SSE_B.resize(M);
        SSE_C.resize(M);
        SSE_D.resize(M);
        for(int i : active_reps)
        {
            OFUL<X>& model = this->oful_algs[i];
            double d = model.linrep.features_dim();
            SSE_A[i] = 0;
            SSE_B[i] = VectorXd::Zero(d);
            SSE_C[i] = VectorXd::Zero(d);
            int k = (d * (d-1)) / 2;
            SSE_D[i] = VectorXd::Zero(k);
        }
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
                if (a==0)
                {
                    this->oful_algs[i].action(context);   // to compute UCB index
                }
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
        for(int i : active_reps)
        {
            this->oful_algs[i].update(context, action, reward);
            int d = this->linrep[i]->features_dim();
            VectorXd x = this->linrep[i]->get_features(context, action);

            SSE_A[i] += reward * reward;
            int idx = 0;
            for (int k = 0; k < d; ++k)
            {
                SSE_B[i][k] -= 2*reward * x[k];
                SSE_C[i][k] += x[k] * x[k];
                for(int j = k+1; j < d; ++j)
                {
                    SSE_D[i][idx] += 2*x[k] * x[j];
                    idx++;
                }
            }
        }

        // context_history.push_back(context);
        // action_history.push_back(action);
        // reward_history.push_back(reward);


        /*
        SSE = \sum_i (y_i - x_i w)^2
            = \sum_i y_i^2 + (x_i w)^2  - 2y_i x_i w
            = \sum_i y_i^2 - 2y_i x_i w
              + \sum_i [ \sum_j (x_{ij}^2 w_j^2) + 2 \sum_{k}\sum_{j=k+1} x_{ik} x_{ij} w_{ik} w_{ij} ]
        Let's
            A = \sum_i y_i^2
            B = \sum_i -2y_i x_i
            C = \sum_i x_{ij}^2  (dim d)
            D = \sum_i 2 x_{ik} x_{ij} (dim d*(d-1)/2)
        Then, for any w
        SSE = A + Bw + \sum_j C_j w_j \sum_{k}\sum_{j=k+1} D_k D_j w_i w_j
        MSE = SSE / t
        */

        if (this->t % 1 == 0)
        {
            int M = this->oful_algs.size();
            std::vector<double> mse(M, -9);
            double min_val = std::numeric_limits<double>::max();
            for(int i : active_reps)
            {
                OFUL<X>& model = this->oful_algs[i];
                double L = model.features_bound;
                double S = model.param_bound;
                int d = model.linrep.features_dim();
                int tt = this->t -1;
                // assert (hlen == context_history.size());
                double offset = (24 * log(2 * tt) + 8* log(M * M / this->delta) + 8 * d * log(12 * L * S * tt) + 1) / tt;

                VectorXd theta = model.inv_A * model.b_vec;

                double sse = SSE_A[i];
                int idx = 0;
                for (int k = 0; k < d; ++k)
                {
                    sse += (SSE_B[i][k] + SSE_C[i][k]*theta[k]) * theta[k];
                    for(int j = k+1; j < d; ++j)
                    {
                        sse += SSE_D[i][idx] * theta[k] * theta[j];
                        idx++;
                    }
                }

                // mse[i] = 0.;
                // for (int k=0; k < tt; ++k) {
                //     auto v = this->linrep[i]->get_features(context_history[k], action_history[k]);
                //     double yhat = theta.dot(v);
                //     mse[i] += (yhat - reward_history[k])*(yhat - reward_history[k]);
                // }
                // std::cout << "i: " << i <<" " << mse[i] << "  " << sse << endl;
                // assert(fabs(mse[i] - sse) <= 1e-4);
                mse[i] = sse / tt;
                // std::cout << min_val << " " << mse[i] << " " << offset << std::endl;
                min_val = min(min_val, mse[i] + offset);
            }

            std::vector<int> new_active_reps;
            for(int i : active_reps)
            {
                if (mse[i] <= min_val)
                {
                    new_active_reps.push_back(i);
                }
                else
                {
                    std::cout << "t" << this->t << ": elim " << i << " since "<< mse[i] << " > " << min_val << std::endl;
                }
            }
            if (new_active_reps.size() != active_reps.size())
            {
                std::cout << "new active set: [ ";
                for(auto& u : new_active_reps)
                {
                    std::cout << u << " ";
                }
                std::cout << " ]\n" << std::endl;
            }
            active_reps = new_active_reps;
        }
    }

public:
    // std::vector<X> context_history;
    // std::vector<int> action_history;
    // std::vector<double> reward_history;
    std::vector<int> active_reps;

    std::vector<double> SSE_A;
    std::vector<VectorXd> SSE_B, SSE_C, SSE_D;
};
#endif

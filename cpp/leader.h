#ifndef MMOFUL_H
#define MMOFUL_H

#include <Eigen/Dense>
#include "abstractclasses.h"
#include "oful.h"

using namespace std;
using namespace Eigen;

template<typename X>
class LEADER : public Algo<X>
{
public:

    LEADER(
        std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep,
        double reg_val, double noise_std,
        double bonus_scale=1., double delta=0.01, bool adaptive_ci=true
    )
        : Algo<X>("LEADER"), linrep(linrep),
          reg_val(reg_val), noise_std(noise_std),
          bonus_scale(bonus_scale), delta(delta), adaptive_ci(adaptive_ci)
    {
        for (int i = 0, ii = linrep.size(); i < ii; ++i)
        {
            oful_algs.push_back(OFUL<X>(*linrep[i], reg_val, noise_std, bonus_scale, delta, adaptive_ci));
        }
        reset();
    }
    ~LEADER() {}

    void reset()
    {
        int n_arms = linrep[0]->n_arms();
        int nreps = linrep.size();
        assert(nreps > 0);
        for (int i = 0; i < nreps; ++i)
        {
            oful_algs[i].reset();
            assert(n_arms == linrep[i]->n_arms());
        }
        t = 1;
    }

    int action(const X& context)
    {
        // cout << t << ", " << context << endl;
        int n_rep = linrep.size();
        int n_arms = linrep[0]->n_arms();

        double max_avalue;
        int action;
        for (int a = 0; a < n_arms; ++a)
        {
            // VectorXd ucb = 1e-15 * VectorXd::Zero(n_rep);
            double min_ucb_rep = std::numeric_limits<double>::max();
            for (int i = 0; i < n_rep; ++i)
            {
                if (a==0) {oful_algs[i].action(context);} // to compute UCB index
                min_ucb_rep = min(oful_algs[i].UCBindex[a], min_ucb_rep);
            }

            if (a == 0 || max_avalue < min_ucb_rep)
            {
                action = a;
                max_avalue = min_ucb_rep;
            }
        }
        t++;
        return action;
    }

    std::vector<double> action_distribution(const X& context) {
        int n_arms = linrep[0]->n_arms();
        std::vector<double> proba(n_arms);
        proba[action(context)] = 1;
        return proba;
    }

    void update(const X& context, int action, double reward)
    {
        // #pragma parallel for
        for (int i = 0, ii = linrep.size(); i < ii; ++i)
        {
            oful_algs[i].update(context, action, reward);
        }
    }

public:
    std::vector<std::shared_ptr<ContRepresentation<X>>>& linrep;
    std::vector<OFUL<X>> oful_algs;
    double reg_val, noise_std, bonus_scale, delta;
    bool adaptive_ci;
    std::vector<MatrixXd> inv_A;
    std::vector<VectorXd> b_vec;
    std::vector<double> features_bound;
    std::vector<double> param_bound;
    int t;
};
#endif

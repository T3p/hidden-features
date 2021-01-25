#ifndef OFULCORRAL_H
#define OFULCORRAL_H

#include <Eigen/Dense>
#include <algorithm>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

#include "abstractclasses.h"
#include "oful.h"
#include "utils.h"

using namespace std;
using namespace Eigen;


/**
 * Implementation of CORRAL
 *
 * Alekh Agarwal, Haipeng Luo, Behnam Neyshabur, Robert E. Schapire:
 * Corralling a Band of Bandit Algorithms
 * COLT 2017
 * http://proceedings.mlr.press/v65/agarwal17b.html
 *
 * This is a specific implementation for selecting the representation of
 * OFUL algorithm.
 */

std::vector<double> log_barrier_omd(
    std::vector<double>& p,
    std::vector<double>& loss,
    std::vector<double>& lr,
    double EPSILON = 1e-6)
{
    auto result = std::minmax_element(loss.begin(), loss.end());
    double minx = *result.first; // min
    double maxx = *result.second; // max

    auto func = [p,loss,lr](double x) -> double
    {
        double val = 0;
        for(int i = 0, ii=p.size(); i < ii; ++i)
        {
            val += 1./(1./p[i] + lr[i]*(loss[i] - x));
        }
        return (1-val)*(1-val);
    };

    int bits = std::numeric_limits<double>::digits;
    auto x = boost::math::tools::brent_find_minima(
                 func,
                 minx,
                 maxx,
                 bits
             );
    // std::cout << "x at minimum = " << x.first << ", f(" << x.first << ") = " << x.second << std::endl;
    double lambda = x.first;

    // compute new p
    std::vector<double> newp(p.size());
    double normfact = 0;
    for(int i = 0, ii=p.size(); i < ii; ++i)
    {
        newp[i] = 1./(1./p[i] + lr[i]*(loss[i] - lambda));
        normfact += newp[i];
    }
    double tot = 0, minv=std::numeric_limits<double>::max();
    for(int i = 0, ii=p.size(); i < ii; ++i)
    {
        newp[i] /= normfact;
        tot += newp[i];
        minv = min(minv, newp[i]);
    }
    assert(abs(tot-1)<1e-6);
    // cout << "\nnewp: ";
    // print_vector(newp);

    // assert(minv > 0);
    if (minv < 0) {
        cout << "!!!!!probabilities are negative, let's try to fix it" << endl;

        normfact = 0;
        double newmin = std::numeric_limits<double>::max();
        for(int i = 0, ii=p.size(); i < ii; ++i) {
            newp[i] = 1 + newp[i] / abs(minv);
            normfact += newp[i];
            newmin = min(newmin, newp[i]);
        }
        assert(abs(newmin) < 1e-6);

        tot = 0;
        for(int i = 0, ii=p.size(); i < ii; ++i) {
            newp[i] /= normfact;
            assert(newp[i] >= 0);
            tot += newp[i];
        }
        assert(abs(tot-1) < 1e-6);


        print_vector(newp);
    }

    return newp;
}

template<typename X>
class Corral : public Algo<X>
{
public:

    Corral(
        std::vector<std::shared_ptr<Algo<X>>>& base_algs,
        double gamma, double beta, double lr0, int seed = 0
    ) :
        Algo<X>("OFULCorral"), base_algs(base_algs),
        gamma(gamma), beta(beta), lr0(lr0), seed(seed)
    {
        reset();
        rng.seed(seed);
    }

    ~Corral() {}

    void reset()
    {
        t = 0;
        // reset all base algs
        for(auto ba : base_algs)
        {
            ba->reset();
        }

        int nreps = base_algs.size();
        //learning rate
        lr.resize(nreps);
        std::fill(lr.begin(), lr.end(), lr0);
        //probabilities
        proba.resize(nreps);
        std::fill(proba.begin(), proba.end(), 1./nreps);
        lower_bound_p.resize(nreps);
        std::fill(lower_bound_p.begin(), lower_bound_p.end(), 0.5/nreps);
        // threshold
        threshold.resize(nreps);
        std::fill(threshold.begin(), threshold.end(), 2*nreps);
    }

    int action(const X& context)
    {
        // Sample model
        std::discrete_distribution<> d(proba.begin(), proba.end());
        this->last_selected_rep = d(rng);

        double action = base_algs[this->last_selected_rep]->action(context);
        t++;
        return action;
    }

    std::vector<double> action_distribution(const X& context) {
        std::cout << "not implemented" << std::endl;
        return std::vector<double>();
    }

    void update(const X& context, int action, double reward)
    {
        // update all the representation
        int nreps = base_algs.size();
        std::vector<double> mod_rewards(nreps);
        for (int i=0; i < nreps; ++i)
        {
            mod_rewards[i] = reward * (this->last_selected_rep == i) / proba[this->last_selected_rep];
            base_algs[i]->update(context, action, mod_rewards[i]);
        }
        std::vector<double> ptp1 = log_barrier_omd(proba, mod_rewards, lr);
        for (int i=0; i < nreps; ++i)
        {
            // update smoothed proba
            proba[i] = (1-gamma) * ptp1[i] + gamma/nreps;
            // update threshold and learning rate
            if(lower_bound_p[i] > proba[i])
            {
                lower_bound_p[i] = proba[i] / 2.;
                lr[i] = beta * lr[i];
            }
            threshold[i] = 1. / lower_bound_p[i];
        }
    }

public:
    int seed, last_selected_rep, t;
    double gamma, beta, lr0;
    std::vector<double> lr;
    std::vector<double> proba;
    std::vector<double> lower_bound_p;
    std::vector<double> threshold;
    std::mt19937 rng;
    std::vector<std::shared_ptr<Algo<X>>> base_algs;
};

#endif

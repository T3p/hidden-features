#ifndef ADVERSARIAL_MASTER_H
#define ADVERSARIAL_MASTER

#include <memory>
#include <random>
#include <numeric>
#include <limits>
#include <boost/math/tools/minima.hpp>
#include <boost/math/tools/roots.hpp>


#include "abstractclasses.h"
#include "utils.h"

const double EPSILON {1e-6};

using namespace std;

/**
 * Implementation of bandit model selection algorithms from
 *
 * Aldo Pacchiano, My Phan, Yasin Abbasi-Yadkori, Anup Rao,
   Julian Zimmert, Tor Lattimore, Csaba Szepesvari:
 * Model Selection in Contextual Stochastic Bandit Problems
 *
 * https://arxiv.org/abs/2003.01704
 */

//Smoothing wrapper
template<typename X>
class SmoothedAlgo: public Algo<X>
{
public:
    SmoothedAlgo(shared_ptr<BaseAlgo<X>> base_alg, long seed=0): Algo<X>("Smoothed" + base_alg->name),
                                        base_alg(base_alg)
    {
        rng.seed(seed);
        reset();
    }

    void reset()
    {
        pending = false;
        base_alg->reset();
        old_policies.clear();
        old_policies.push_back(base_alg->clone());
    }

    int action(const X& context)
    {
        if(!pending)
        {
            //std::cerr << "Step 1\n";
            pending = true;
            return base_alg->action(context);

        }
        else
        {
            //std::cerr << "Step 2\n";
            uniform_int_distribution<int> uid(0, old_policies.size()-1);
            int q = uid(rng);
            pending = false;
            old_policies[q];
            /*Check that state of copies is preserved
            for (auto& ptr : old_policies)
            {
                std::cerr << ptr->action(context);
            }
            std::cerr << std::endl;
            */
            return old_policies[q]->action(context);
        }
    }

    void update(const X& context, int action, double reward)
    {
        base_alg->update(context, action, reward);
        old_policies.push_back(base_alg->clone());
    }

    std::vector<double> action_distribution(const X& context) {}

    double upper_bound()
    {
        return base_alg->upper_bound();
    }

    shared_ptr<BaseAlgo<X>> base()
    {
        return base_alg;
    }

private:
    shared_ptr<BaseAlgo<X>> base_alg;
    vector<unique_ptr<Algo<X>>> old_policies;
    bool pending;
    std::mt19937 rng;
};


//EXP3.P for smoothed base
template<typename X>
class EXP3dotP: public Algo<X>
{
public:
    EXP3dotP(vector<shared_ptr<SmoothedAlgo<X>>>& base_algs, double exprate, long seed=0)
    : Algo<X>("EXP3.P"),
        base_algs(base_algs), exprate(exprate), nbases(base_algs.size())
    {
        rng.seed(seed);
        probs.resize(nbases);
        cum_gains.resize(nbases);
        reset();
    }

    int action(const X& context)
    {
        if(!pending)
        {
            //Step 1
            //print_vector(probs);
            assert(is_distr(probs, EPSILON));

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            selection = dist(rng);
            return base_algs[selection]->action(context);
            pending = true;
        }
        else
        {
            //Step 2
            return base_algs[selection]->action(context);
            pending = false;
        }
    }

    void update(const X& context, int action, double reward)
    {
        //No update AFTER step 1
        if(!pending)
        {
            //AFTER Step 2
            //same reward for TWO steps
            for(int i=0; i<nbases; ++i)
            {
                double gain = (i==selection) ? reward : 0.;
                cum_gains[i] += 2. * (gain + exprate/2.) / (probs[i] + EPSILON);
            }

            //update probs
            double stabilizer = std::numeric_limits<double>::lowest();
            for(int i=0; i<nbases; ++i)
            {
                stabilizer = max(stabilizer, cum_gains[i]);
            }

            double  normalization = 0.;
            for(int i=0; i<nbases; ++i)
            {
                normalization += exp(cum_gains[i] - stabilizer);
            }

            for(int i=0; i<nbases; ++i)
            {
                //same reward for the two steps
                probs[i] = (1-exprate)*exp(cum_gains[i]-stabilizer)/normalization + exprate/nbases;
            }

            //update base
            base_algs[selection]->update(context, action, reward);

            pending = false;
        }
    }

    std::vector<double> action_distribution(const X& context) {}

    void reset()
    {
        pending = false;
        for(int i=0; i<nbases; ++i)
        {
            base_algs[i]->reset();
            probs[i] = 1. / nbases;
            cum_gains[i] = 0.;
        }
    }

private:
    vector<shared_ptr<SmoothedAlgo<X>>> base_algs;
    double exprate;
    int nbases;
    vector<double> probs;
    bool pending;
    int selection;
    std::mt19937 rng;
    vector<double> cum_gains;
};


//log-barrier Online Mirror Descent update
vector<double> log_barrier_OMD(const vector<double>& probs, const vector<double>& losses, const vector<double>& lrs)
{
    int M = probs.size();
    assert(losses.size()==M);
    assert(lrs.size()==M);
    assert(is_distr(probs, EPSILON));

    //Compute lambda
    auto boundaries = std::minmax_element(losses.begin(), losses.end());
    double xmin = *boundaries.first;
    double xmax = *boundaries.second;
    auto normalizer = [&](double x) -> double
    {
        double val = 0.;
        for(int i = 0; i < M; ++i)
        {
            if(probs[i] > 0)
            {
                val += 1./(1./(probs[i]) + lrs[i]*(losses[i] - x));
            }
        }
        return val;
    };

    auto func = [&](double x) -> double
    {
        return (1-normalizer(x))*(1-normalizer(x));
    };

    int bits = numeric_limits<double>::digits;
    auto opt = boost::math::tools::brent_find_minima(
                 func,
                 xmin,
                 xmax,
                 bits
             );
    double lambda = opt.first;
    //assert(abs(normalizer(lambda)-1.)<EPSILON);

    //Compute new probs
    vector<double> newprobs(M);
    for(int i=0; i<M; ++i)
    {
        newprobs[i] = probs[i]>0? (1. / (1./probs[i] + lrs[i]*(losses[i]-lambda))) : 0;
    }

    if(is_distr(newprobs, EPSILON)){
        return newprobs;
    }

    //fix probabilities
    std::cerr << "Fixing probabilities" << std::endl;
    double minprob = *std::min_element(newprobs.begin(), newprobs.end());
    if(minprob<0)
    {
        for(int i=0; i<M; ++i)
        {
            newprobs[i] = 1 + newprobs[i] / abs(minprob);
        }
    }
    double normalization = std::accumulate(newprobs.begin(), newprobs.end(), 0.);
    for(int i=0; i<M; ++i)
    {
        newprobs[i] /= normalization;
    }
    assert(is_distr(newprobs, EPSILON));
    return newprobs;

}


//Corral for smoothed base
template<typename X>
class Corral: public Algo<X>
{
public:
    Corral(vector<shared_ptr<SmoothedAlgo<X>>>& base_algs, double lr_init, int horizon, long seed=0)
    : Algo<X>("Corral"),
        base_algs(base_algs), lr_init(lr_init), nbases(base_algs.size()),
        gamma(1./horizon), beta(exp(1./log(horizon)))
    {
        rng.seed(seed);
        probs.resize(nbases);
        inf_probs.resize(nbases);
        gains.resize(nbases);
        lrs.resize(nbases);
        reset();
    }

    int action(const X& context)
    {
        if(!pending)
        {
            //Step 1
            //print_vector(probs);
            assert(is_distr(probs, EPSILON));

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            selection = dist(rng);
            return base_algs[selection]->action(context);
            pending = true;
        }
        else
        {
            //Step 2
            return base_algs[selection]->action(context);
            pending = false;
        }
    }

    void update(const X& context, int action, double reward)
    {
        //No update AFTER step 1
        if(!pending)
        {
            //AFTER Step 2
            //same gain for TWO steps
            for(int i=0; i<nbases; ++i)
            {
                gains[i] = (i==selection) ? 2*reward/(probs[i]+EPSILON) : 0.;
            }

            //update probs
            probs = log_barrier_OMD(probs, gains, lrs);

            //compute final probs and update learning rates
            for(int i=0; i<nbases; ++i)
            {
                probs[i] = (1.-gamma)*probs[i] + gamma/nbases;
                if(inf_probs[i] > probs[i])
                {
                    inf_probs[i] = probs[i]/2;
                    lrs[i] = lrs[i]*beta;
                }
            }

            //update base
            base_algs[selection]->update(context, action, reward);

            pending = false;
        }
    }

    std::vector<double> action_distribution(const X& context) {}

    void reset()
    {
        pending = false;
        for(int i=0; i<nbases; ++i)
        {
            base_algs[i]->reset();
            probs[i] = 1./nbases;
            inf_probs[i] = 1./(2.*nbases);
            lrs[i] = lr_init;
        }
    }

private:
    vector<shared_ptr<SmoothedAlgo<X>>> base_algs;
    double lr_init;
    double gamma;
    double beta;
    int nbases;
    vector<double> probs;
    bool pending;
    int selection;
    std::mt19937 rng;
    vector<double> inf_probs;
    vector<double> gains;
    vector<double> lrs;
};


//EXP4.IX with changing experts (heuristic)
template<typename X>
class EXP4dotIX: public Algo<X>
{
public:
    EXP4dotIX(vector<shared_ptr<Algo<X>>>& base_algs, double lr, double gamma, long seed=0)
        :Algo<X>("EXP4.IX"), base_algs(base_algs), lr(lr), gamma(gamma), nbases(base_algs.size())
    {
        rng.seed(seed);
        expert_weights.resize(nbases);
        cum_losses.resize(nbases);
        advice.resize(nbases);
        reset();
    }

    int action(const X& context)
    {
        vector<double> probs = action_distribution(context);
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }

    void update(const X& context, int action, double reward)
    {
        //arm losses
        int narms = advice[0].size();
        vector<double> arm_losses(narms);
        for(int i=0; i<narms; ++i)
        {
            //Implicit eXploration (IX)
            arm_losses[i] = (i==action)? -reward/(arm_probs[i]+gamma) : 0.;
        }

        //update cumulative expert losses
        double normalization = 0.;
        vector<double> expert_scores(nbases);
        double stabilizer = std::numeric_limits<double>::lowest();;
        for(int i=0; i<nbases; ++i)
        {
            for(int j=0; j<narms; ++j)
            {
                cum_losses[i] += advice[i][j] * arm_losses[i];
            }
            expert_scores[i] = -lr * cum_losses[i];
            stabilizer = max(stabilizer, expert_scores[i]);
        }

        //update expert weights
        for(int i=0; i<nbases; ++i)
        {
            expert_weights[i] = exp(-lr * cum_losses[i] - stabilizer);
            normalization += expert_weights[i];
        }
        for(int i=0; i<nbases; ++i)
        {
            expert_weights[i] /= normalization;
        }
        assert(is_distr(expert_weights));

        //update experts (NOT COVERED BY EXP4 THEORY)
        for(auto& expert : base_algs)
        {
            expert->update(context, action, reward);
        }
    }

    vector<double> action_distribution(const X& context) {
        for(int i=0; i<nbases; ++i)
        {
            advice[i] = (base_algs[i]->action_distribution(context));
        }

        int narms = advice[0].size();
        arm_probs.resize(narms);
        double normalization = 0.;
        for(int i=0; i<narms; ++i)
        {
            arm_probs[i] = 0.;
            for(int j=0; j<nbases; ++j)
            {
                arm_probs[i] += expert_weights[j] * advice[j][i];
            }
            normalization += arm_probs[i];
        }
        for(int i=0; i<narms; ++i)
        {
            arm_probs[i] /= normalization;
        }

        assert(is_distr(arm_probs));
        return arm_probs;
    }

    void reset()
    {
        for(int i=0; i<nbases; ++i)
        {
            cum_losses[i] = 0.;
            expert_weights[i] = 1. / nbases;
        }
    }

private:
    vector<shared_ptr<Algo<X>>> base_algs;
    double lr;
    double gamma;
    int nbases;
    vector<vector<double>> advice;
    vector<double> arm_probs;
    vector<double> cum_losses;
    vector<double> expert_weights;
    std::mt19937 rng;
};


#endif

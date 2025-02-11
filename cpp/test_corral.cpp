#include <iostream>
#include <vector>
#include <random>

#include "adversarial_master.h"
#include "oful.h"
#include "finitelinrep.h"
#include "abstractclasses.h"
#include "bandit.h"
#include "utils.h"

int main()
{
    int n_contexts = 100;
    int n_arms = 5;
    int dim = 10;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.3;
    double bonus_scale = 1.;
    bool adaptive_ci = true;
    int horizon = 100;

    FiniteLinearRepresentation rep1 = make_random(n_contexts, n_arms, dim);
    FiniteLinearRepresentation rep2 = make_random(n_contexts, n_arms, dim);
    shared_ptr<OFUL<int>> oful1 = make_shared<OFUL<int>>(rep1, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
    shared_ptr<OFUL<int>> oful2 = make_shared<OFUL<int>>(rep2, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
    shared_ptr<SmoothedAlgo<int>> smoful1 = make_shared<SmoothedAlgo<int>>(oful1);
    shared_ptr<SmoothedAlgo<int>> smoful2 = make_shared<SmoothedAlgo<int>>(oful2);
    vector<shared_ptr<SmoothedAlgo<int>>> bases {smoful1, smoful2};
    vector<shared_ptr<Algo<int>>> raw_bases {oful1, oful2};

    //Test the smoothing
    ContBanditProblem<int> prb(rep1, *smoful1);
    prb.reset();
    prb.run(horizon);
    //print_vector(prb.exp_instant_regret);

    //Test EXP3.P
    EXP3dotP<int> exp3(bases, oful1->exp3_rate(horizon, 2), 0, true);
    ContBanditProblem<int> metaprb1(rep1, exp3);
    metaprb1.reset();
    metaprb1.run(horizon);
    //print_vector(metaprb.exp_instant_regret);

    //Test OMD
    vector<double> probs = {0.1, 0.2, 0.7};
    vector<double> losses = {1., 2., 3.};
    vector<double> lrs = {0.1, 0.2, 0.3};
    log_barrier_OMD(probs, losses, lrs);

    //Test Corral
    Corral<int> corr(bases, oful1->corral_lr(horizon, 2), horizon, 0, true);
    ContBanditProblem<int> metaprb2(rep1, corr);
    metaprb2.reset();
    metaprb2.run(horizon);

    //Test EXP4.IX
    double exp4_gamma = sqrt(2*log(raw_bases.size())/(n_arms*horizon));
    double exp4_lr = 2*exp4_gamma;
    EXP4dotIX<int> exp4(raw_bases, exp4_lr, exp4_gamma);
    ContBanditProblem<int> metaprb3(rep1, exp4);
    metaprb3.reset();
    metaprb3.run(horizon);

}

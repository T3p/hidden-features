// standard includes and external libraries
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <ctime>
// project includes
#include "abstractclasses.h"
#include "bandit.h"
#include "oful.h"
#include "leader.h"
#include "regbalancing.h"
#include "adversarial_master.h"
#include "finitelinrep.h"
#include "utils.h"
#include "gzip.h"

using json = nlohmann::json;
using namespace std;
using namespace Eigen;

size_t PREC = 4;   // for saving numbers are rounded to PREC decimals
size_t EVERY = 1;  // save EVERY round

int main()
{
    std::time_t t = std::time(nullptr);
    char MY_TIME[100];
    std::strftime(MY_TIME, sizeof(MY_TIME), "%Y%m%d%H%M%S", std::localtime(&t));
    std::cout << MY_TIME << '\n';

    typedef std::vector<std::vector<double>> vec2double;

    int seed = time(NULL);
    // seed= 1611835201;
    seed=1611836268;
    srand (seed);
    cout << "seed: " << seed << endl;
    // rng.seed(10000); // warm it up
    int n_runs = 5, T = 10000;
    double delta = 0.01;
    double reg_val = 1.;
    double noise_std = 0.3;
    double bonus_scale = 1.;
    bool adaptive_ci = true;
    //double cor_gamma = 1./T, cor_beta = exp(1./log(T)), cor_lr0 = 20. / sqrt(T);

    std::vector<long> seeds(n_runs);
    std::generate(seeds.begin(), seeds.end(), [] ()
    {
        return rand();
    });

    // FiniteLinearRepresentation rep = make_random(20, 5, 6, true, noise_std, seed);
    // rep.save("linrep.json"); // save current model

    FiniteLinearRepresentation rep = flr_loadjson("linrep3.json", noise_std, seed);
    int dim = rep.features_dim();
    cout << "Dimension: " << dim << endl;

    std::vector<FiniteLinearRepresentation> reps;
    double MMM = rep.features_bound();
    for(int i = 1; i < dim; ++i)
    {
        FiniteLinearRepresentation rr = derank_hls(rep, i, false, true, true);
        // cout << i << ": " << rr.features_bound() << ", " << rr.param_bound() << std::endl;
        rr.normalize_features(MMM);
        // cout << i << ": " << rr.features_bound() << ", " << rr.param_bound() << std::endl;
        reps.push_back(rr);
        bool flag = rr.is_equal(rep);
        if (!flag) {
            std::cout << "Error: non realizable representation" << std::endl;
            exit(1);
        }
    }
    assert(rep.is_equal(rep));
    reps.push_back(rep);
    // cout << dim << ": " << rep.features_bound() << ", " << rep.param_bound() << std::endl;
    assert(reps.size() == dim);


    // //LEADER
    // vec2double regrets(n_runs), pseudo_regrets(n_runs);
    // #pragma omp parallel for
    // for (int i = 0; i < n_runs; ++i)
    // {
    //     std::vector<std::shared_ptr<ContRepresentation<int>>> lreps;
    //     for(auto& ll : reps)
    //     {
    //         auto tmp = std::make_shared<FiniteLinearRepresentation>(ll.copy(seeds[i]));
    //         lreps.push_back(tmp);
    //     }
    //     LEADER<int> localg(lreps, reg_val, noise_std, bonus_scale, delta/lreps.size(), adaptive_ci);
    //     ContBanditProblem<int> prb(*lreps[0], localg);
    //     prb.reset();
    //     prb.run(T);
    //     regrets[i] = prb.instant_regret;
    //     pseudo_regrets[i] = prb.exp_instant_regret;
    //     // delete localg;
    // }
    // // save_vector_csv(regrets, "LEADER_regrets.csv", EVERY, PREC);
    // save_vector_csv_gzip(regrets, "LEADER_regrets.csv.gz", EVERY, PREC);
    // // save_vector_csv(pseudo_regrets, "LEADER_pseudoregrets.csv", EVERY, PREC);
    // save_vector_csv_gzip(pseudo_regrets, "LEADER_pseudoregrets.csv.gz", EVERY, PREC);


    //just OFUL
    for(int j = 0; j < reps.size(); ++j)
    {
        vec2double regrets(n_runs), pseudo_regrets(n_runs);

        #pragma omp parallel for
        for (int i = 0; i < n_runs; ++i)
        {
            FiniteLinearRepresentation lrep = reps[j].copy(seeds[i]);
            OFUL<int> localg(lrep, reg_val, noise_std, bonus_scale, delta, adaptive_ci);
            ContBanditProblem<int> prb(lrep, localg);
            prb.reset();
            prb.run(T);
            regrets[i] = prb.instant_regret;
            pseudo_regrets[i] = prb.exp_instant_regret;
        }
        // save_vector_csv(regrets, "OFUL-rep"+std::to_string(j)+"_regrets.csv", EVERY, PREC);
        save_vector_csv_gzip(regrets, "OFUL-rep"+std::to_string(j)+"_regrets.csv.gz", EVERY, PREC);
        // save_vector_csv(pseudo_regrets, "OFUL-rep"+std::to_string(j)+"_pseudoregrets.csv", EVERY, PREC);
        save_vector_csv_gzip(pseudo_regrets, "OFUL-rep"+std::to_string(j)+"_pseudoregrets.csv.gz", EVERY, PREC);

#if 0
        #pragma omp parallel for
        for (int i = 0; i < n_runs; ++i)
        {
            std::vector<FiniteLinearRepresentation> llr{reps[j].copy(seeds[i])};
            LEADER ddd(llr, reg_val, 1, delta/llr.size(), adaptive_ci);
            LinBanditProblem prb(llr[0], ddd);
            prb.run(T);
            regrets[i] = prb.instant_regret;
            pseudo_regrets[i] = prb.exp_instant_regret;
        }
        save_vector_csv(regrets, "LEADER-rep"+std::to_string(j)+"_regrets.txt");
        save_vector_csv(pseudo_regrets, "LEADER-rep"+std::to_string(j)+"_pseudoregrets.txt");
#endif
    }
    return 0;
}

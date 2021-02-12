
#ifndef LINEARREP_H
#define LINEARREP_H

#include <vector>
#include "abstractclasses.h"


class ContToy1_phi1 : public ContRepresentation<std::vector<double>>
{
public:

    ContToy1_phi1(double noise_std, int seed=0)
        : noise_std(noise_std), seed(seed), feature_bound(1),
          actions({{0,0}, {0,1}, {1,0}, {1,1}})
    {
        param = VectorXd::Ones(2);
        nd = normal_distribution<double>(0,1.);
        urd = std::uniform_real_distribution<double>(0,1.);
        rng.seed(seed);

        feats = [](const std::vector<double>& context, int action){
            std::vector<std::vector<double>> localactions{{0,0}, {0,1}, {1,0}, {1,1}};
            Eigen::VectorXd f(2);
            std::vector<double>& a = localactions[action];
            f(0) = context[0] * a[0];
            f(1) = context[1] * a[1];
            return f;
        };
    }

    ContToy1_phi1(const ContToy1_phi1& other)
        : ContToy1_phi1(other.noise_std, other.seed)
    {}

    ~ContToy1_phi1() {}

    ContToy1_phi1 copy(int new_seed)
    {
        return ContToy1_phi1(noise_std, new_seed);
    }

    int n_arms()
    {
        return actions.size();
    }
    int features_dim()
    {
        return param.size();
    }
    double features_bound()
    {
        return feature_bound;
    }
    double param_bound()
    {
        return param.lpNorm<2>();
    }

    std::vector<double> sample_context()
    {
        double x=2, y=2;
        while(y > 0)
        {
            x = nd(rng);
            y = nd(rng);
            double norm = sqrt(x*x + y*y);
            double r = sqrt(urd(rng));
            x = r * x / norm;
            y = r * y / norm;
        }
        return std::vector<double> {x,y};
    }

    double sample_reward(const std::vector<double>& context, int action)
    {
        double r = reward(context, action) + nd(rng) * noise_std;
        return r;
    }

    Eigen::VectorXd get_features(const std::vector<double>& context, int action)
    {
        return feats(context, action);
    }

    double optimal_reward(const std::vector<double>& context)
    {
        double max_rew = reward(context, 0);
        for (size_t i=1; i < actions.size(); ++i)
        {
            max_rew = max(max_rew, reward(context, i));
        }
        return max_rew;
    }

    double reward(const std::vector<double>& context, int action)
    {
        return get_features(context, action).dot(param);
    }

protected:
    double noise_std, feature_bound;
    int seed;
    VectorXd param;
    std::vector<std::vector<double>> actions;
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> urd;
    std::mt19937 rng;
    std::function<Eigen::VectorXd(const std::vector<double>&, int)> feats;
};

class ContToy1_phi2 : public ContToy1_phi1
{
public:
    ContToy1_phi2(double noise_std, int seed=0)
        :  ContToy1_phi1(noise_std, seed)
    {
        feature_bound = 1 + sqrt(2);
        param = VectorXd::Ones(3);

        feats = [](const std::vector<double>& context, int action){
            Eigen::VectorXd f(3);
            std::vector<std::vector<double>> localactions{{0,0}, {0,1}, {1,0}, {1,1}};
            std::vector<double>& a = localactions[action];
            f(0) = context[0] * a[0] - context[0];
            f(1) = context[1] * a[1] - context[1];
            f(2) = context[0] + context[1];
            return f;
        };
    }
};


template <class T>
std::vector<T> polynomialFeatures( const std::vector<T>& input, unsigned int degree, bool interaction_only, bool include_bias )
{
    std::vector<T> features = input;
    std::vector<T> prev_chunk = input;
    std::vector<size_t> indices( input.size() );
    std::iota( indices.begin(), indices.end(), 0 );

    for ( int d = 1 ; d < degree ; ++d )
    {
        // Create a new chunk of features for the degree d:
        std::vector<T> new_chunk;
        // Multiply each component with the products from the previous lower degree:
        for ( size_t i = 0 ; i < input.size() - ( interaction_only ? d : 0 ) ; ++i )
        {
            // Store the index where to start multiplying with the current component at the next degree up:
            size_t next_index = new_chunk.size();
            for ( auto coef_it = prev_chunk.begin() + indices[i + ( interaction_only ? 1 : 0 )] ; coef_it != prev_chunk.end() ; ++coef_it )
            {
                new_chunk.push_back( input[i]**coef_it );
            }
            indices[i] = next_index;
        }
        // Extend the feature vector with the new chunk of features:
        features.reserve( features.size() + std::distance( new_chunk.begin(), new_chunk.end() ) );
        features.insert( features.end(), new_chunk.begin(), new_chunk.end() );

        prev_chunk = new_chunk;
    }
    if ( include_bias )
        features.insert( features.begin(), 1 );

    return features;
}

#endif

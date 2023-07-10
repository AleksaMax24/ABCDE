#ifndef MU_TMP_SCRIPTS_SUMMARY_STATISTICS_COMPUTER_H
#define MU_TMP_SCRIPTS_SUMMARY_STATISTICS_COMPUTER_H

#include <iostream>
#include <cmath>
#include <vector>
#include "kernel_computer.h"
#include "NeuralNetwork.h"

class DistributionDifferenceStatistics {
public:
    virtual double compute(const std::vector<double>& x, const std::vector<double>& y) = 0;
};

/**
 * MMD there is mean MMD^2 baseStatistics theta
 * */
class MMD : public DistributionDifferenceStatistics {
public:
    explicit MMD(const Kernel& kernel) : kernel_(kernel) {}
    double compute(const std::vector<double>& x, const std::vector<double>& y) override {
        int n = (int) x.size();
        double sum = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; ++j) {
                if (i != j)
                    sum += kernel_.compute(x[i], x[j]) + kernel_.compute(y[i], y[j]) -
                           kernel_.compute(x[i], y[j]) - kernel_.compute(x[j], y[i]);
            }
        }

        return fabs(sum / (n * (n - 1)));
    }
private:
    const Kernel& kernel_;
};

class MSE: public DistributionDifferenceStatistics {
public:
    double compute(const std::vector<double> &x, const std::vector<double> &y) override {
        double sum = 0.0;
        int n = (int) x.size();

        for(int i = 0; i < n; i++) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }

        return sum / n;
    }
};

class MAE: public DistributionDifferenceStatistics {
public:
    double compute(const std::vector<double> &x, const std::vector<double> &y) override {
        double sum = 0.0;
        int n = (int) x.size();

        for(int i = 0; i < n; i++) {
            sum += fabs(x[i] - y[i]);
        }

        return sum / n;
    }
};

/**
 * @param baseStatistics need to compute distance for NN resulting vector
 * */
class NeuralNetworkDifferenceComputer : public DistributionDifferenceStatistics{
public:
    explicit NeuralNetworkDifferenceComputer(NeuralNetworkDistribution networkDistribution, DistributionDifferenceStatistics& baseStatistics)
            : networkDistribution(std::move(networkDistribution)), baseStatistics(baseStatistics) {}
    double compute(const std::vector<double> &x, const std::vector<double> &y) override {
        std::vector<double> x_res = networkDistribution.predict(x);
        std::vector<double> y_res = networkDistribution.predict(y);

        return baseStatistics.compute(x_res, y_res);
    }
private:
    NeuralNetworkDistribution networkDistribution;
    DistributionDifferenceStatistics& baseStatistics;
};
#endif //MU_TMP_SCRIPTS_SUMMARY_STATISTICS_COMPUTER_H

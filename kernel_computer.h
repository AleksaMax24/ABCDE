#ifndef MU_TMP_SCRIPTS_KERNEL_COMPUTER_H
#define MU_TMP_SCRIPTS_KERNEL_COMPUTER_H

class Kernel {
public:
    virtual double compute(double x, double y) const = 0;
};

class GaussianKernel : public Kernel {
public:
    explicit GaussianKernel(double sigma) : sigma_(sigma) {}
    double compute(double x, double y) const override {
        double distance_squared = (x - y) * (x - y);
        return std::exp(-distance_squared / (2 * sigma_ * sigma_));
    }
private:
    double sigma_;
};


class PolynomialKernel : public Kernel {
public:
    PolynomialKernel(double degree, double coefficient, double constant) :
            degree_(degree), coefficient_(coefficient), constant_(constant) {}

    double compute(double x, double y) const override {
        return std ::pow(coefficient_ * x * y + constant_, degree_);
    }
private:
    double degree_;
    double coefficient_;
    double constant_;
};
#endif //MU_TMP_SCRIPTS_KERNEL_COMPUTER_H

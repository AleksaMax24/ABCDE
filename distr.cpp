#include "pch.h"


double Distribution::prior_distribution(Distribution::TYPE_DISTR mode, const double param1, const double param2)
{
	if (mode == NORM)
	{
		return getNormalSample();
	}
	if (mode == NORM_WITH_PARAM)
	{
		return getNormalSampleWithParam(param1, param2);
	}
	if (mode == EXPON)
	{
		return getLrand(param1);
	}
	if (mode == RANDOM)
	{
		return getRandomSample(param1, param2);
	}
	return getRandomSample(param1, param2);
}

double Distribution::getRandomSample(double param1, double param2)
{
	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_real_distribution<double> distribution(param1, param2);
	double x = distribution(generator);
	return x;
}
double Distribution::getNormalSample()
{
	double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double r = u * u + v * v;
	if (r == 0 || r > 1) return getNormalSample();
	double c = sqrt(-2 * log(r) / r);
	return u * c;
}

double Distribution::getNormalSampleWithParam(double mean, double var)
{
	std::random_device mch;
	std::default_random_engine gen(mch());
	std::normal_distribution<double> d(mean, var);
	return d(gen);
}

double Distribution::getLrand(double l)
{
	std::random_device mch;
	std::default_random_engine gen(mch());
	std::exponential_distribution<double> d(l);
	return d(gen);
}

double Distribution::kernelNormalSampleWithParam(double x, double mean, double var)
{
	normal norm(mean, var);
	return cdf(norm, x);
}

double Distribution::kernel_function(Distribution::TYPE_DISTR mode, double x, const double param1, const double param2)
{
	if (mode == NORM_WITH_PARAM)
	{
		return kernelNormalSampleWithParam(x, param1, param2);
	}
	return kernelNormalSampleWithParam(x, param1, param2);
}

int Distribution::generate_seed()
{
	vector<string> components;
	string output;
	for (int i = 0; i < 3; i++)
	{
		components.push_back(to_string(static_cast<int>(prior_distribution(Distribution::TYPE_DISTR::RANDOM, 1.0, 1000.0))));
	}
	for (int i = 0; i < 3; i++)
	{
		output += components[i];
	}
	return stoi(output);
}

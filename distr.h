#pragma once
#include "pch.h"

class Distribution
{
    public:
	    enum TYPE_DISTR
            {
	        NORM,
	        EXPON,
	        NORM_WITH_PARAM,
	        RANDOM
             };

		typedef struct
		{
			vector<double> param;
			double delta;
		}Thetha;

		typedef struct
		{
			Thetha * thetha;
			double * w;
			double * error;
			double delta_one;
		}Posterior;

		double prior_distribution(TYPE_DISTR mode, const double param1 = 0.0, const double param2 = 0.0);
		double getRandomSample(double param1, double param2);
		double getNormalSample();	
		double getNormalSampleWithParam(double mean, double var);
		double getLrand(double l);
		double kernelNormalSampleWithParam(double x, double mean, double var);
		double kernel_function(TYPE_DISTR mode, double x, const double param1 = 0.0, const double param2 = 0.0);
		int generate_seed();
};

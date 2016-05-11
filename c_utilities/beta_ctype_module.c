#include <gsl/gsl_sf_gamma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double bgbb_likelihood(double a, double b,double g, double d,float* x,float* tx,float* T,float* N,int n_samples){
    double numerator_j,denominator,res = 0.0,new_a,left_b,left_d,new_g;

    int i,j,max_i_j;

    //denominator = special.beta(a, b) * special.beta(g, d)
    denominator = gsl_sf_beta(a,b) * gsl_sf_beta(g,d);

    for(j = 0; j < n_samples; j++){
        //numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)
        new_a = a + x[j];
        numerator_j = gsl_sf_beta(new_a, b + T[j] - x[j]) * gsl_sf_beta(g, d + T[j]);
        //max_i = (T - tx - 1)
        max_i_j = T[j] - tx[j] - 1;
        /*
        for j in range(len(max_i)):
            xj = x[j]
            txj = tx[j]
            i = np.arange(max_i[j] + 1)
            numerator[j] += np.sum(special.beta(a + xj, b + txj - xj + i) * special.beta(g + 1, d + txj + i))
        */
        new_g = g + 1;
        left_b = b + tx[j]  - x[j];
        left_d = d + tx[j];
        for(i = 0; i <= max_i_j; i++){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }
        //ll = np.log(numerator / denominator)
        res += log(numerator_j / denominator) * N[j];
    }
    //return -(ll * N).sum()
    return -res;
}


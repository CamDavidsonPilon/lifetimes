#include <gsl/gsl_sf_gamma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//compile on osX
//gcc -shared -Wl,-install_name,testlib.so -I/usr/include/  -lgsl -lcblas -lm -o betalib.so -fPIC beta_ctype_module.c


double bgbb_likelihood_compressed(double a, double b,double g, double d,float* x,float* tx,float* T,float* N,int n_samples){

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
        for(i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }
        //ll = np.log(numerator / denominator)
        res += log(numerator_j / denominator) * N[j];
    }
    //return -(ll * N).sum()
    return -res;
}

double bgbb_likelihood(double a, double b,double g, double d,float* x,float* tx,float* T,int n_samples){

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
        for(i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }
        //ll = np.log(numerator / denominator)
        res += log(numerator_j / denominator);
    }
    //return -(ll * N).sum()
    return -res;
}

double bgbbbb_likelihood_compressed(double a, double b,double g, double d, double e, double z,float* x,float* tx,float* T, float* xp,float* N,int n_samples){

    double numerator_j,denominator,res = 0.0,new_a,left_b,left_d,new_g,purchase_term;

    int i,j,max_i_j;

    //denominator = special.beta(a, b) * special.beta(g, d)
    denominator = gsl_sf_beta(a,b) * gsl_sf_beta(g,d);

    for(j = 0; j < n_samples; j++){
        //numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)
        new_a = a + x[j];
        numerator_j = gsl_sf_beta(new_a, b + T[j] - x[j]) * gsl_sf_beta(g, d + T[j]);
        //max_i = (T - tx - 1)
        max_i_j = T[j] - tx[j] - 1.0;
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
        for(i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }

        //purchase_term = special.beta(e + xp, x - xp + z + 1) / special.beta(e, z)
        purchase_term = gsl_sf_beta(e + xp[j], x[j] - xp[j] + z + 1.0) / gsl_sf_beta(e, z);

        //ll = np.log(numerator / denominator)
        res += (log(numerator_j / denominator) + log(purchase_term)) * N[j];
    }
    //return ll_purchases + BGBBFitter._negative_log_likelihood(sub_params, freq, rec, T, penalizer_coef, N)
    return -res;
}

double bgbbbb_likelihood(double a, double b,double g, double d, double e, double z,float* x,float* tx,float* T, float* xp,int n_samples){

    double numerator_j,denominator,res = 0.0,new_a,left_b,left_d,new_g,purchase_term;

    int i,j,max_i_j;

    //denominator = special.beta(a, b) * special.beta(g, d)
    denominator = gsl_sf_beta(a,b) * gsl_sf_beta(g,d);

    for(j = 0; j < n_samples; j++){
        //numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)
        new_a = a + x[j];
        numerator_j = gsl_sf_beta(new_a, b + T[j] - x[j]) * gsl_sf_beta(g, d + T[j]);
        //max_i = (T - tx - 1)
        max_i_j = T[j] - tx[j] - 1.0;
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
        for(i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }

        //purchase_term = special.beta(e + xp, x - xp + z + 1) / special.beta(e, z)
        purchase_term = gsl_sf_beta(e + xp[j], x[j] - xp[j] + z + 1.0) / gsl_sf_beta(e, z);

        //ll = np.log(numerator / denominator)
        res += (log(numerator_j / denominator) + log(purchase_term));
    }
    //return ll_purchases + BGBBFitter._negative_log_likelihood(sub_params, freq, rec, T, penalizer_coef, N)
    return -res;
}
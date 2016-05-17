#include <gsl/gsl_sf_gamma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//compile on osX
//gcc -shared -Wl,-install_name,-.so -I/usr/include/  -lgsl -lcblas -lm -o betalib.so -fPIC beta_ctype_module.c


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

double bgbbbb_likelihood_compressed_func(double a, double b,double g, double d, double e, double z,float* x,float* tx,float* T, float* xp,float* N,int n_samples){

    double purchase_term = 0,bgbb;
    int j;

    bgbb = bgbb_likelihood_compressed(a, b, g, d, x, tx, T, N, n_samples);


    for(j = 0; j < n_samples; j++){
        //purchase_term = special.beta(e + xp, x - xp + z + 1) / special.beta(e, z)
        purchase_term += log(gsl_sf_beta(e + xp[j], x[j] - xp[j] + z + 1.0) / gsl_sf_beta(e, z)) * N[j];
    }

    //return ll_purchases + BGBBFitter._negative_log_likelihood(sub_params, freq, rec, T, penalizer_coef, N)
    return bgbb - purchase_term;
}

double bgbbbb_likelihood_compressed_optimized(double a, double b,double g, double d, double e, double z,float* data /*x0_tx1_T2_xp3_N4*/,int n_samples){

    double numerator_j,denominator,res = 0.0,new_a,left_b,left_d,new_g,purchase_term;
    float x_j,tx_j,T_j,xp_j,N_j;
    int i,j,max_i_j,N;

    //denominator = special.beta(a, b) * special.beta(g, d)
    denominator = gsl_sf_beta(a,b) * gsl_sf_beta(g,d);
    N = n_samples * 5;
    for(j = 0; j < N; j += 5){
        x_j = data[j];
        tx_j = data[j+1];
        T_j = data[j+2];
        xp_j = data[j+3];
        N_j = data[j+4];

        //printf("x:%0.0f tx:%0.0f T:%0.0f xp:%0.0f N:%0.0f\n",x_j,tx_j,T_j,xp_j,N_j);
        //numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)
        new_a = a + x_j;
        numerator_j = gsl_sf_beta(new_a, b + T_j - x_j) * gsl_sf_beta(g, d + T_j);
        //max_i = (T - tx - 1)
        max_i_j = T_j - tx_j - 1.0;
        /*
        for j in range(len(max_i)):
            xj = x[j]
            txj = tx[j]
            i = np.arange(max_i[j] + 1)
            numerator[j] += np.sum(special.beta(a + xj, b + txj - xj + i) * special.beta(g + 1, d + txj + i))
        */
        new_g = g + 1;
        left_b = b + tx_j  - x_j;
        left_d = d + tx_j;
        for(i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }

        //purchase_term = special.beta(e + xp, x - xp + z + 1) / special.beta(e, z)
        purchase_term = gsl_sf_beta(e + xp_j, x_j - xp_j + z + 1.0) / gsl_sf_beta(e, z);

        //ll = np.log(numerator / denominator)
        res += (log(numerator_j / denominator) + log(purchase_term)) * N_j;
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

double bgbbbb_likelihood_compressed_float(double a, double b,double g, double d, double e, double z,float* x,float* tx,float* T, float* xp,float* N,int n_samples){

    float res = 0.0;

    //denominator = special.beta(a, b) * special.beta(g, d)
    float log_denominator = gsl_sf_lnbeta(a,b) * gsl_sf_beta(g,d);

    for(int j = 0; j < n_samples; j++){
        //numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)
        float new_a = a + x[j];
        float numerator_j = gsl_sf_beta(new_a, b + T[j] - x[j]) * gsl_sf_beta(g, d + T[j]);
        //max_i = (T - tx - 1)
        int max_i_j = T[j] - tx[j] - 1;
        /*
        for j in range(len(max_i)):
            xj = x[j]
            txj = tx[j]
            i = np.arange(max_i[j] + 1)
            numerator[j] += np.sum(special.beta(a + xj, b + txj - xj + i) * special.beta(g + 1, d + txj + i))
        */
        float new_g = g + 1;
        float left_b = b + tx[j]  - x[j];
        float left_d = d + tx[j];
        for(int i = 0; i <= max_i_j; ++i){
            numerator_j += gsl_sf_beta(new_a, left_b + i) * gsl_sf_beta(new_g, left_d + i);
        }

        //purchase_term = special.beta(e + xp, x - xp + z + 1) / special.beta(e, z)
        float purchase_term = gsl_sf_beta(e + xp[j], x[j] - xp[j] + z + 1.0) / gsl_sf_beta(e, z);

        //ll = np.log(numerator / denominator)
        res += ((log(numerator_j) - log_denominator) + log(purchase_term)) * N[j];
    }
    //return ll_purchases + BGBBFitter._negative_log_likelihood(sub_params, freq, rec, T, penalizer_coef, N)
    return -res;
}
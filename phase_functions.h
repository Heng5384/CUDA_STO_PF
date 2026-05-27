#ifndef PHASE_FUNCTIONS_H
#define PHASE_FUNCTIONS_H

// Phase field interpolation functions

// Quintic h(phi) = phi^3*(6*phi^2 - 15*phi + 10)
__device__ __host__ static inline double h_of_phi(double phi){
    double p2 = phi * phi;
    double p3 = p2 * phi;
    return p3 * (6.0 * p2 - 15.0 * phi + 10.0);
}

// h'(phi) = 30*phi^2*(1-phi)^2
__device__ __host__ static inline double h_prime_of_phi(double phi){
    double p2 = phi * phi;
    return 30.0 * p2 * (1.0 - phi) * (1.0 - phi);
}

__device__ __host__ static inline double h_of_eta(double eta){
    return h_of_phi(eta);
}

__device__ __host__ static inline double h_prime_of_eta(double eta){
    return h_prime_of_phi(eta);
}

__device__ __host__ static inline void phase_fractions_gp(double phi, double eta,
                                                          double *h_alpha,
                                                          double *h_GP,
                                                          double *h_beta) {
    double h_phi = h_of_phi(phi);
    double h_eta = h_of_eta(eta);
    if (h_beta) {
        *h_beta = h_phi;
    }
    if (h_GP) {
        *h_GP = (1.0 - h_phi) * h_eta;
    }
    if (h_alpha) {
        *h_alpha = (1.0 - h_phi) * (1.0 - h_eta);
    }
}

__device__ __host__ static inline double stiffness_beta_weight_gp(double phi, double eta) {
    double h_beta = 0.0;
    phase_fractions_gp(phi, eta, NULL, NULL, &h_beta);
    return h_beta;
}

// Double-well potential g(phi) = phi^2*(1-phi)^2
__device__ __host__ static inline double g_of_phi(double phi){
    return phi*phi * (1.0 - phi) * (1.0 - phi);
}

// g'(phi) = 2*phi*(1-phi)*(1-2*phi) for double-well
__device__ __host__ static inline double g_prime_of_phi(double phi){
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi);
}

#endif // PHASE_FUNCTIONS_H


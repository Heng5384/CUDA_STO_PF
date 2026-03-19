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

// Double-well potential g(phi) = phi^2*(1-phi)^2
__device__ __host__ static inline double g_of_phi(double phi){
    return phi*phi * (1.0 - phi) * (1.0 - phi);
}

// g'(phi) = 2*phi*(1-phi)*(1-2*phi) for double-well
__device__ __host__ static inline double g_prime_of_phi(double phi){
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi);
}

#endif // PHASE_FUNCTIONS_H




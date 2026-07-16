#ifndef PHASE_PDAS_REDUCTION_H
#define PHASE_PDAS_REDUCTION_H

#include "phase_kkt_utils.h"

// Exact integer-only summaries for the PDAS trial active-set comparison.
// The phase field and active-code arrays remain authoritative per-cell state;
// these structures only replace contended global diagnostic counters.
struct PhasePdasTrialBlockSummary {
    unsigned long long lower_count;
    unsigned long long upper_count;
    unsigned long long free_count;
    unsigned long long active_change_count;
    unsigned long long invalid_count;
    unsigned long long nonfinite_count;
    unsigned long long bound_failure_count;
};

struct PhasePdasTrialDecisionPacket {
    unsigned long long lower_count;
    unsigned long long upper_count;
    unsigned long long free_count;
    unsigned long long active_change_count;
    unsigned long long invalid_count;
    unsigned long long nonfinite_count;
    unsigned long long bound_failure_count;
    int active_set_stable;
    int trial_valid;
};

#if defined(__CUDACC__)

enum { PHASE_PDAS_TRIAL_REDUCTION_THREADS = 256 };

__global__ void ctot_phase_pdas_compare_trial_active_kernel(
    const double *trial_phi_r, const double *ctot_r,
    const double *trial_raw_residual_r,
    const double *previous_active_code_r, double *trial_active_code_r,
    double alpha, double v_B, double x_min, double x_max,
    double bound_tol, PhasePdasTrialBlockSummary *block_summaries, int n)
{
    __shared__ unsigned int block_counts[7][PHASE_PDAS_TRIAL_REDUCTION_THREADS];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    unsigned int lower_count = 0;
    unsigned int upper_count = 0;
    unsigned int free_count = 0;
    unsigned int changed_count = 0;
    unsigned int invalid_count = 0;
    unsigned int nonfinite_count = 0;
    unsigned int bound_failure_count = 0;

    if (idx < n) {
        const double phi = trial_phi_r[idx];
        const double residual = trial_raw_residual_r[idx];
        const PhaseKktBounds bounds = phase_kkt_bounds_from_ctot(
            ctot_r[idx], v_B, x_min, x_max, bound_tol);
        const int code = phase_kkt_active_code(phi, residual, alpha, bounds);
        trial_active_code_r[idx] = (double)code;
        lower_count = code == PHASE_KKT_LOWER_ACTIVE;
        upper_count = code == PHASE_KKT_UPPER_ACTIVE;
        free_count = code == PHASE_KKT_FREE;
        invalid_count = code == PHASE_KKT_INVALID;
        nonfinite_count = !isfinite(phi) || !isfinite(residual) ||
                          !isfinite(ctot_r[idx]);
        bound_failure_count = !bounds.valid;
        if (!invalid_count &&
            (int)llrint(previous_active_code_r[idx]) != code)
            changed_count = 1;
    }

    block_counts[0][tid] = lower_count;
    block_counts[1][tid] = upper_count;
    block_counts[2][tid] = free_count;
    block_counts[3][tid] = changed_count;
    block_counts[4][tid] = invalid_count;
    block_counts[5][tid] = nonfinite_count;
    block_counts[6][tid] = bound_failure_count;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int field = 0; field < 7; ++field)
                block_counts[field][tid] += block_counts[field][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        PhasePdasTrialBlockSummary summary;
        summary.lower_count = block_counts[0][0];
        summary.upper_count = block_counts[1][0];
        summary.free_count = block_counts[2][0];
        summary.active_change_count = block_counts[3][0];
        summary.invalid_count = block_counts[4][0];
        summary.nonfinite_count = block_counts[5][0];
        summary.bound_failure_count = block_counts[6][0];
        block_summaries[blockIdx.x] = summary;
    }
}

__global__ void ctot_phase_pdas_reduce_trial_summaries_kernel(
    const PhasePdasTrialBlockSummary *block_summaries, int summary_count,
    PhasePdasTrialDecisionPacket *packet)
{
    __shared__ unsigned long long block_counts[7][PHASE_PDAS_TRIAL_REDUCTION_THREADS];
    const int tid = threadIdx.x;
    unsigned long long counts[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int summary = tid; summary < summary_count; summary += blockDim.x) {
        const PhasePdasTrialBlockSummary value = block_summaries[summary];
        counts[0] += value.lower_count;
        counts[1] += value.upper_count;
        counts[2] += value.free_count;
        counts[3] += value.active_change_count;
        counts[4] += value.invalid_count;
        counts[5] += value.nonfinite_count;
        counts[6] += value.bound_failure_count;
    }
#pragma unroll
    for (int field = 0; field < 7; ++field)
        block_counts[field][tid] = counts[field];
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int field = 0; field < 7; ++field)
                block_counts[field][tid] += block_counts[field][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        PhasePdasTrialDecisionPacket result;
        result.lower_count = block_counts[0][0];
        result.upper_count = block_counts[1][0];
        result.free_count = block_counts[2][0];
        result.active_change_count = block_counts[3][0];
        result.invalid_count = block_counts[4][0];
        result.nonfinite_count = block_counts[5][0];
        result.bound_failure_count = block_counts[6][0];
        result.active_set_stable = result.active_change_count == 0;
        result.trial_valid = result.invalid_count == 0;
        *packet = result;
    }
}

#endif

#endif

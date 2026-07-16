# CUDA相场模拟Makefile
NVCC = nvcc
CUDA_ROOT ?= /usr/local/cuda-12.9

# 编译选项
# 默认架构：RTX 5080 (compute capability 12.0) -> sm_120
# 如需兼容其它 GPU，可在命令行覆盖 CUDA_ARCH
CUDA_ARCH ?= sm_120
PF_T380_PARAMS ?=
PF_T400_PARAMS ?=
PF_CUDA_GATE_LOG ?=
# 屏蔽 gcc/clang 的 format-truncation 噪声警告（snprintf 生成文件名路径处常见）
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++14 --expt-relaxed-constexpr -Xcompiler -Wno-format-truncation
INCLUDES = -I$(CUDA_ROOT)/include
LDFLAGS = -L$(CUDA_ROOT)/lib64
LDLIBS = -lcufft -lcudart -lm

# 目标程序
BIN_MAIN = main_cuda
BIN_TEST = test_memory_ledger
BIN_PF_GATE_HOST = pf_ctot_hard_gate_host_bin
BIN_PF_GATE_CUDA = pf_ctot_hard_gate_cuda_bin
BIN_PF_CTOT_STATE_HOST = pf_ctot_state_host_bin
BIN_PF_CTOT_ADJOINT_CUDA = pf_ctot_adjoint_cuda_bin
BIN_PF_PHASE_KKT_HOST = pf_ctot_phase_kkt_host_bin
BIN_PF_P2_ACTIVE_REDUCTION = pf_ctot_p2_active_reduction_bin
BIN_COARSE4_MOBILITY_HOST = coarse4_mobility_host_bin
BIN_CTOT_FEASIBLE_STORAGE_HOST = ctot_feasible_storage_host_bin
BIN_BOUNDED_RETRY_BDF2_HOST = bounded_retry_bdf2_host_bin

# 源文件
SRC_MAIN = main_cuda.cu cuda_kernels.cu cuda_common.cu
SRC_TEST = test_memory_ledger.cu

# 头文件
HDR = cuda_common.h cuda_kernels.h pf_params.h phase_functions.h thermo_utils.h io_vtk_cuda.h phase_kkt_utils.h phase_pdas_reduction.h ctot_transport_bound_utils.h bounded_retry_bdf2_utils.h

.PHONY: all clean test test_circle test_pf_ctot_hard_gate_host test_pf_ctot_hard_gate_cuda test_pf_ctot_state_host test_pf_ctot_adjoint_cuda test_pf_ctot_phase_kkt_host test_pf_ctot_p2_active_reduction test_coarse4_mobility_host test_ctot_feasible_storage_host test_bounded_retry_bdf2_host help

all: $(BIN_MAIN)

help:
	@echo "Targets:"
	@echo "  make main_cuda            # build main program"
	@echo "  make test                 # build+run smoke test (memory ledger)"
	@echo "  make test_circle          # smoke test + summary plot helper"
	@echo "  make test_pf_ctot_hard_gate_host # host thermodynamic/mobility hard gates"
	@echo "  make test_pf_ctot_hard_gate_cuda # tiny CUDA/cuFFT parity hard gate"
	@echo "  make test_pf_ctot_state_host     # Ctot state/restart/rollback contract"
	@echo "  make test_pf_ctot_adjoint_cuda   # production face-flux/incidence adjoint identity"
	@echo "  make test_pf_ctot_phase_kkt_host # constrained phase KKT/active-set oracle"
	@echo "  make test_pf_ctot_p2_active_reduction # exact two-stage PDAS trial reduction"
	@echo "  make test_coarse4_mobility_host # coarse4 mobility/SPD/default-off contract"
	@echo "  make test_ctot_feasible_storage_host # Ctot feasible storage-map oracle"
	@echo "  make test_bounded_retry_bdf2_host # bounded-retry acceptance oracle"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ROOT=/usr/local/cuda-12.9 # CUDA toolkit path (must contain include/ and lib64/)"
	@echo "  CUDA_ARCH=sm_120              # GPU arch, e.g. sm_120 for RTX 5080"
	@echo "  PF_T380_PARAMS=<path>          # required by PF hard-gate audit"
	@echo "  PF_T400_PARAMS=<path>          # required by PF hard-gate audit"
	@echo "  PF_CUDA_GATE_LOG=<path>        # optional CUDA gate output"

# 主程序
$(BIN_MAIN): $(SRC_MAIN) $(HDR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC_MAIN) $(LDFLAGS) $(LDLIBS)

# 测试程序：当前使用不依赖 GPU 运行时的显存账本 smoke test
$(BIN_TEST): $(SRC_TEST)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC_TEST) $(LDFLAGS) $(LDLIBS)

# 测试目标
test: $(BIN_TEST)
	@echo "运行 smoke test（memory ledger）..."
	./$(BIN_TEST)

test_circle: test
	@echo "生成测试摘要图..."
	@python3 tools/analysis/plot_kirsch_comparison.py

$(BIN_PF_GATE_HOST): tests/pf_ctot_hard_gate_host.cpp phase_functions.h thermo_utils.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/pf_ctot_hard_gate_host.cpp

$(BIN_PF_GATE_CUDA): tests/pf_ctot_hard_gate_cuda.cu phase_functions.h thermo_utils.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I. -o $@ tests/pf_ctot_hard_gate_cuda.cu $(LDFLAGS) $(LDLIBS)

test_pf_ctot_hard_gate_host: $(BIN_PF_GATE_HOST)
	@mkdir -p reports/pf_ctot_production_candidate
	@test -n "$(PF_T380_PARAMS)" || (echo "PF_T380_PARAMS is required" >&2; exit 2)
	@test -n "$(PF_T400_PARAMS)" || (echo "PF_T400_PARAMS is required" >&2; exit 2)
	python3 scripts/run_pf_ctot_hard_gate_audit.py \
		--t380-params "$(PF_T380_PARAMS)" \
		--t400-params "$(PF_T400_PARAMS)" \
		$(if $(PF_CUDA_GATE_LOG),--cuda-log "$(PF_CUDA_GATE_LOG)",)

test_pf_ctot_hard_gate_cuda: $(BIN_PF_GATE_CUDA)
	./$(BIN_PF_GATE_CUDA)

$(BIN_PF_CTOT_STATE_HOST): tests/pf_ctot_state_host.cpp phase_functions.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/pf_ctot_state_host.cpp

test_pf_ctot_state_host: $(BIN_PF_CTOT_STATE_HOST)
	@mkdir -p reports/pf_ctot_production_candidate
	./$(BIN_PF_CTOT_STATE_HOST) reports/pf_ctot_production_candidate/ctot_state_unit_test_results.csv

$(BIN_BOUNDED_RETRY_BDF2_HOST): tests/test_bounded_retry_bdf2_utils.cpp bounded_retry_bdf2_utils.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/test_bounded_retry_bdf2_utils.cpp

test_bounded_retry_bdf2_host: $(BIN_BOUNDED_RETRY_BDF2_HOST)
	./$(BIN_BOUNDED_RETRY_BDF2_HOST)

$(BIN_PF_CTOT_ADJOINT_CUDA): tests/pf_ctot_adjoint_cuda.cu cuda_kernels.cu cuda_kernels.h phase_functions.h thermo_utils.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I. -o $@ tests/pf_ctot_adjoint_cuda.cu cuda_kernels.cu $(LDFLAGS) $(LDLIBS)

test_pf_ctot_adjoint_cuda: $(BIN_PF_CTOT_ADJOINT_CUDA)
	@mkdir -p reports/pf_ctot_production_candidate
	./$(BIN_PF_CTOT_ADJOINT_CUDA) reports/pf_ctot_production_candidate/prompt7g_cuda_adjoint_identity.csv

$(BIN_PF_PHASE_KKT_HOST): tests/pf_ctot_phase_kkt_host.cpp phase_kkt_utils.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/pf_ctot_phase_kkt_host.cpp

test_pf_ctot_phase_kkt_host: $(BIN_PF_PHASE_KKT_HOST)
	@mkdir -p reports/pf_ctot_production_candidate
	./$(BIN_PF_PHASE_KKT_HOST) reports/pf_ctot_production_candidate/prompt8_phase_kkt_host.csv

$(BIN_PF_P2_ACTIVE_REDUCTION): tests/pf_ctot_p2_active_reduction.cu phase_kkt_utils.h phase_pdas_reduction.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I. -o $@ tests/pf_ctot_p2_active_reduction.cu $(LDFLAGS) $(LDLIBS)

test_pf_ctot_p2_active_reduction: $(BIN_PF_P2_ACTIVE_REDUCTION)
	./$(BIN_PF_P2_ACTIVE_REDUCTION)

$(BIN_COARSE4_MOBILITY_HOST): tests/coarse4_mobility_host.cpp thermo_utils.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/coarse4_mobility_host.cpp

test_coarse4_mobility_host: $(BIN_COARSE4_MOBILITY_HOST)
	@mkdir -p reports/pf_ctot_production_candidate
	./$(BIN_COARSE4_MOBILITY_HOST) \
		reports/pf_ctot_production_candidate/coarse4_mobility_host_test.csv

$(BIN_CTOT_FEASIBLE_STORAGE_HOST): tests/ctot_feasible_storage_host.cpp ctot_transport_bound_utils.h
	$(CXX) -O2 -std=c++14 -I. -o $@ tests/ctot_feasible_storage_host.cpp

test_ctot_feasible_storage_host: $(BIN_CTOT_FEASIBLE_STORAGE_HOST)
	@mkdir -p reports/pf_ctot_production_candidate
	./$(BIN_CTOT_FEASIBLE_STORAGE_HOST) \
		reports/pf_ctot_production_candidate/ctot_feasible_storage_host_test.csv

clean:
	rm -f $(BIN_MAIN) $(BIN_TEST) $(BIN_PF_GATE_HOST) $(BIN_PF_GATE_CUDA) $(BIN_PF_CTOT_STATE_HOST) $(BIN_PF_CTOT_ADJOINT_CUDA) $(BIN_PF_PHASE_KKT_HOST) $(BIN_PF_P2_ACTIVE_REDUCTION) $(BIN_COARSE4_MOBILITY_HOST) $(BIN_CTOT_FEASIBLE_STORAGE_HOST) *.o
	rm -f circle_void_*.dat circle_void_*.vtk
	rm -f kirsch_comparison_*.png kirsch_comparison_placeholder.png kirsch_comparison_memory_ledger.txt

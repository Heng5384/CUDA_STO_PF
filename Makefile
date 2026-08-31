# CUDA相场模拟Makefile
NVCC = nvcc
CUDA_ROOT ?= /usr/local/cuda-12.9

# 编译选项
# 默认架构：RTX 5080 (compute capability 12.0) -> sm_120
# 如需兼容其它 GPU，可在命令行覆盖 CUDA_ARCH
CUDA_ARCH ?= sm_120
# 屏蔽 gcc/clang 的 format-truncation 噪声警告（snprintf 生成文件名路径处常见）
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++14 --expt-relaxed-constexpr -Xcompiler -Wno-format-truncation
INCLUDES = -I$(CUDA_ROOT)/include
LDFLAGS = -L$(CUDA_ROOT)/lib64
LDLIBS = -lcufft -lcudart -lm

# 目标程序
BIN_MAIN = main_cuda
BIN_TEST = test_memory_ledger

# 源文件
SRC_MAIN = main_cuda.cu cuda_kernels.cu cuda_common.cu pf_zero_mode_checkpoint.cpp
SRC_TEST = test_memory_ledger.cu

# 头文件
HDR = cuda_common.h cuda_kernels.h pf_params.h phase_functions.h thermo_utils.h io_vtk_cuda.h pf_zero_mode_checkpoint.h generated/pf_kwn_validation_contract_v1.h

BIN_ZERO_MODE_CHECKPOINT_TEST = test_pf_zero_mode_checkpoint_bin
BIN_THERMO_PROBE = pf_thermo_probe

.PHONY: all clean test test_circle test_pf_zero_mode_checkpoint test_pf_thermo_probe help

all: $(BIN_MAIN)

help:
	@echo "Targets:"
	@echo "  make main_cuda            # build main program"
	@echo "  make test                 # build+run smoke test (memory ledger)"
	@echo "  make test_circle          # smoke test + summary plot helper"
	@echo "  make test_pf_zero_mode_checkpoint # host-only restart provenance test"
	@echo "  make test_pf_thermo_probe # build generated-contract host thermo probe"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ROOT=/usr/local/cuda-12.9 # CUDA toolkit path (must contain include/ and lib64/)"
	@echo "  CUDA_ARCH=sm_120              # GPU arch, e.g. sm_120 for RTX 5080"

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

test_pf_zero_mode_checkpoint: $(BIN_ZERO_MODE_CHECKPOINT_TEST)
	./$(BIN_ZERO_MODE_CHECKPOINT_TEST)

$(BIN_ZERO_MODE_CHECKPOINT_TEST): tests/test_pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h
	$(CXX) -O2 -std=c++14 -Wall -Wextra -pedantic -o $@ tests/test_pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.cpp

test_pf_thermo_probe: $(BIN_THERMO_PROBE)
	./$(BIN_THERMO_PROBE) --expected-contract-hash "$$(python3 tools/generate_pf_contract_header.py --check | python3 -c 'import json,sys; print(json.load(sys.stdin)["hash"])')" >/dev/null

$(BIN_THERMO_PROBE): tools/pf_thermo_probe.cpp generated/pf_kwn_validation_contract_v1.h
	$(CXX) -O2 -std=c++14 -Wall -Wextra -pedantic -I. -o $@ tools/pf_thermo_probe.cpp

test_circle: test
	@echo "生成测试摘要图..."
	@python3 tools/analysis/plot_kirsch_comparison.py

clean:
	rm -f $(BIN_MAIN) $(BIN_TEST) $(BIN_ZERO_MODE_CHECKPOINT_TEST) $(BIN_THERMO_PROBE) *.o
	rm -f circle_void_*.dat circle_void_*.vtk
	rm -f kirsch_comparison_*.png kirsch_comparison_placeholder.png kirsch_comparison_memory_ledger.txt

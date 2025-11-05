# pspaQR - Parallel Sparsified QR Factorization

A distributed-memory parallel implementation of the sparsified QR factorization algorithm using the TaskTorrent runtime system. This repository provides efficient QR factorization for large sparse matrices through hierarchical sparsification and task-based parallelization.

Key features:
- Task-based parallelization with TaskTorrent runtime
- Hierarchical sparsification for reduced computational complexity
- Support for both square and rectangular sparse matrices
- Geometric and graph-based matrix partitioning
- Iterative refinement (GMRES, CGLS) when used as a preconditioner 
- MPI-based distributed memory parallelization

## Citation

If you use this code in your research, please cite Chapter 5 of:

```bibtex
@book{gnanasekaran2022fast,
  title={Fast Orthogonal Factorization for Sparse Matrices: Theory, Implementation, and Application},
  author={Gnanasekaran, Abeynaya},
  year={2022},
  publisher={Stanford University}
}
```

Thesis available at: https://purl.stanford.edu/yh423ny0931

## Dependencies

Required:
- C++14 compatible compiler (g++, clang++)
- CMake (>= 3.13.0)
- MPI (with `MPI_THREAD_FUNNELED` support)
- [TaskTorrent](https://github.com/leopoldcambier/TaskTorrent) runtime system
- Eigen3 (>= 3.3)
- BLAS/LAPACK (Intel MKL or OpenBLAS)
- METIS (or PaToH for hypergraph partitioning)

Optional:
- HSL MC64 for bipartite matching-based row ordering (required for rectangular matrices)
- HWLOC for hardware topology awareness

## Installation

1. **Install dependencies:**
   ```bash
   # Set your software installation directory
   export SOFTROOT="${HOME}/Softwares"

   # Install TaskTorrent, Eigen3, METIS, etc.
   ./install_deps.sh
   ```

2. **Configure build settings:**
   Edit `install.sh` to set:
   - `SOFTROOT`: Path to dependencies
   - `SPAQR_USE_MKL`: Use Intel MKL (ON) or OpenBLAS (OFF)
   - `SPAQR_USE_METIS`: Use METIS (ON) or PaToH (OFF)
   - `SPAQR_USE_HSL`: Enable HSL MC64 (OFF by default)
   - `TASKTORRENT_ROOT_DIR`: Path to TaskTorrent installation

3. **Build:**
   ```bash
   ./install.sh
   ```

   The executable `spaQR` will be created in the `build/` directory.

## Usage

Basic usage:
```bash
mpirun -n <num_procs> ./build/spaQR -m <matrix_file> [options]
```

### Required Arguments
- `-m, --matrix <file>`: Input sparse matrix in Matrix Market format

### Key Options

**Factorization:**
- `-l, --lvl <int>`: Number of hierarchical levels (default: ceil(log2(ncols/64)))
- `-t, --tol <float>`: Sparsification tolerance (default: 1e-1)
- `--skip <int>`: Skip sparsification for top levels (default: 0)
- `--order <float>`: Sparsification scheme order: 1 or 1.5 (default: 1)
- `--scale <0|1>`: Enable diagonal scaling (default: 1, required for rectangular matrices)

**Partitioning:**
- `--coordinates <file>`: Coordinates file for geometric partitioning (Matrix Market format)
- `-n <int>`: Use tensor grid n^d for geometric partitioning
- `-d <int>`: Dimension for tensor grid partitioning
- `--hsl <0|1>`: Use HSL MC64 for row ordering (requires HSL compilation)

**Iterative Solver:**
- `-i, --iterations <int>`: Maximum solver iterations (default: 300)
- `--res <float>`: Target relative residual (default: 1e-12)
- `--rhs <file>`: Right-hand side vector (Matrix Market format)

**Performance:**
- `--n_threads <int>`: Number of threads per MPI process (default: 1)
- `--verb <0|1|2|3>`: Verbosity level for debugging (default: 0)
- `--log <0|1>`: Enable TaskTorrent profiling/logging (default: 0)

### Example

```bash
# Run on 4 MPI processes with 8 threads each
mpirun -n 4 ./build/spaQR \
    -m matrix.mtx \
    --lvl 5 \
    --tol 1e-2 \
    --n_threads 8 \
    --iterations 500 \
    --res 1e-10
```

## Project Structure

```
pspaQR_public/
├── include/          # Header files
│   ├── spaQR.hpp    # Main include
│   ├── ptree.h      # Parallel tree structures
│   ├── cluster.h    # Matrix clustering
│   ├── partition.h  # Partitioning routines
│   └── ...
├── src/             # Implementation files
│   ├── ptree.cpp    # Parallel tree implementation
│   ├── operations.cpp
│   ├── toperations.cpp
│   └── ...
├── external/        # External dependencies
│   ├── cxxopts.hpp # Command-line parsing
│   └── mmio.hpp    # Matrix Market I/O
├── tests/          # Test programs
├── pspaQR.cpp      # Main driver program
├── CMakeLists.txt  # CMake configuration
└── install.sh      # Build script
```

## Related Work

This implementation builds upon:
- **spaQR**: Sequential sparsified QR algorithm (https://github.com/Abeynaya/spaQR_public)
- **TaskTorrent**: Task-based runtime system for scientific computing (https://github.com/leopoldcambier/tasktorrent)
- Theory and algorithms detailed in Chapter 5 of the thesis

## Limitation

**Matrix Loading:** The current implementation does **not** distribute matrix loading across nodes. The entire sparse matrix must fit in memory on each node during the initial loading and partitioning phase. This creates a scaling limitation based on memory available per node. However, the fill-in blocks are distributed across the nodes and are efficiently allocated throughout the factorization.

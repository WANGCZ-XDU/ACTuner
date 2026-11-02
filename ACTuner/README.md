# ACTuner

ACTuner tunes HNSW construction parameters using LLM guidance, a surrogate
ensemble, and PPO. The objective is to maximize query throughput under a
target Recall@10 constraint. This repository uses GIST1M as an example.
The code can also be adapted to other datasets, such as SIFT and DEEP.

## Project Structure

| Folder/File | Description |
| --- | --- |
| `tuner.py` | Tuning workflow, case retrieval, surrogate model, and LLM guidance. |
| `lrrl.py` | PPO, reward calculation, LLM client, and evaluation environment. |
| `hnsw_backend.py` | Index construction, search execution, and QPS interpolation. |
| `config.yaml` | Dataset paths, tuning parameters, and evaluation settings. |
| `hnswlib/` | HNSW library and benchmark helpers. |
| `data/` | Base vectors, query vectors, and ground-truth neighbors. |
| `tests/` | HNSW benchmark executables, their C++ source, and Python unit tests. |

## Usage Instructions

### Environment Setup

Use Linux x86-64 and Python 3.11. The default preflight check requires at least
24 GiB of available RAM and 12 GiB of free disk space.

From the project directory `ACTuner`, create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.4.1 pandas==3.0.2 torch==2.7.1 \
  requests==2.34.2 PyYAML==6.0.3 openpyxl==3.1.5
chmod +x tests/test_index tests/test_search
```

The bundled executables require the system C/C++ and OpenMP runtime libraries.
If they are incompatible with your CPU or Linux environment, install GCC with
C++11 and OpenMP support and rebuild them:

```bash
for program in test_index test_search; do
  g++ -Ofast -std=c++11 -DHAVE_CXX0X -march=native -fPIC \
    -fopenmp -ftree-vectorize -I. "tests/${program}.cpp" \
    -o "tests/${program}" -lrt
done
sha256sum tests/test_index tests/test_search
```

After rebuilding, copy the two SHA-256 values into
`runtime.hnsw_build_exe_sha256` and `runtime.hnsw_search_exe_sha256` in
`config.yaml`, respectively.
PyTorch uses CUDA when available and otherwise runs on the CPU.

### Data Preparation

Download GIST1M from the [TEXMEX ANN benchmark](http://corpus-texmex.irisa.fr/).
The dataset contains 1,000,000 base vectors of 960 dimensions, 1,000 query
vectors, and ground-truth nearest neighbors.

From `ACTuner`, download and extract the required files:

```bash
mkdir -p data
curl -fL ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz -o data/gist.tar.gz
tar -xzf data/gist.tar.gz -C data --strip-components=1 \
  gist/gist_base.fvecs gist/gist_query.fvecs gist/gist_groundtruth.ivecs
```

The resulting layout should be:

```text
ACTuner/
  data/
    gist_base.fvecs
    gist_query.fvecs
    gist_groundtruth.ivecs
```

The dataset is not included in the repository. Git does not track empty
directories, so create `data/` after cloning. Dataset paths in `config.yaml`
are relative to the ACTuner directory.

### LLM Configuration

Set the API key before running:

```bash
export DOUBAO_API_KEY='your-api-key'
```

The model and API URL are defined in the `llm` section of `config.yaml`.
They can also be set through `DOUBAO_MODEL` and `DOUBAO_BASE_URL`. Choose a
model accessible to your API account. Keep API keys in environment variables.

### Online Tuning

From `ACTuner`, run:

```bash
bash run_server.sh --target-recall 0.90
```

Supported recall targets are `[0.90, 0.92, 0.94, 0.96, 0.98, 0.99]`.
The launcher checks the dataset and runtime settings before starting.

The default budget is **30 real evaluations**: 10 warm-up evaluations followed
by 10 online iterations with 2 real evaluations each. Each index is evaluated
on all 1,000 queries with five timed repeats. Final validation reuses the
selected configuration's measured curve.

OpenMP uses all available CPUs by default. Set `OMP_NUM_THREADS` before
running to choose a smaller thread count. Use the same hardware and thread
settings when comparing query throughput, and run experiments sequentially.

Results are saved under `ACTuner/runs/`, including measured curves, parameter
configurations, timing records, and summary spreadsheets. Successful runs
update `state/case_base.json` for retrieval in later runs. Use a fresh copy
of the repository when an independent run with an empty case base is required.

### Evaluate a Single Configuration

To build and evaluate one index without running the tuning loop:

```bash
python -B tests/test_backend.py --target-recall 0.90 --M 32 --ef-construction 400
```

This command uses the full GIST1M dataset and writes its results under `runs/`.

## Example

After installing dependencies and preparing the data, run an experiment with
a target recall of 0.96:

```bash
export DOUBAO_API_KEY='your-api-key'
bash run_server.sh --target-recall 0.96
```

Algorithm parameters and relative paths can be edited in `config.yaml`.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

These tests use synthetic data and mocked API responses; they do not require
GIST1M or an API key.

## License

The bundled hnswlib code is distributed under the
[Apache License 2.0](hnswlib/LICENSE).

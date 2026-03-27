# ring_tsfast

PyTorch re-implementation of [RNNO](https://github.com/simon-bachhuber/ring) training using [tsfast](https://github.com/daniel-om-weber/tsfast).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a local clone of `tsfast` as a sibling directory.

```bash
uv sync
```

For comparison runs using the original JAX-based `imt-ring`:

```bash
uv sync --extra jax
```

## Scripts

| Script | Description |
|--------|-------------|
| `train_rnno.py` | Simplified RNNO training using `tsfast` (PyTorch) |
| `train_step2_trainRing_v2_rnno_tsfast.py` | Full-featured RNNO training using `tsfast` |
| `train_step1_generateData_v2.py` | Dataset generation (requires `imt-ring`) |
| `train_step2_trainRing_v2.py` | Original RING training (JAX, for comparison) |
| `train_step2_trainRing_v2_rnno.py` | Original RNNO training (JAX, for comparison) |

## Usage

### Training RNNO (tsfast)

Set `PATH_LAM4` in `train_rnno.py` to your dataset folder, then:

```bash
python train_rnno.py
```

### Generating training data (requires `--extra jax`)

```bash
python train_step1_generateData_v2.py 65536 ring_data --mot-art --dof-configuration "['111']"
```

### Training RING (JAX, requires `--extra jax`)

```bash
python train_step2_trainRing_v2.py ring_data 512 4800 --drop-dof 1.0 --lin-d 2 --layernorm --four-seg --drop-ja-2d 1.0
```

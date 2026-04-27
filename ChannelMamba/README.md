# ChannelMamba

H. Shi, K. Jin, X. Ren, W. Li and Y. Zhou, "ChannelMamba: A Mamba-Driven Selective State-Space Model for Channel Prediction of High-Mobility MIMO in 6G IoT," *IEEE Transactions on Wireless Communications*, vol. 25, pp. 5291-5305, 2026, doi: 10.1109/TWC.2025.3617502. [[paper](https://doi.org/10.1109/TWC.2025.3617502)]

Official implementation of **ChannelMamba**.

## Requirements

- Python 3.10
- PyTorch
- NVIDIA GPU + CUDA for full reproduction

```bash
conda env create -f environment.yml
conda activate channelmamba
```

## Dataset

The dataset should be organized as:

```text
<data-root>/
├── train/
│   ├── H_U_his_train.mat
│   ├── H_U_pre_train.mat
│   └── H_D_pre_train.mat
└── test/
    ├── H_U_his_test.mat
    ├── H_U_pre_test.mat
    └── H_D_pre_test.mat
```

The default `.mat` keys are:

- Training input: `H_U_his_train`
- TDD training target: `H_U_pre_train`
- FDD training target: `H_D_pre_train`
- Testing input: `H_U_his_test`
- TDD testing target: `H_U_pre_test`
- FDD testing target: `H_D_pre_test`

The datasets in this work are generated with QuaDRiGa following the settings described in the paper.

## Training

- Train in a TDD setting:

```bash
python scripts/train.py \
  --config configs/train_tdd.yaml \
  --data-root /path/to/dataset \
  --duplex tdd
```

- Train in an FDD setting:

```bash
python scripts/train.py \
  --config configs/train_fdd.yaml \
  --data-root /path/to/dataset \
  --duplex fdd
```

## Evaluation

- Single-condition evaluation:

```bash
python scripts/eval.py \
  --config configs/eval_tdd.yaml \
  --data-root /path/to/dataset \
  --duplex tdd \
  --suite single \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

- SNR evaluation:

```bash
python scripts/eval.py \
  --config configs/eval_tdd.yaml \
  --data-root /path/to/dataset \
  --duplex tdd \
  --suite snr \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

- Velocity evaluation:

```bash
python scripts/eval.py \
  --config configs/eval_tdd.yaml \
  --data-root /path/to/dataset \
  --duplex tdd \
  --suite velocity \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

- Run all released evaluation suites:

```bash
python scripts/benchmark.py \
  --config configs/eval_tdd.yaml \
  --data-root /path/to/dataset \
  --duplex tdd \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

Parts of this repository were developed by borrowing some code from the open-source [LLM4CP repository](https://github.com/PKU-PCNI/LLM4CP). We gratefully acknowledge their contribution.

## Citation

If you find this repository helpful, please cite our paper.

```bibtex
@ARTICLE{shi2026channelmamba,
  author={Shi, Huaguang and Jin, Kaibo and Ren, Xiaoquan and Li, Wei and Zhou, Yi},
  journal={IEEE Transactions on Wireless Communications}, 
  title={ChannelMamba: A Mamba-Driven Selective State-Space Model for Channel Prediction of High-Mobility MIMO in 6G IoT}, 
  year={2026},
  volume={25},
  number={},
  pages={5291-5305},
  keywords={Computer architecture;Computational modeling;Transformers;Predictive models;MIMO;Data models;Accuracy;Adaptation models;Time series analysis;Deep learning;Channel prediction;Mamba;massive multi-input multi-output (m-MIMO);orthogonal frequency division multiplexing (OFDM);deep learning},
  doi={10.1109/TWC.2025.3617502}}
```

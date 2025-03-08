<div align="center">
  <img src="https://github.com/lartpang/PyIRSTDMetrics/blob/main/images/logo.png?raw=true" alt="Logo" width="320" height="auto">
  </br>
  <h2>PyIRSTDMetrics: A simple and efficient implementation for the IRSTD performance analysis</h2>
  <img src="https://img.shields.io/pypi/v/pyirstdmetrics">
  <img src="https://img.shields.io/pypi/dm/pyirstdmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white">
  <img src="https://img.shields.io/github/last-commit/lartpang/PyIRSTDMetrics">
  <img src="https://img.shields.io/github/last-commit/lartpang/PyIRSTDMetrics">
  <img src="https://img.shields.io/github/release-date/lartpang/PyIRSTDMetrics">
  </br>
  <img src="https://api.star-history.com/svg?repos=lartpang/PyIRSTDMetrics&type=Date" alt="Star History Chart" width="600" height="auto">
</div>

## Introduction

A simple and efficient implementation for the IRSTD performance analysis.

- Based on `numpy`、`scikit-image` and `scipy`.
- Verification based on <https://github.com/XinyiYing/BasicIRSTD>
- The code structure is simple and easy to extend
- The code is lightweight and fast

Your improvements and suggestions are welcome.

### Supported Metrics

| Metric                        | Sample-based           | Whole-based | Related Class                                 | Level  |
| ----------------------------- | ---------------------- | ----------- | --------------------------------------------- | ------ |
| IoU                           | max,avg,adp,bin (nIoU) | bin (IoU)   | `CMMetrics`+`IOUHandler`                      | pixel  |
| F1                            | max,avg,adp,bin        | bin         | `CMMetrics`+`FmeasureHandler`                 | pixel  |
| Precision                     | max,avg,adp,bin        | bin         | `CMMetrics`+`PrecisionHandler`                | pixel  |
| Recall                        | max,avg,adp,bin        | bin         | `CMMetrics`+`RecallHandler`                   | pixel  |
| TPR                           | max,avg,adp,bin        | bin         | `CMMetrics`+`TPRHandler`                      | pixel  |
| FPR                           | max,avg,adp,bin        | bin         | `CMMetrics`+`FPRHandler`                      | pixel  |
| Pd/Fa                         |                        | ✔           | `MatchingBasedMetrics`+`DistanceOnlyMatching` | target |
| hIoU                          |                        | ✔           | `MatchingBasedMetrics`+`DistanceOnlyMatching` | hybrid |
| hIoU-based loc error analysis |                        | ✔           | `HierarchicalIoUBasedErrorAnalysis`           |        |
| hIoU-based seg error analysis |                        | ✔           | `HierarchicalIoUBasedErrorAnalysis`           |        |

As shown in `plot_average_metrics` of [examples/metric_recorder.py](./examples/metric_recorder.py):

- precision and recall sequences can be used to plot the PR curve.
- TPR and FPR sequences can be used to plot the ROC curve.

## Usage

The core files are in the folder `py_irstd_metrics`.

- **[Latest, but may be unstable]** Install from the source code: `pip install git+https://github.com/lartpang/PyIRSTDMetrics.git`
- **[More stable]** Install from PyPI: `pip install pyirstdmetrics`

### Examples

- [examples/test_metrics.py](./examples/test_metrics.py)
- [examples/metric_recorder.py](./examples/metric_recorder.py)

```text
@inproceedings{IRSTD-ACM-nIoU,
  title     = {Asymmetric Contextual Modulation for Infrared Small Target Detection},
  booktitle = WACV,
  author    = {Dai, Yimian and Wu, Yiquan and Zhou, Fei and Barnard, Kobus},
  year      = {2021},
  volume    = {},
  number    = {},
  pages     = {949-958},
  doi       = {10.1109/WACV48630.2021.00099},
  issn      = {2642-9381},
  month     = {Jan},
}
@article{IRSTD-DNANet-PdFa,
  title    = {Dense Nested Attention Network for Infrared Small Target Detection},
  author   = {Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, Zaiping and Li, Miao and An, Wei and Guo, Yulan},
  journal  = IEEE_J_IP,
  year     = {2023},
  volume   = {32},
  number   = {},
  pages    = {1745-1758},
  doi      = {10.1109/TIP.2022.3199107},
  issn     = {1941-0042},
  month    = {},
}
```

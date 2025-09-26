<div align="center">
  <img src="https://github.com/lartpang/PyIRSTDMetrics/blob/main/images/logo.png?raw=true" alt="Logo" width="320" height="auto">
  </br>
  <h2>PyIRSTDMetrics: A simple and efficient implementation for the IRSTD performance analysis</h2>
   <a href='https://arxiv.org/abs/2509.16888'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv'>
   </a>
  <img src="https://img.shields.io/pypi/v/pyirstdmetrics">
  <img src="https://img.shields.io/pypi/dm/pyirstdmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white">
  <img src="https://img.shields.io/github/last-commit/lartpang/PyIRSTDMetrics">
  <img src="https://img.shields.io/github/release-date/lartpang/PyIRSTDMetrics">
  </br>
</div>

> [!important]
>
> Our exploration in this field continues with PyIRSTDMetrics, a project born from the same core motivation as our previous project [PySODMetrics](https://github.com/lartpang/PySODMetrics/tree/main).
> ​​Think of them as twin initiatives: [PySODMetrics](https://github.com/lartpang/PySODMetrics/tree/main) maps the landscape of current evaluation, while its sibling (PyIRSTDMetrics) takes the next step to expand upon and rethink it.
> We'd love your star! 🌟

## News

- 2025-9-23: Our paper (*Rethinking Evaluation of Infrared Small Target Detection*) accepted by NeurIPS 2025 (D&B) has been released on the [arXiv (2509.16888)](https://arxiv.org/abs/2509.16888).

## Introduction

A simple and efficient implementation for the IRSTD performance analysis.

- Based on `numpy`、`scikit-image` and `scipy`.
- Verification based on <https://github.com/XinyiYing/BasicIRSTD>
- The code structure is simple and easy to extend
- The code is lightweight and fast

Your improvements and suggestions are welcome.

### Supported Metrics

| Metric                        | Sample-based           | Whole-based | Related Class                                                | Level  |
| ----------------------------- | ---------------------- | ----------- | ------------------------------------------------------------ | ------ |
| IoU                           | max,avg,adp,bin (nIoU) | bin (IoU)   | `CMMetrics`+`IOUHandler`                                     | pixel  |
| F1                            | max,avg,adp,bin        | bin         | `CMMetrics`+`FmeasureHandler`                                | pixel  |
| Precision                     | max,avg,adp,bin        | bin         | `CMMetrics`+`PrecisionHandler`                               | pixel  |
| Recall                        | max,avg,adp,bin        | bin         | `CMMetrics`+`RecallHandler`                                  | pixel  |
| TPR                           | max,avg,adp,bin        | bin         | `CMMetrics`+`TPRHandler`                                     | pixel  |
| FPR                           | max,avg,adp,bin        | bin         | `CMMetrics`+`FPRHandler`                                     | pixel  |
| Pd/Fa                         |                        | ✔           | `MatchingBasedMetrics`+`DistanceOnlyMatching`/`OPDCMatching` | target |
| hIoU                          |                        | ✔           | `MatchingBasedMetrics`+`OPDCMatching`                        | hybrid |
| hIoU-based loc error analysis |                        | ✔           | `HierarchicalIoUBasedErrorAnalysis`                          |        |
| hIoU-based seg error analysis |                        | ✔           | `HierarchicalIoUBasedErrorAnalysis`                          |        |

**NOTE**:

- If you want to align the original implementation, use `DistanceOnlyMatching`.
- If you want a more reasonable matching effect, use `OPDCMatching` we designed.
- hIoU is a new metric that balances both pixel-level and target-level performance analysis and we provide a detailed error analysis tool based on it.

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

## Thanks

- [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD) is a PyTorch-based open-source and easy-to-use toolbox for infrared small target detction (IRSTD).

## Citation

```text
@misc{HierarchicalIoUBasedAnalysis,
  title         = {Rethinking Evaluation of Infrared Small Target Detection},
  author        = {Youwei Pang and Xiaoqi Zhao and Lihe Zhang and Huchuan Lu and Georges El Fakhri and Xiaofeng Liu and Shijian Lu},
  year          = {2025},
  eprint        = {2509.16888},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
}
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
  title   = {Dense Nested Attention Network for Infrared Small Target Detection},
  author  = {Li, Boyang and Xiao, Chao and Wang, Longguang and Wang, Yingqian and Lin, Zaiping and Li, Miao and An, Wei and Guo, Yulan},
  journal = IEEE_J_IP,
  year    = {2023},
  volume  = {32},
  number  = {},
  pages   = {1745-1758},
  doi     = {10.1109/TIP.2022.3199107},
  issn    = {1941-0042},
  month   = {},
}
```

import abc
from typing import Dict

import numpy as np

from .utils import TYPE, divide_func


class _BaseHandler:
    def __init__(self, *, with_dynamic: bool = False, with_binary: bool = True, sample_based: bool = True):
        """
        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_binary (bool, optional): Record binary results for binary version.
            sample_based (bool, optional): Whether to average the metric of each sample or calculate
                the metric of the dataset. Defaults to True.
        """
        self.dynamic_results = [] if with_dynamic else None
        self.sample_based = sample_based
        if with_binary:
            if self.sample_based:
                self.binary_results = []
            else:
                self.binary_results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        else:
            self.binary_results = None

    @abc.abstractmethod
    def __call__(self, *args, **kwds):
        pass


class IoUHandler(_BaseHandler):
    """Intersection over Union
    Threat score (TS), critical success index (CSI), Jaccard index

    iou = tp / (tp + fp + fn)
    """

    def __call__(self, tp, fp, tn, fn):
        return divide_func(tp, tp + fp + fn)


class PrecisionHandler(_BaseHandler):
    """Precision

    precision = tp / (tp + fp)
    """

    def __call__(self, tp, fp, tn, fn):
        return divide_func(tp, tp + fp)


class RecallHandler(_BaseHandler):
    """True positive rate (TPR)/recall/sensitivity (SEN)/probability of detection/hit rate/power

    recall = tp / (tp + fn)
    """

    def __call__(self, tp, fp, tn, fn):
        return divide_func(tp, tp + fn)


TPRHandler = RecallHandler


class FPRHandler(_BaseHandler):
    """False positive rate (FPR), probability of false alarm, fall-out type I error

    FPR = fp / (fp + tn)
    """

    def __call__(self, tp, fp, tn, fn):
        return divide_func(fp, fp + tn)


class FmeasureHandler(_BaseHandler):
    """F-measure

    fmeasure = (beta + 1) * precision * recall / (beta * precision + recall)
    """

    def __init__(self, *, with_dynamic: bool, with_binary: bool = False, sample_based: bool = True, beta: float = 0.3):
        """
        Args:
            with_dynamic (bool, optional): Record dynamic results for max/avg/curve versions.
            with_binary (bool, optional): Record binary results for binary version.
            sample_based (bool, optional): Whether to average the metric of each sample or calculate
                the metric of the dataset. Defaults to True.
            beta (bool, optional): β^2 in F-measure. Defaults to 0.3.
        """
        super().__init__(with_dynamic=with_dynamic, with_binary=with_binary, sample_based=sample_based)

        self.beta = beta
        self.precision = PrecisionHandler(with_binary=False, with_dynamic=False)
        self.recall = RecallHandler(with_binary=False, with_dynamic=False)

    def __call__(self, tp, fp, tn, fn):
        p = self.precision(tp, fp, tn, fn)
        r = self.recall(tp, fp, tn, fn)
        return divide_func((self.beta + 1) * p * r, self.beta * p + r)


class CMMetrics:
    def __init__(self, num_bins=10, threshold=0.5, metric_handlers: dict = None):
        """Metrics based on Confusion Matrix.

        Args:
            metric_handlers (dict, optional): Handlers of different metrics. Defaults to None.
        """
        self.num_bins = num_bins
        self.threshold = threshold
        self._metric_handlers = metric_handlers if metric_handlers else {}

    def add_handler(self, handler_name, metric_handler):
        self._metric_handlers[handler_name] = metric_handler

    @staticmethod
    def get_statistics(binary: np.ndarray, mask: np.ndarray, FG: int, BG: int) -> dict:
        """Calculate the TP, FP, TN and FN based a adaptive threshold.

        Args:
            binary (np.ndarray[bool]): binary image
            mask (np.ndarray[bool]): binary mask
            FG (int): the number of foreground pixels in mask
            BG (int): the number of background pixels in mask

        Returns:
            dict: TP, FP, TN, FN
        """
        TP = np.count_nonzero(binary[mask])
        FP = np.count_nonzero(binary[~mask])
        FN = FG - TP
        TN = BG - FP
        return {"tp": TP, "fp": FP, "tn": TN, "fn": FN}

    def dynamically_binarizing(self, prob: np.ndarray, mask: np.ndarray, FG: int, BG: int) -> dict:
        """Calculate the corresponding TP, FP, TN and FNs when the threshold changes from 0 to 255.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary mask.
            FG (int): the number of foreground pixels in mask
            BG (int): the number of background pixels in mask

        Returns:
            dict: TPs, FPs, TNs, FNs
        """
        bins: np.ndarray = np.linspace(0, 1, self.num_bins)
        tp_hist, _ = np.histogram(prob[mask], bins=bins)  # the last bin is [1, 1+bin]
        fp_hist, _ = np.histogram(prob[~mask], bins=bins)

        tp_w_thrs = np.cumsum(np.flip(tp_hist))  # >= 1, >= (num_bins-1)*bin, ... >= 1*bin, >= 0
        fp_w_thrs = np.cumsum(np.flip(fp_hist))

        TPs = tp_w_thrs
        FPs = fp_w_thrs
        FNs = FG - TPs
        TNs = BG - FPs
        return {"tp": TPs, "fp": FPs, "tn": TNs, "fn": FNs}

    def update(self, prob: np.ndarray, mask: np.ndarray):
        """Statistics the metrics for the pair of prob and mask.

        Args:
            prob (np.ndarray[float]): grayscale prediction with values in 0~1.
            mask (np.ndarray[bool]): binary mask.

        Raises:
            ValueError: Please add your metric handler before using `update()`.
        """
        if not self._metric_handlers:
            raise ValueError("Please add your metric handler before using `update()`.")
        assert prob.shape == mask.shape, (prob.shape, mask.shape)
        assert 0 <= prob.min() <= prob.max() <= 1, (prob.dtype, prob.min(), prob.max())
        assert mask.dtype == bool, mask.dtype

        FG = np.count_nonzero(mask)  # FG=(TPs+FNs)
        BG = mask.size - FG  # BG=(TNs+FPs)

        dynamical_tpfptnfn = None
        binary_tpfptnfn = None
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None:
                if dynamical_tpfptnfn is None:
                    dynamical_tpfptnfn = self.dynamically_binarizing(prob=prob, mask=mask, FG=FG, BG=BG)
                handler.dynamic_results.append(handler(**dynamical_tpfptnfn))

            if handler.binary_results is not None:
                if binary_tpfptnfn is None:  # `prob > 0.5`: Simulating the effect of the `argmax` function.
                    binary_tpfptnfn = self.get_statistics(binary=prob > self.threshold, mask=mask, FG=FG, BG=BG)
                if handler.sample_based:
                    handler.binary_results.append(handler(**binary_tpfptnfn))
                else:
                    handler.binary_results["tp"] += binary_tpfptnfn["tp"]
                    handler.binary_results["fp"] += binary_tpfptnfn["fp"]
                    handler.binary_results["tn"] += binary_tpfptnfn["tn"]
                    handler.binary_results["fn"] += binary_tpfptnfn["fn"]

    def get(self) -> Dict[str, float]:
        """Return the results of the specific metric names.

        Returns:
            dict: All results corresponding to different metrics.
        """
        results = {}
        for handler_name, handler in self._metric_handlers.items():
            res = {}
            if handler.dynamic_results is not None:
                res["dynamic"] = np.mean(np.array(handler.dynamic_results, dtype=TYPE), axis=0)
            if handler.binary_results is not None:
                if handler.sample_based:
                    res["binary"] = np.mean(np.array(handler.binary_results, dtype=TYPE))
                else:
                    # NOTE: use `np.mean` to simplify output format (`array(123)` -> `123`)
                    res["binary"] = np.mean(handler(**handler.binary_results))
            results[handler_name] = res
        return results

"""
saliency base class and basic gradient tools for different methods
from:
    https://github.com/PAIR-code/saliency/blob/master/saliency/base.py
    https://github.com/PAIR-code/saliency/blob/master/saliency/integrated_gradients.py

NOTE: totally not checked
"""

import tensorflow
if tensorflow.__version__.startswith("1."):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from skimage import segmentation
from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.transform import resize
from numbers import Real
from typing import Optional, List

from ..common import ArrayLike
from .io import normalize_image


__all__ = ["XRAI"]


class SaliencyMask(object):
    """
    Base class for saliency masks. Alone, this class doesn't do anything.
    """
    def __init__(self,
                graph:tf.Graph,
                session:tf.Session,
                y:tf.Tensor,
                x:tf.Tensor):
        """Constructs a SaliencyMask by computing dy/dx.
        Args:
        graph: The TensorFlow graph to evaluate masks on.
        session: The current TensorFlow session.
        y: The output tensor to compute the SaliencyMask against. This tensor
            should be of size 1.
        x: The input tensor to compute the SaliencyMask against. The outer
            dimension should be the batch size.
        """

        # y must be of size one, otherwise the gradient we get from tf.gradients
        # will be summed over all ys.
        size = 1
        for shape in y.shape:
            size *= shape
        assert size == 1

        self.graph = graph
        self.session = session
        self.y = y
        self.x = x

    def GetMask(self, x_value:np.ndarray, feed_dict:Optional[dict]=None):
        """Returns an unsmoothed mask.
        Args:
        x_value: Input value, not batched.
        feed_dict: (Optional) feed dictionary to pass to the session.run call.
        """
        raise NotImplementedError('A derived class should implemented GetMask()')

    def GetSmoothedMask(self,
                        x_value:np.ndarray,
                        feed_dict:Optional[dict]=None,
                        stdev_spread:float=.15,
                        nsamples:int=25,
                        magnitude:bool=True,
                        **kwargs) -> np.ndarray:
        """
        Returns a mask that is smoothed with the SmoothGrad method.

        Parameters:
        ----------
        x_value: ndarray,
            input value, not batched.
        feed_dict: dict, optional,
            feed dictionary to pass to the session.run call.
        stdev_spread: float, default .15,
            amount of noise to add to the input, as fraction of the total spread (x_max - x_min)
        nsamples: int, default 25,
            number of samples to average across to get the smooth gradient.
        magnitude: bool, default True,
            if true, computes the sum of squares of gradients instead of just the sum
        """
        stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

        total_gradients = np.zeros_like(x_value)
        for _ in range(nsamples):
            noise = np.random.normal(0, stdev, x_value.shape)
        x_plus_noise = x_value + noise
        grad = self.GetMask(x_plus_noise, feed_dict, **kwargs)
        if magnitude:
            total_gradients += (grad * grad)
        else:
            total_gradients += grad

        return total_gradients / nsamples


class GradientSaliency(SaliencyMask):
    r"""A SaliencyMask class that computes saliency masks with a gradient."""

    def __init__(self,
                graph:tf.Graph,
                session:tf.Session,
                y:tf.Tensor,
                x:tf.Tensor):
        super().__init__(graph, session, y, x)
        self.gradients_node = tf.gradients(y, x)[0]

    def GetMask(self,
                x_value:np.ndarray,
                feed_dict:Optional[dict]=None) -> np.ndarray:
        """
        Returns a vanilla gradient mask.

        Parameters:
        ----------
        x_value: ndarray,
            input value, not batched.
        feed_dict: dict, optional,
            feed dictionary to pass to the session.run call.
        """
        feed_dict = feed_dict or {}
        feed_dict[self.x] = [x_value]
        return self.session.run(self.gradients_node, feed_dict=feed_dict)[0]


class IntegratedGradients(GradientSaliency):
    """
    A SaliencyMask class that implements the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """

    def GetMask(self,
                x_value:np.ndarray,
                feed_dict:Optional[dict]=None,
                x_baseline:Optional[np.ndarray]=None,
                x_steps:int=25) -> np.ndarray:
        """
        Returns a integrated gradients mask.

        Parameters:
        ----------
        x_value: ndarray,
            input ndarray
        x_baseline: ndarray, default 0.,
            baseline value used in integration
        x_steps: int, default 25,
            number of integrated steps between baseline and x
        """
        if x_baseline is None:
            x_baseline = np.zeros_like(x_value)

        assert x_baseline.shape == x_value.shape

        x_diff = x_value - x_baseline

        total_gradients = np.zeros_like(x_value)

        for alpha in np.linspace(0, 1, x_steps):
            x_step = x_baseline + alpha * x_diff

        total_gradients += super().GetMask(
            x_step, feed_dict)

        return total_gradients * x_diff / x_steps


_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _get_segments_felzenszwalb(img:np.ndarray,
                               resize_image:bool=True,
                               scale_range:Optional[ArrayLike]=None,
                               dilation_rad:int=5):
    """
    Compute image segments based on Felzenszwalb's algorithm.
    
    Parameters:
    -----------
    img: ndarray,
        input image.
    resize_image: bool, default True,
        if True, the image is resized to 224,224 for the segmentation purposes.
        The resulting segments are rescaled back to match the original image size. It is done for consistency w.r.t. segmentation parameter range.
    scale_range: array_like, optional,
        range of image values to use for segmentation algorithm.
        Segmentation algorithm is sensitive to the input image values, therefore we need to be consistent with the range for all images.
        If None is passed, the range is scaled to [-1.0, 1.0]
    dilation_rad:
        sets how much each segment is dilated to include edges, larger values cause more blobby segments, smaller values get sharper areas. Defaults to 5.
    
    Returns:
    --------
    masks: list,
        a list of boolean masks as np.ndarrays if size HxW for img size of HxWxC.

    References:
    -----------
    [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and Huttenlocher, D.P. International Journal of Computer Vision, 2004
    """

    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = img.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        normalized_img = normalize_image(img, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        normalized_img = normalize_image(img, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(normalized_img,
                                            scale=scale,
                                            sigma=sigma,
                                            min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE)
        if resize_image:
            seg = resize(seg,
                        original_shape,
                        order=0,
                        preserve_range=True,
                        mode='constant',
                        anti_aliasing=False).astype(np.int)
        segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        selem = disk(dilation_rad)
        masks = [dilation(mask, selem=selem) for mask in masks]
    return masks


def _attr_aggregation_max(attr:np.ndarray, axis=-1) -> np.ndarray:
    """
    """
    return attr.max(axis=axis)


def _gain_density(mask1:np.ndarray, attr:np.ndarray, mask2:Optional[np.ndarray]=None) -> float:
    """
    Compute the attr density over mask1. If mask2 is specified, compute density for mask1 \ mask2
    """
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _get_diff_mask(add_mask:np.ndarray, base_mask:np.ndarray) -> np.ndarray:
    """
    """
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask:np.ndarray, base_mask:np.ndarray) -> Real:
    """
    """
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs:List[np.ndarray]) -> np.ndarray:
    """
    """
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


class XRAIParameters(object):
    """
    TODO: consider using a namedtuple instead?
    """
    def __init__(self,
                steps:int=100,
                area_threshold:Real=1.0,
                return_baseline_predictions:bool=False,
                return_ig_attributions:bool=False,
                return_xrai_segments:bool=False,
                flatten_xrai_segments:bool=True,
                algorithm:str='full',
                **kwargs):
        # TODO(tolgab) add return_ig_for_every_step functionality

        # Number of steps to use for calculating the Integrated Gradients
        # attribution. The higher the number of steps the higher is the precision
        # but lower the performance. (see also XRAIOutput.error).
        self.steps = steps
        # The fraction of the image area that XRAI should calculate the segments
        # for. All segments that exceed that threshold will be merged into a single
        # segment. The parameter is used to accelerate the XRAI computation if the
        # caller is only interested in the top fraction of segments, e.g. 20%. The
        # value should be in the [0.0, 1.0] range, where 1.0 means that all segments
        # should be returned (slowest). Fast algorithm ignores this setting.
        self.area_threshold = area_threshold
        # TODO(tolgab) Enable return_baseline_predictions
        # If set to True returns predictions for the baselines as float32 [B] array,
        # where B is the number of baselines. (see XraiOutput.baseline_predictions).
        self.return_baseline_predictions = kwargs.get("return_baseline_predictions", False)
        # If set to True, the XRAI output returns Integrated Gradients attributions
        # for every baseline. (see XraiOutput.ig_attribution)
        self.return_ig_attributions = return_ig_attributions
        # If set to True the XRAI output returns XRAI segments in the order of their
        # importance. This parameter works in conjunction with the
        # flatten_xrai_sements parameter. (see also XraiOutput.segments)
        self.return_xrai_segments = return_xrai_segments
        # If set to True, the XRAI segments are returned as an integer array with
        # the same dimensions as the input (excluding color channels). The elements
        # of the array are set to values from the [1,N] range, where 1 is the most
        # important segment and N is the least important segment. If
        # flatten_xrai_sements is set to False, the segments are returned as a
        # boolean array, where the first dimension has size N. The [0, ...] mask is
        # the most important and the [N-1, ...] mask is the least important. This
        # parameter has an effect only if return_xrai_segments is set to True.
        self.flatten_xrai_segments = flatten_xrai_segments
        # Specifies a flavor of the XRAI algorithm. full - executes slower but more
        # precise XRAI algorithm. fast - executes faster but less precise XRAI
        # algorithm.
        self.return_ig_for_every_step = kwargs.get("return_ig_for_every_step", False)
        self.algorithm = algorithm
        # EXPERIMENTAL - Contains experimental parameters that may change in future.
        self.experimental_params = {'min_pixel_diff': 50}


class XRAIOutput(object):
    """
    TODO: consider using a namedtuple instead?
    """
    def __init__(self, attribution_mask:np.ndarray):
        # The saliency mask of individual input features. For an [HxWx3] image, the
        # returned attribution is [H,W,1] float32 array. Where HxW are the
        # dimensions of the image.
        self.attribution_mask = attribution_mask
        # Baselines that were used for IG calculation. The shape is [B,H,W,C], where
        # B is the number of baselines, HxWxC are the image dimensions.
        self.baselines = None
        # TODO(tolgab) add return IG error functionality from XRAI API
        # The average error of the IG attributions as a percentage. The error can be
        # decreased by increasing the number of steps (see XraiParameters.steps).
        # self.error = None
        # TODO(tolgab) add return baseline predictions functionality from XRAI API
        # Predictions for the baselines that were used for the calculation of IG
        # attributions. The value is set only when
        # XraiParameters.return_baseline_predictions is set to True.
        # self.baseline_predictions = None
        # IG attributions for individual baselines. The value is set only when
        # XraiParameters.ig_attributions is set to True. For the dimensions of the
        # output see XraiParameters.return_ig_for_every _step.
        self.ig_attribution = None
        # The result of the XRAI segmentation. The value is set only when
        # XraiParameters.return_xrai_segments is set to True. For the dimensions of
        # the output see XraiParameters.flatten_xrai_segments.
        self.segments = None


class XRAI(SaliencyMask):
    """
    """
    def __init__(self, graph:tf.Graph, session:tf.Session, y:tf.Tensor, x:tf.Tensor):
        super(XRAI, self).__init__(graph, session, y, x)
        # Initialize integrated gradients.
        self._integrated_gradients = IntegratedGradients(graph, session, y, x)

    def _get_integrated_gradients(self, img:np.ndarray, feed_dict:dict, baselines:List[np.ndarray], steps:int) -> List[np.ndarray]:
        """ Takes mean of attributions from all baselines
        """
        grads = []
        for baseline in baselines:
            grads.append(
                self._integrated_gradients.GetMask(img,
                                                    feed_dict=feed_dict,
                                                    x_baseline=baseline,
                                                    x_steps=steps))
        return grads

    def _make_baselines(self, x_value:np.ndarray, x_baselines:List[np.ndarray]) -> List[np.ndarray]:
        # If baseline is not provided default to img min and max values
        if x_baselines is None:
            x_baselines = []
            x_baselines.append(np.min(x_value) * np.ones_like(x_value))
            x_baselines.append(np.max(x_value) * np.ones_like(x_value))
        else:
            for baseline in x_baselines:
                if baseline.shape != x_value.shape:
                    raise ValueError(
                        "Baseline size {} does not match input size {}".format(
                            baseline.shape, x_value.shape))
        return x_baselines

    def _predict(self, x:tf.Tensor):
        raise NotImplementedError

    def GetMask(self,
                x_value:np.ndarray,
                feed_dict:Optional[dict]=None,
                baselines:Optional[List[np.ndarray]]=None,
                segments:List[np.ndarray]=None,
                base_attribution:Optional[np.ndarray]=None,
                extra_parameters:Optional[XRAIParameters]=None) -> np.ndarray:
        """
        Applies XRAI method on an input image and returns the result saliency heatmap.

        Parameters:
        -----------
        x_value: ndarray,
            input value, not batched
        feed_dict: dict, optional,
            feed dictionary to pass to the TF session.run() call
        baselines: list, optional,
            a list of baselines to use for calculating Integrated Gradients attribution.
            Every baseline in the list should have the same dimensions as the input. If the value is not set then the algorithm will make the best effort to select default baselines
        segments: list, optional,
            the list of precalculated image segments that should be passed to XRAI.
            Each element of the list is an [N,M] boolean array, where NxM are the image dimensions. Each elemeent on the list contains exactly the mask that corresponds to one segment. If the value is None, Felzenszwalb's segmentation algorithm will be applied
        base_attribution: ndarray, optional,
            an optional pre-calculated base attribution that XRAI should use.
            The shape of the parameter should match the shape of `x_value`. If the value is None, the method calculates Integrated Gradients attribution and uses it.
        extra_parameters: XRAIParameters, optional,
            an XRAIParameters object that specifies additional parameters for the XRAI saliency method.
            If it is None, an XRAIParameters object will be created with default parameters. See `XRAIParameters` for more details.
        
        Raises:
        -------
        ValueError:
            If algorithm type is unknown (not full or fast).
            If the shape of `base_attribution` dosn't match the shape of `x_value`
        
        Returns:
        --------
        ndarray: a numpy array that contains the saliency heatmap.
        
        TODO(tolgab) Add output_selector functionality from XRAI API doc
        """
        results = self.GetMaskWithDetails(x_value,
                                        feed_dict=feed_dict,
                                        baselines=baselines,
                                        segments=segments,
                                        base_attribution=base_attribution,
                                        extra_parameters=extra_parameters)
        return results.attribution_mask

    def GetMaskWithDetails(self,
                            x_value:np.ndarray,
                            feed_dict:Optional[dict]=None,
                            baselines:Optional[List[np.ndarray]]=None,
                            segments:List[np.ndarray]=None,
                            base_attribution:Optional[np.ndarray]=None,
                            extra_parameters:Optional[XRAIParameters]=None) -> XRAIOutput:
        """
        Applies XRAI method on an input image and returns the result saliency heatmap along with other detailed information.

        Parameters:
        ----------
        x_value: ndarray,
            input value, not batched.
        feed_dict: dict, optional,
            feed dictionary to pass to the TF session.run() call
        baselines: list, optional,
            a list of baselines to use for calculating Integrated Gradients attribution.
            Every baseline in the list should have the same dimensions as the input. If the value is not set then the algorithm will make the best effort to select default baselines.
        segments:
            the list of precalculated image segments that should be passed to XRAI.
            Each element of the list is an [N,M] boolean array, where NxM are the image dimensions. Each elemeent on the list contains exactly the mask that corresponds to one segment. If the value is None, Felzenszwalb's segmentation algorithm will be applied.
        base_attribution:
            an optional pre-calculated base attribution that XRAI should use.
            The shape of the parameter should match the shape of `x_value`. If the value is None, the method calculates Integrated Gradients attribution and uses it.
        extra_parameters:
            an XRAIParameters object that specifies additional parameters for the XRAI saliency method.
            If it is None, an XRAIParameters object will be created with default parameters. See `XRAIParameters` for more details.
        
        Raises:
        -------
        ValueError:
            If algorithm type is unknown (not full or fast).
            If the shape of `base_attribution` dosn't match the shape of `x_value`
        
        Returns:
        --------
        XRAIOutput: an object that contains the output of the XRAI algorithm.

        TODO(tolgab) Add output_selector functionality from XRAI API doc
        """
        if extra_parameters is None:
            extra_parameters = XRAIParameters()

        # Check the shape of base_attribution.
        if base_attribution is not None:
            _ba = np.array(base_attribution)
            if _ba.shape != x_value.shape:
                raise ValueError(
                'The base attribution shape should be the same as the shape of '
                '`x_value`. Expected {}, got {}'.format(
                    x_value.shape, _ba.shape))

        # Calculate IG attribution if not provided by the caller.
        if base_attribution is None:
            # _logger.info("Computing IG...")
            x_baselines = self._make_baselines(x_value, baselines)

            attrs = self._get_integrated_gradients(x_value,
                                                    feed_dict=feed_dict,
                                                    baselines=x_baselines,
                                                    steps=extra_parameters.steps)
            # Merge attributions from different baselines.
            attr = np.mean(attrs, axis=0)
        else:
            x_baselines = None
            attrs = _ba
            attr = _ba

        # Merge attribution channels for XRAI input
        attr = _attr_aggregation_max(attr)

        # _logger.info("Done with IG. Computing XRAI...")
        if segments is not None:
            segs = segments
        else:
            segs = _get_segments_felzenszwalb(x_value)

        if extra_parameters.algorithm == 'full':
            attr_map, attr_data = self._xrai(
                attr=attr,
                segs=segs,
                area_perc_th=extra_parameters.area_threshold,
                min_pixel_diff=extra_parameters.experimental_params['min_pixel_diff'],
                gain_fun=_gain_density,
                integer_segments=extra_parameters.flatten_xrai_segments)
        elif extra_parameters.algorithm == 'fast':
            attr_map, attr_data = self._xrai_fast(
                attr=attr,
                segs=segs,
                min_pixel_diff=extra_parameters.experimental_params['min_pixel_diff'],
                gain_fun=_gain_density,
                integer_segments=extra_parameters.flatten_xrai_segments)
        else:
            raise ValueError('Unknown algorithm type: {}'.format(
                extra_parameters.algorithm))

        results = XRAIOutput(attr_map)
        results.baselines = x_baselines
        if extra_parameters.return_xrai_segments:
            results.segments = attr_data
        
        # TODO(tolgab) Enable return_baseline_predictions
        # if extra_parameters.return_baseline_predictions:
        #   baseline_predictions = []
        #   for baseline in x_baselines:
        #     baseline_predictions.append(self._predict(baseline))
        #   results.baseline_predictions = baseline_predictions
    
        if extra_parameters.return_ig_attributions:
            results.ig_attribution = attrs
        return results

    @staticmethod
    def _xrai(attr:np.ndarray,
            segs:List[np.ndarray],
            gain_fun:callable=_gain_density,
            area_perc_th:float=1.0,
            min_pixel_diff:int=50,
            integer_segments:bool=True) -> tuple:
        """
        Run XRAI saliency given attributions and segments.

        Parameters:
        -----------
        attr: ndarray,
            source attributions for XRAI.
            XRAI attributions will be same size as the input attr.
        segs: list,
            input segments as a list of boolean masks.
            XRAI uses these to compute attribution sums.
        gain_fun: callable, default `_gain_density`,
            the function that computes XRAI area attribution from source attributions
        area_perc_th: float, default 1.0,
            the saliency map is computed to cover area_perc_th of the image. Lower values will run faster, but produce uncomputed areas in the image that will be filled to satisfy completeness
        min_pixel_diff: int, default 50,
            Do not consider masks that have difference less than this number compared to the current mask. Set it to 1 to remove masks that completely overlap with the current mask.
        integer_segments: bool, default True,
            see `XRAIParameters`

        Returns:
        --------
        tuple: saliency heatmap and list of masks or an integer image with
            area ranks depending on the parameter integer_segments.
        """
        output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

        n_masks = len(segs)
        current_area_perc = 0.0
        current_mask = np.zeros(attr.shape, dtype=bool)

        masks_trace = []
        remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

        added_masks_cnt = 1
        # While the mask area is less than area_th and remaining_masks is not empty
        while current_area_perc <= area_perc_th:
            best_gain = -np.inf
            best_key = None
            remove_key_queue = []
            for mask_key in remaining_masks:
                mask = remaining_masks[mask_key]
                # If mask does not add more than min_pixel_diff to current mask, remove
                mask_pixel_diff = _get_diff_cnt(mask, current_mask)
                if mask_pixel_diff < min_pixel_diff:
                    remove_key_queue.append(mask_key)
                    # if _logger.isEnabledFor(logging.DEBUG):
                    #     _logger.debug("Skipping mask with pixel difference: {:.3g},".format(
                    #         mask_pixel_diff))
                    continue
                gain = gain_fun(mask, attr, mask2=current_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_key = mask_key
            for key in remove_key_queue:
                del remaining_masks[key]
            if len(remaining_masks) == 0:
                break
            added_mask = remaining_masks[best_key]
            mask_diff = _get_diff_mask(added_mask, current_mask)
            masks_trace.append((mask_diff, best_gain))

            current_mask = np.logical_or(current_mask, added_mask)
            current_area_perc = np.mean(current_mask)
            output_attr[mask_diff] = best_gain
            del remaining_masks[best_key]  # delete used key
            # if _logger.isEnabledFor(logging.DEBUG):
            #     current_attr_sum = np.sum(attr[current_mask])
            #     _logger.debug(
            #         "{} of {} masks added,"
            #         "attr_sum: {}, area: {:.3g}/{:.3g}, {} remaining masks".format(
            #             added_masks_cnt, n_masks, current_attr_sum, current_area_perc,
            #             area_perc_th, len(remaining_masks)))
            added_masks_cnt += 1

        uncomputed_mask = output_attr == -np.inf
        # Assign the uncomputed areas a value such that sum is same as ig
        output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
        masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
        if np.any(uncomputed_mask):
            masks_trace.append(uncomputed_mask)
        if integer_segments:
            attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace

    @staticmethod
    def _xrai_fast(attr:np.ndarray,
                    segs:List[np.ndarray],
                    gain_fun:callable=_gain_density,
                    area_perc_th:float=1.0,
                    min_pixel_diff:int=50,
                    integer_segments:bool=True) -> tuple:
        """
        Run approximate XRAI saliency given attributions and segments.
        This version does not consider mask overlap during importance ranking, significantly speeding up the algorithm for less accurate results.

        Parameters:
        -----------
        attr: ndarray,
            source attributions for XRAI.
            XRAI attributions will be same size as the input attr.
        segs: list,
            input segments as a list of boolean masks.
            XRAI uses these to compute attribution sums.
        gain_fun: callable, default `_gain_density`,
            the function that computes XRAI area attribution from source attributions
        area_perc_th: float, default 1.0,
            the saliency map is computed to cover area_perc_th of the image. Lower values will run faster, but produce uncomputed areas in the image that will be filled to satisfy completeness
        min_pixel_diff: int, default 50,
            Do not consider masks that have difference less than this number compared to the current mask. Set it to 1 to remove masks that completely overlap with the current mask.
        integer_segments: bool, default True,
            see `XRAIParameters`

        Returns:
        --------
        tuple: saliency heatmap and list of masks or an integer image with
            area ranks depending on the parameter integer_segments.
        """
        output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

        n_masks = len(segs)
        current_mask = np.zeros(attr.shape, dtype=bool)

        masks_trace = []

        # Sort all masks based on gain, ignore overlaps
        seg_attrs = [gain_fun(seg_mask, attr) for seg_mask in segs]
        segs, seg_attrs = list(
            zip(*sorted(zip(segs, seg_attrs), key=lambda x: -x[1])))

        for i, added_mask in enumerate(segs):
            mask_diff = _get_diff_mask(added_mask, current_mask)
            # If mask does not add more than min_pixel_diff to current mask, skip
            mask_pixel_diff = _get_diff_cnt(added_mask, current_mask)
            if mask_pixel_diff < min_pixel_diff:
                # if _logger.isEnabledFor(logging.DEBUG):
                #     _logger.debug("Skipping mask with pixel difference: {:.3g},".format(mask_pixel_diff))
                continue
            mask_gain = gain_fun(mask_diff, attr)
            masks_trace.append((mask_diff, mask_gain))
            output_attr[mask_diff] = mask_gain
            current_mask = np.logical_or(current_mask, added_mask)
            # if _logger.isEnabledFor(logging.DEBUG):
            #     current_attr_sum = np.sum(attr[current_mask])
            #     current_area_perc = np.mean(current_mask)
            #     _logger.debug("{} of {} masks processed,"
            #                 "attr_sum: {}, area: {:.3g}/{:.3g}".format(
            #                     i + 1, n_masks, current_attr_sum, current_area_perc,
            #                     area_perc_th))
        uncomputed_mask = output_attr == -np.inf
        # Assign the uncomputed areas a value such that sum is same as ig
        output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
        masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
        if np.any(uncomputed_mask):
            masks_trace.append(uncomputed_mask)
        if integer_segments:
            attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
            for i, mask in enumerate(masks_trace):
                attr_ranks[mask] = i + 1
            return output_attr, attr_ranks
        else:
            return output_attr, masks_trace

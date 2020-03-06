# -*- coding: utf-8 -*-
"""
Grad-CAM and Grad-CAM++ for visualization of CNN models

TODO: implement a torch version
"""
from keras.layers.core import Layer
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
from typing import Union, Optional, Tuple, List
from numbers import Real

from ..common import ArrayLike, modulo


__all__ = [
    "preprocess_img",
    "grad_cam_keras",
    "grad_cam_naive_keras",
    "grad_cam_plusplus_keras",
]


def _target_category_loss(x:Union[tf.Tensor, K.variable], category_index:int, nb_classes:int):
    """
    """
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def _target_category_loss_output_shape(input_shape:tuple) -> tuple:
    """
    """
    return input_shape


def _normalize(x:Union[tf.Tensor, K.variable]) -> Union[tf.Tensor, K.variable]:
    """
    utility function to normalize a tensor by its L2 norm
    """
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def _compute_gradients(tensor:Union[tf.Tensor, K.variable], var_list:List[Union[tf.Tensor, K.variable]]) -> list:
    """
    """
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


def preprocess_img(img:np.ndarray, rescale:Real=1/255, resize:ArrayLike=(299,299), dtype:type=np.float32) -> np.ndarray:
    """
    """
    processed = cv2.resize(img, resize)
    processed = (processed * rescale).astype(dtype)
    return np.array([processed])


def grad_cam_naive_keras(input_model:Model, img:np.ndarray, target_layer:Union[str, int, Layer], prediction_layer:Optional[Union[str, int, Layer]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """ finished, fully checked,

    Grad-CAM, the naive (local) one

    Parameters:
    -----------
    input_model: keras `Model`,
        the model to compute Grad-CAM
    img: ndarray,
        the image to compute Grad-CAM, in RGB
    target_layer: str, or int, or keras `Layer`,
        to write
    prediction_layer: str, or int, or keras `Layer`, optional,
        to write
    kwargs: dict, with allowed items:
        "preprocess_func": callable, default `preprocess_img`,
            function to preprocess `img` to valid format to call `input_model.predict`,
        "preprocess_func_kw": dict, default "{}",
            key word arguments for "preprocess_func"
        "color_map": int, default "cv2.COLORMAP_JET"
            cv2 color map for obtaining `cam` from `heatmap`
        "base_model": str, default "",
            name of the base model of `input_model`,
        "d_class_index_to_name": dict, default {},
            the dict mapping the predicted class index of `img` to class name
        "correct_prediction": int, default None,
            the user-input 'correct' prediction
        "fast": bool, default False,
            if is True, and "correct_prediction" is given,
            then `input_model.predict` will not be called on the processed image
        "verbose": int, or list of str,
            default 0
        "kw_plot": dict, default {},
            key word arguments for plot,
            used only when "verbose" >= 3 or "plot" in "verbose"

    Returns:
    --------
    cam, img_with_cam, heatmap, pred_class_index, pred_class_name: tuple,
        of type ndarray, ndarray, ndarray, int, str resp.,
        as the names imply

    TODO:
    1. use `base_model` and `_GRAD_CAM_RECOMMENDATIONS`

    1 deprecated, will be used in tcm.services.application.sseg

    References:
    -----------
    [1] Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization, https://arxiv.org/abs/1610.02391
    [2] https://github.com/jacobgil/keras-grad-cam
    [3] https://github.com/adityac94/Grad_CAM_plus_plus
    [4] https://en.wikipedia.org/wiki/Taylor_series#Taylor_series_in_several_variables

    Remarks:
    --------
    Grad-CAM is mathematically essentially Taylor series in several variables,
    using only the linear part as approximation:
    .. math:: (to write)
    """
    img_preprocess_func = kwargs.get("preprocess_func", preprocess_img)
    img_preprocess_func_kw = kwargs.get("preprocess_func_kw", {})
    cam_color_map = kwargs.get("color_map", cv2.COLORMAP_JET)
    base_model = kwargs.get("base_model", "")
    base_model = (''.join([c for c in list(base_model) if (c.isdigit() or c.isalpha())])).lower()
    d_class_index_to_name = kwargs.get("d_class_index_to_name", {})
    correct_prediction = kwargs.get("correct_prediction", None)
    fast = kwargs.get("fast", False) and (correct_prediction is not None)
    verbose = kwargs.get("verbose", 0)
    if isinstance(verbose, int):
        if verbose == 0:
            _verbose = []
        elif verbose == 1:
            _verbose = ['brief']
        elif verbose == 2:
            _verbose = ['brief', 'detail']
        elif verbose >= 3:
            _verbose = ['brief', 'detail', 'plot']
    else:
        _verbose = verbose
    
    preprocessed_img = img_preprocess_func(img, **img_preprocess_func_kw)

    if 'brief' in _verbose:
        print("preprocessed_img.shape =", preprocessed_img.shape)
        print("preprocessed_img.dtype =", preprocessed_img.dtype)
    
    if not fast:
        predictions = input_model.predict(preprocessed_img)[0]
        pred_class_index = np.argmax(predictions)
    else:
        pred_class_index = correct_prediction

    pred_class_name = d_class_index_to_name.get(pred_class_index, "")
    pred_class_name = pred_class_name or d_class_index_to_name.get(str(pred_class_index), "")

    nb_classes = input_model.output_shape[-1]
    
    if 'brief' in _verbose:
        print("nb_classes =", nb_classes)
        if not fast:
            print("predictions =", predictions)
        else:
            print("prediction skipped")
            print("in the following, pred_* are given by the kwargs `correct_prediction`")
        print("pred_class_index =", pred_class_index)
        print("pred_class_name =", pred_class_name)

    if correct_prediction is not None:
        pred_class_index = correct_prediction
        
    # target_layer = lambda x: _target_category_loss(x, pred_class_index, nb_classes)
    # x = Lambda(target_layer, output_shape = _target_category_loss_output_shape)(input_model.output)

    if prediction_layer is None:
        pred_layer = input_model.layers[-1]
        # pred_output = input_model.output[:, pred_class_index]
        pl_name = pred_layer.name  # for verbose printing
        pl_idx = -1  # for verbose printing
    elif isinstance(prediction_layer, str):
        pred_layer = [l for l in input_model.layers if l.name == prediction_layer][0]
        pl_name = prediction_layer
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    elif isinstance(prediction_layer, int):
        pred_layer = input_model.layers[prediction_layer]
        pl_name = pred_layer.name
        pl_idx = prediction_layer
    elif isinstance(prediction_layer, Layer):
        if prediction_layer not in input_model.layers:
            raise ValueError("`prediction_layer` should be a layer of `input_model`!")
        pred_layer = prediction_layer
        pl_name = pred_layer.name
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)

    pred_output = pred_layer.output[:, pred_class_index]
    
    # loss = K.sum(input_model.output)
    if isinstance(target_layer, str):
        tar_layer = [l for l in input_model.layers if l.name == target_layer][0]
        tl_name = target_layer
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)
    elif isinstance(target_layer, int):
        tar_layer = input_model.layers[target_layer]
        tl_name = tar_layer.name
        tl_idx = target_layer
    elif isinstance(target_layer, Layer):
        if target_layer not in input_model.layers:
            raise ValueError("`target_layer` should be a layer of `input_model`!")
        tar_layer = target_layer
        tl_name = tar_layer.name
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)

    target_output = tar_layer.output

    if 'brief' in _verbose:
        print("prediction_layer is {}, the {}-th layer of `input_model`".format(pl_name, pl_idx))
        print("target_layer is {}, the {}-th layer of `input_model`".format(tl_name, tl_idx))
        layers_tot_num = len(input_model.layers)
        print("distance of the two layers is {}".format(modulo(pl_idx,layers_tot_num) - modulo(tl_idx,layers_tot_num)))

    grads = _normalize(_compute_gradients(pred_output, [target_output])[0])
    # print('grads =', grads)
    gradient_function = K.function([input_model.input], [target_output, grads])

    output, grads_val = gradient_function([preprocessed_img])
    # print('output.shape =', output.shape)
    if 'detail' in _verbose:
        print('-'*100)
        print('output =', output)
        print('grads_val =', grads_val)
        print('-'*100)
    # output, grads_val = output[0, :], grads_val[0, :, :, :]
    output, grads_val = output[0, ...], grads_val[0, ...]  # input is a single image
    # print('output.shape =', output.shape)

    cam = np.sum(output * grads_val, axis=-1)
    
    cam = cv2.resize(cam, img.shape[:2][::-1])
    cam = np.maximum(cam, 0)  # ReLU in eq (2) in Ref.[1]
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cam_color_map)
    cam = cam[...,::-1]  # from BGR to RGB
    # print('cam.shape =', cam.shape)
    img_with_cam = np.float32(cam) + np.float32(img)
    img_with_cam = (255 * img_with_cam / np.max(img_with_cam)).astype(np.uint8)

    if 'plot' in _verbose:
        import matplotlib.pyplot as plt
        kw_plot = kwargs.get("kw_plot", {})
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=kw_plot.get('figsize', (18,5)))
        ax1.imshow(img)
        ax2.imshow(cam)
        ax3.imshow(img_with_cam)
        if 'title' in kw_plot:
            fig.suptitle(kw_plot['title'], fontsize=kw_plot.get('fontsize', 16))
        plt.show()

    return cam, img_with_cam, heatmap, pred_class_index, pred_class_name


_GRAD_CAM_RECOMMENDATIONS = {
    "xception": {
        "target_layer": "add_12",
        "prediction_layer": -1,
    },
    "inceptionv4": {},
    "inceptionresnetv2": {},
}

def grad_cam_keras(input_model:Model, img:np.ndarray, target_layer:Union[str, int, Layer], prediction_layer:Optional[Union[str, int, Layer]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """ finished, fully checked,

    Grad-CAM

    Parameters:
    -----------
    input_model: keras `Model`,
        the model to compute Grad-CAM
    img: ndarray,
        the image to compute Grad-CAM, in RGB
    target_layer: str, or int, or keras `Layer`,
        to write
    prediction_layer: str, or int, or keras `Layer`, optional,
        to write
    kwargs: dict, with allowed items:
        "preprocess_func": callable, default `preprocess_img`,
            function to preprocess `img` to valid format to call `input_model.predict`,
        "preprocess_func_kw": dict, default "{}",
            key word arguments for "preprocess_func"
        "color_map": int, default "cv2.COLORMAP_JET"
            cv2 color map for obtaining `cam` from `heatmap`
        "base_model": str, default "",
            name of the base model of `input_model`,
        "d_class_index_to_name": dict, default {},
            the dict mapping the predicted class index of `img` to class name
        "correct_prediction": int, default None,
            the user-input 'correct' prediction
        "fast": bool, default False,
            if is True, and "correct_prediction" is given,
            then `input_model.predict` will not be called on the processed image
        "verbose": int, or list of str,
            default 0
        "kw_plot": dict, default {},
            key word arguments for plot,
            used only when "verbose" >= 3 or "plot" in "verbose"

    Returns:
    --------
    cam, img_with_cam, heatmap, pred_class_index, pred_class_name: tuple,
        of type ndarray, ndarray, ndarray, int, str resp.,
        as the names imply

    TODO:
    1. use `base_model` and `_GRAD_CAM_RECOMMENDATIONS`

    1 deprecated, will be used in tcm.services.application.sseg

    References:
    -----------
    [1] Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization, https://arxiv.org/abs/1610.02391
    [2] https://github.com/jacobgil/keras-grad-cam
    [3] https://github.com/adityac94/Grad_CAM_plus_plus

    Remarks:
    --------
    Grad-CAM is mathematically essentially Taylor series in several variables,
    using only the linear part as approximation:
    .. math:: (to write)
    """
    img_preprocess_func = kwargs.get("preprocess_func", preprocess_img)
    img_preprocess_func_kw = kwargs.get("preprocess_func_kw", {})
    cam_color_map = kwargs.get("color_map", cv2.COLORMAP_JET)
    base_model = kwargs.get("base_model", "")
    base_model = (''.join([c for c in list(base_model) if (c.isdigit() or c.isalpha())])).lower()
    d_class_index_to_name = kwargs.get("d_class_index_to_name", {})
    correct_prediction = kwargs.get("correct_prediction", None)
    fast = kwargs.get("fast", False) and (correct_prediction is not None)
    verbose = kwargs.get("verbose", 0)
    if isinstance(verbose, int):
        if verbose == 0:
            _verbose = []
        elif verbose == 1:
            _verbose = ['brief']
        elif verbose == 2:
            _verbose = ['brief', 'detail']
        elif verbose >= 3:
            _verbose = ['brief', 'detail', 'plot']
    else:
        _verbose = verbose
    
    preprocessed_img = img_preprocess_func(img, **img_preprocess_func_kw)

    if 'brief' in _verbose:
        print("preprocessed_img.shape =", preprocessed_img.shape)
        print("preprocessed_img.dtype =", preprocessed_img.dtype)
    
    if not fast:
        predictions = input_model.predict(preprocessed_img)[0]
        pred_class_index = np.argmax(predictions)
    else:
        pred_class_index = correct_prediction

    pred_class_name = d_class_index_to_name.get(pred_class_index, "")
    pred_class_name = pred_class_name or d_class_index_to_name.get(str(pred_class_index), "")

    nb_classes = input_model.output_shape[-1]
    
    if 'brief' in _verbose:
        print("nb_classes =", nb_classes)
        if not fast:
            print("predictions =", predictions)
        else:
            print("prediction skipped")
            print("in the following, pred_* are given by the kwargs `correct_prediction`")
        print("pred_class_index =", pred_class_index)
        print("pred_class_name =", pred_class_name)

    if correct_prediction is not None:
        pred_class_index = correct_prediction
        
    # target_layer = lambda x: _target_category_loss(x, pred_class_index, nb_classes)
    # x = Lambda(target_layer, output_shape = _target_category_loss_output_shape)(input_model.output)

    if prediction_layer is None:
        pred_layer = input_model.layers[-1]
        # pred_output = input_model.output[:, pred_class_index]
        pl_name = pred_layer.name  # for verbose printing
        pl_idx = -1  # for verbose printing
    elif isinstance(prediction_layer, str):
        pred_layer = [l for l in input_model.layers if l.name == prediction_layer][0]
        pl_name = prediction_layer
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    elif isinstance(prediction_layer, int):
        pred_layer = input_model.layers[prediction_layer]
        pl_name = pred_layer.name
        pl_idx = prediction_layer
    elif isinstance(prediction_layer, Layer):
        if prediction_layer not in input_model.layers:
            raise ValueError("`prediction_layer` should be a layer of `input_model`!")
        pred_layer = prediction_layer
        pl_name = pred_layer.name
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)

    pred_output = pred_layer.output[:, pred_class_index]
    
    # loss = K.sum(input_model.output)
    if isinstance(target_layer, str):
        tar_layer = [l for l in input_model.layers if l.name == target_layer][0]
        tl_name = target_layer
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)
    elif isinstance(target_layer, int):
        tar_layer = input_model.layers[target_layer]
        tl_name = tar_layer.name
        tl_idx = target_layer
    elif isinstance(target_layer, Layer):
        if target_layer not in input_model.layers:
            raise ValueError("`target_layer` should be a layer of `input_model`!")
        tar_layer = target_layer
        tl_name = tar_layer.name
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)

    target_output = tar_layer.output

    if 'brief' in _verbose:
        print("prediction_layer is {}, the {}-th layer of `input_model`".format(pl_name, pl_idx))
        print("target_layer is {}, the {}-th layer of `input_model`".format(tl_name, tl_idx))
        layers_tot_num = len(input_model.layers)
        print("distance of the two layers is {}".format(modulo(pl_idx,layers_tot_num) - modulo(tl_idx,layers_tot_num)))

    grads = _normalize(_compute_gradients(pred_output, [target_output])[0])
    # print('grads =', grads)
    gradient_function = K.function([input_model.input], [target_output, grads])

    output, grads_val = gradient_function([preprocessed_img])
    # print('output.shape =', output.shape)
    if 'detail' in _verbose:
        print('-'*100)
        print('output =', output)
        print('grads_val =', grads_val)
        print('-'*100)
    # output, grads_val = output[0, :], grads_val[0, :, :, :]
    output, grads_val = output[0, ...], grads_val[0, ...]  # input is a single image
    # print('output.shape =', output.shape)

    weights = np.mean(grads_val, axis = (0, 1))  # eq (1) in Ref.[1]

    # cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    # for i, w in enumerate(weights):
    #     cam += w * output[:, :, i]  # eq (2) in Ref.[1], before ReLU
    cam = np.dot(output, weights)  # eq (2) in Ref.[1], before ReLU
    
    cam = cv2.resize(cam, img.shape[:2][::-1])
    cam = np.maximum(cam, 0)  # ReLU in eq (2) in Ref.[1]
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cam_color_map)
    cam = cam[...,::-1]  # from BGR to RGB
    # print('cam.shape =', cam.shape)
    img_with_cam = np.float32(cam) + np.float32(img)
    img_with_cam = (255 * img_with_cam / np.max(img_with_cam)).astype(np.uint8)

    if 'plot' in _verbose:
        import matplotlib.pyplot as plt
        kw_plot = kwargs.get("kw_plot", {})
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=kw_plot.get('figsize', (18,5)))
        ax1.imshow(img)
        ax2.imshow(cam)
        ax3.imshow(img_with_cam)
        if 'title' in kw_plot:
            fig.suptitle(kw_plot['title'], fontsize=kw_plot.get('fontsize', 16))
        plt.show()

    return cam, img_with_cam, heatmap, pred_class_index, pred_class_name


_GRAD_CAM_PLUSPLUS_RECOMMENDATIONS = {
    "xception": {
        "target_layer": "conv2d_4",
        "prediction_layer": -3,
    },
    "inceptionv4": {},
    "inceptionresnetv2": {},
}


def grad_cam_plusplus_keras(input_model:Model, img:np.ndarray, target_layer:Union[str, int, Layer], prediction_layer:Optional[Union[str, int, Layer]]=None, prediction_activation:str="linear", **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """ finished, not checked, has error,

    Grad-CAM++

    Parameters:
    -----------
    input_model: keras `Model`,
        the model to compute Grad-CAM
    img: ndarray,
        the image to compute Grad-CAM, in RGB
    target_layer: str, or int, or keras `Layer`,
        to write
    prediction_layer: str, or int, or keras `Layer`, optional,
        to write
    prediction_activation: str, default "linear",
        to write
    kwargs: dict, with allowed items:
        "preprocess_func": callable, default `preprocess_img`,
            function to preprocess `img` to valid format to call `input_model.predict`,
        "preprocess_func_kw": dict, default "{}",
            key word arguments for "preprocess_func"
        "color_map": int, default "cv2.COLORMAP_JET"
            cv2 color map for obtaining `cam` from `heatmap`
        "base_model": str, default "",
            name of the base model of `input_model`,
        "d_class_index_to_name": dict, default {},
            the dict mapping the predicted class index of `img` to class name
        "correct_prediction": int, default None,
            the user-input 'correct' prediction
        "fast": bool, default False,
            if is True, and "correct_prediction" is given,
            then `input_model.predict` will not be called on the processed image
        "verbose": int, or list of str,
            default 0
        "kw_plot": dict, default {},
            key word arguments for plot,
            used only when "verbose" >= 3 or "plot" in "verbose"

    TODO:
    1. use `base_model` and `_GRAD_CAM_PLUSPLUS_RECOMMENDATIONS`,
    2. implement the case where `prediction_layer` has `softmax` activation

    1 deprecated, will be used in tcm.services.application.sseg

    Returns:
    --------
    cam, img_with_cam, heatmap, pred_class_index, pred_class_name: tuple,
        of type ndarray, ndarray, ndarray, int, str resp.,
        as the names imply

    References:
    -----------
    [1] Chattopadhay A, Sarkar A, Howlader P, et al. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks, https://arxiv.org/abs/1710.11063
    [2] https://github.com/adityac94/Grad_CAM_plus_plus
    [3] to add

    Remarks:
    --------
    Grad-CAM++ tried to use something of "weighted" 1st order Taylor expansion,
    in attemp to achieve better approximation
    """
    pa = prediction_activation.lower()
    if pa == "linear":
        return _grad_cam_plusplus_keras_linear(
            input_model=input_model,
            img=img,
            target_layer=target_layer,
            prediction_layer=prediction_layer,
            **kwargs
        )
    elif pa == "softmax":
        return _grad_cam_plusplus_keras_softmax(
            input_model=input_model,
            img=img,
            target_layer=target_layer,
            prediction_layer=prediction_layer,
            **kwargs
        )
    else:
        raise NotImplementedError("the case where the prediction layer has {} activation has not yet been implemented!".format(prediction_activation))


def _grad_cam_plusplus_keras_linear(input_model:Model, img:np.ndarray, target_layer:Union[str, int, Layer], prediction_layer:Optional[Union[str, int, Layer]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """
    Grad-CAM++, when the prediction layer has "linear" activation

    Parameters:
    -----------
    ref. `grad_cam_plusplus_keras`

    TODO:
    ref. `grad_cam_plusplus_keras`

    Returns:
    --------
    cam, img_with_cam, heatmap, pred_class_index, pred_class_name: tuple,
        of type ndarray, ndarray, ndarray, int, str resp.,
        as the names imply

    References:
    -----------
    [1] Chattopadhay A, Sarkar A, Howlader P, et al. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks, https://arxiv.org/abs/1710.11063
    [2] https://github.com/adityac94/Grad_CAM_plus_plus
    """
    img_preprocess_func = kwargs.get("preprocess_func", preprocess_img)
    img_preprocess_func_kw = kwargs.get("preprocess_func_kw", {})
    cam_color_map = kwargs.get("color_map", cv2.COLORMAP_JET)
    base_model = kwargs.get("base_model", "")
    base_model = (''.join([c for c in list(base_model) if (c.isdigit() or c.isalpha())])).lower()
    d_class_index_to_name = kwargs.get("d_class_index_to_name", {})
    correct_prediction = kwargs.get("correct_prediction", None)
    fast = kwargs.get("fast", False) and (correct_prediction is not None)
    verbose = kwargs.get("verbose", 0)
    if isinstance(verbose, int):
        if verbose == 0:
            _verbose = []
        elif verbose == 1:
            _verbose = ['brief']
        elif verbose == 2:
            _verbose = ['brief', 'detail']
        elif verbose >= 3:
            _verbose = ['brief', 'detail', 'plot']
    else:
        _verbose = verbose
    
    preprocessed_img = img_preprocess_func(img, **img_preprocess_func_kw)
    if 'brief' in _verbose:
        print("preprocessed_img.shape =", preprocessed_img.shape)
        print("preprocessed_img.dtype =", preprocessed_img.dtype)
    
    if not fast:
        predictions = input_model.predict(preprocessed_img)[0]
        pred_class_index = np.argmax(predictions)
    else:
        pred_class_index = correct_prediction

    pred_class_name = d_class_index_to_name.get(pred_class_index, "")
    pred_class_name = pred_class_name or d_class_index_to_name.get(str(pred_class_index), "")

    nb_classes = input_model.output_shape[-1]
    
    if 'brief' in _verbose:
        print("nb_classes =", nb_classes)
        if not fast:
            print("predictions =", predictions)
        else:
            print("prediction skipped")
            print("in the following, pred_* are given by the kwargs `correct_prediction`")
        print("pred_class_index =", pred_class_index)
        print("pred_class_name =", pred_class_name)

    if correct_prediction is not None:
        pred_class_index = correct_prediction
    
    if prediction_layer is None:
        pred_layer = input_model.layers[-1]
        # pred_output = input_model.output[:, pred_class_index]
        pl_name = pred_layer.name  # for verbose printing
        pl_idx = -1  # for verbose printing
    elif isinstance(prediction_layer, str):
        pred_layer = [l for l in input_model.layers if l.name == prediction_layer][0]
        pl_name = prediction_layer
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    elif isinstance(prediction_layer, int):
        pred_layer = input_model.layers[prediction_layer]
        pl_name = pred_layer.name
        pl_idx = prediction_layer
    elif isinstance(prediction_layer, Layer):
        if prediction_layer not in input_model.layers:
            raise ValueError("`prediction_layer` should be a layer of `input_model`!")
        pred_layer = prediction_layer
        pl_name = pred_layer.name
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    
    # loss = K.sum(input_model.output)
    if isinstance(target_layer, str):
        tar_layer = [l for l in input_model.layers if l.name == target_layer][0]
        tl_name = target_layer
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)
    elif isinstance(target_layer, int):
        tar_layer = input_model.layers[target_layer]
        tl_name = tar_layer.name
        tl_idx = target_layer
    elif isinstance(target_layer, Layer):
        if target_layer not in input_model.layers:
            raise ValueError("`target_layer` should be a layer of `input_model`!")
        tar_layer = target_layer
        tl_name = tar_layer.name
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)

    if 'brief' in _verbose:
        print("prediction_layer is {}, the {}-th layer of `input_model`".format(pl_name, pl_idx))
        print("target_layer is {}, the {}-th layer of `input_model`".format(tl_name, tl_idx))
        layers_tot_num = len(input_model.layers)
        print("distance of the two layers is {}".format(modulo(pl_idx,layers_tot_num) - modulo(tl_idx,layers_tot_num)))

    pred_output = pred_layer.output  # note the difference between `grad_cam_keras`, check why
    # pred_output = pred_layer.output[:, pred_class_index]
    target_output = tar_layer.output

    # grads = K.gradients(pred_output, target_output)[0]
    grads = _normalize(_compute_gradients(pred_output, [target_output])[0])

    first_derivative = K.exp(pred_output)[0][pred_class_index] * grads
    second_derivative = K.exp(pred_output)[0][pred_class_index] * grads * grads
    third_derivative = K.exp(pred_output)[0][pred_class_index] * grads * grads * grads

    gradient_function = K.function([input_model.input], [target_output, first_derivative, second_derivative, third_derivative])
    output, first_grad, second_grad, third_grad = gradient_function([preprocessed_img])

    if 'detail' in _verbose:
        print('output.shape =', output.shape)
        print('-'*100)
        print('output =', output)
        print('first_grad =', first_grad)
        print('second_grad =', second_grad)
        print('third_grad =', third_grad)
        print('-'*100)
    output, first_grad, second_grad, third_grad = output[0], first_grad[0], second_grad[0], third_grad[0]

    global_sum = np.sum(output.reshape((-1, first_grad.shape[2])), axis=0)
    alpha_num = second_grad
    alpha_denom = second_grad * 2.0 + third_grad * global_sum.reshape((1, 1, first_grad.shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
    alpha_normalization_constant_processed = np.where(
        alpha_normalization_constant != 0.0,
        alpha_normalization_constant,
        np.ones(alpha_normalization_constant.shape)
    )
    alphas /= alpha_normalization_constant_processed.reshape((1, 1, first_grad.shape[2]))

    weights = np.maximum(first_grad, 0.0)
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, first_grad.shape[2])))

    heatmap = np.sum(deep_linearization_weights * output, axis=2)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    # print("heatmap =", heatmap)

    heatmap = cv2.resize(heatmap, img.shape[:2][::-1], cv2.INTER_LINEAR)
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cam_color_map)
    cam = cam[...,::-1]  # BGR to RGB
    img_with_cam = np.float32(cam) + np.float32(img)
    img_with_cam = (255 * img_with_cam / np.max(img_with_cam)).astype(np.uint8)

    if 'plot' in _verbose:
        import matplotlib.pyplot as plt
        kw_plot = kwargs.get("kw_plot", {})
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=kw_plot.get('figsize', (18,5)))
        ax1.imshow(img)
        ax2.imshow(cam)
        ax3.imshow(img_with_cam)
        if 'title' in kw_plot:
            fig.suptitle(kw_plot['title'], fontsize=kw_plot.get('fontsize', 16))
        plt.show()

    return cam, img_with_cam, heatmap, pred_class_index, pred_class_name


def _grad_cam_plusplus_keras_softmax(input_model:Model, img:np.ndarray, target_layer:Union[str, int, Layer], prediction_layer:Optional[Union[str, int, Layer]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """ not finished,

    Grad-CAM++, when the prediction layer has "softmax" activation

    Parameters:
    -----------
    ref. `grad_cam_plusplus_keras`

    TODO:
    ref. `grad_cam_plusplus_keras`

    Returns:
    --------
    cam, img_with_cam, heatmap, pred_class_index, pred_class_name: tuple,
        of type ndarray, ndarray, ndarray, int, str resp.,
        as the names imply

    References:
    -----------
    [1] Chattopadhay A, Sarkar A, Howlader P, et al. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks, https://arxiv.org/abs/1710.11063
    """
    img_preprocess_func = kwargs.get("preprocess_func", preprocess_img)
    img_preprocess_func_kw = kwargs.get("preprocess_func_kw", {})
    cam_color_map = kwargs.get("color_map", cv2.COLORMAP_JET)
    base_model = kwargs.get("base_model", "")
    base_model = (''.join([c for c in list(base_model) if (c.isdigit() or c.isalpha())])).lower()
    d_class_index_to_name = kwargs.get("d_class_index_to_name", {})
    correct_prediction = kwargs.get("correct_prediction", None)
    fast = kwargs.get("fast", False) and (correct_prediction is not None)
    verbose = kwargs.get("verbose", 0)
    if isinstance(verbose, int):
        if verbose == 0:
            _verbose = []
        elif verbose == 1:
            _verbose = ['brief']
        elif verbose == 2:
            _verbose = ['brief', 'detail']
        elif verbose >= 3:
            _verbose = ['brief', 'detail', 'plot']
    else:
        _verbose = verbose
    
    preprocessed_img = img_preprocess_func(img, **img_preprocess_func_kw)
    if 'brief' in _verbose:
        print("preprocessed_img.shape =", preprocessed_img.shape)
        print("preprocessed_img.dtype =", preprocessed_img.dtype)
    
    if not fast:
        predictions = input_model.predict(preprocessed_img)[0]
        pred_class_index = np.argmax(predictions)
    else:
        pred_class_index = correct_prediction

    pred_class_name = d_class_index_to_name.get(pred_class_index, "")
    pred_class_name = pred_class_name or d_class_index_to_name.get(str(pred_class_index), "")

    nb_classes = input_model.output_shape[-1]
    
    if 'brief' in _verbose:
        print("nb_classes =", nb_classes)
        if not fast:
            print("predictions =", predictions)
        else:
            print("prediction skipped")
            print("in the following, pred_* are given by the kwargs `correct_prediction`")
        print("pred_class_index =", pred_class_index)
        print("pred_class_name =", pred_class_name)

    if correct_prediction is not None:
        pred_class_index = correct_prediction
    
    if prediction_layer is None:
        pred_layer = input_model.layers[-1]
        # pred_output = input_model.output[:, pred_class_index]
        pl_name = pred_layer.name  # for verbose printing
        pl_idx = -1  # for verbose printing
    elif isinstance(prediction_layer, str):
        pred_layer = [l for l in input_model.layers if l.name == prediction_layer][0]
        pl_name = prediction_layer
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    elif isinstance(prediction_layer, int):
        pred_layer = input_model.layers[prediction_layer]
        pl_name = pred_layer.name
        pl_idx = prediction_layer
    elif isinstance(prediction_layer, Layer):
        if prediction_layer not in input_model.layers:
            raise ValueError("`prediction_layer` should be a layer of `input_model`!")
        pred_layer = prediction_layer
        pl_name = pred_layer.name
        pl_idx = input_model.layers.index(pred_layer) - len(input_model.layers)
    
    # loss = K.sum(input_model.output)
    if isinstance(target_layer, str):
        tar_layer = [l for l in input_model.layers if l.name == target_layer][0]
        tl_name = target_layer
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)
    elif isinstance(target_layer, int):
        tar_layer = input_model.layers[target_layer]
        tl_name = tar_layer.name
        tl_idx = target_layer
    elif isinstance(target_layer, Layer):
        if target_layer not in input_model.layers:
            raise ValueError("`target_layer` should be a layer of `input_model`!")
        tar_layer = target_layer
        tl_name = tar_layer.name
        tl_idx = input_model.layers.index(tar_layer) - len(input_model.layers)

    if 'brief' in _verbose:
        print("prediction_layer is {}, the {}-th layer of `input_model`".format(pl_name, pl_idx))
        print("target_layer is {}, the {}-th layer of `input_model`".format(tl_name, tl_idx))
        layers_tot_num = len(input_model.layers)
        print("distance of the two layers is {}".format(modulo(pl_idx,layers_tot_num) - modulo(tl_idx,layers_tot_num)))

    pred_output = pred_layer.output  # note the difference between `grad_cam_keras`, check why
    target_output = tar_layer.output

    raise NotImplementedError


def get_mask_from_cam(cam:np.ndarray, threshold:Real, **kwargs) -> np.ndarray:
    """
    to implement
    """
    raise NotImplementedError


def get_contour_from_cam(cam:np.ndarray, threshold:Real, **kwargs) -> np.ndarray:
    """
    to implement
    """
    raise NotImplementedError


#--------------------------------------------------
# PyTorch version Grad-CAM, Grad-CAM++

import torch.nn as nn

def grad_cam_torch(input_model:nn.Module, img:np.ndarray, target_layer:Union[str, int, nn.Module], prediction_layer:Optional[Union[str, int, nn.Module]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    to implement
    """
    raise NotImplementedError


def grad_cam_plusplus_torch(input_model:nn.Module, img:np.ndarray, target_layer:Union[str, int, nn.Module], prediction_layer:Optional[Union[str, int, nn.Module]]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    to implement
    """
    raise NotImplementedError

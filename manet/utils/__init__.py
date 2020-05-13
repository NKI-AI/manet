# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import re
import sys
import importlib


def ensure_list(data):
    if not isinstance(data, (list, tuple)):
        data = [data]
    return data


def to_snake_case(camel_input):
    """
    Converts camelcase to snake case.

    From: https://stackoverflow.com/a/46493824/576363

    Parameters
    ----------
    camel_input : str

    Returns
    -------
    string
    """

    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))


def str_to_class(module_name, class_name):
    """
    Convert a string to a class

    From: https://stackoverflow.com/a/1176180/576363

    Parameters
    ----------
    module_name : str
        e.g. manet.data.transforms
    class_name : str
        e.g. Identity

    Returns
    -------
    object

    """

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    the_class = getattr(module, class_name)

    return the_class

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:36:55 2022

@author: mikemartinez
"""

import numpy as np
import torch
import time
import os

from sklearn import (linear_model, model_selection, preprocessing,
                     pipeline)
from scipy.spatial.distance import pdist

from kymatio.torch import HarmonicScattering3D

from kymatio.scattering3d.backend.torch_backend \
    import TorchBackend3D

from kymatio.scattering3d.utils \
    import generate_weighted_sum_of_gaussians

from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
from models import SharedState
import shutil
import runpy
import base64

logger = logging.getLogger(__name__)

def run_fallback_visualization(metrics: Dict[str, Any]):
    [... implementation ...]

def get_base64_image(image_path: str) -> str:
    [... implementation ...]

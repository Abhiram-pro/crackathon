"""
YOLOv8-ES: Enhanced YOLOv8 for Road Crack Detection

Paper: "Efficient and accurate road crack detection technology based on YOLOv8-ES"
"""

__version__ = '1.0.0'
__author__ = 'YOLOv8-ES Implementation'

from .model import EDCM, SGAM, WIoUv3Loss

__all__ = ['EDCM', 'SGAM', 'WIoUv3Loss']

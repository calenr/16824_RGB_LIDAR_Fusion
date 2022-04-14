"""
This file contains custom data structures that we need
"""
import torch


class Bbox3D:
    def __init__(self, x1):
        self.x1 = x1

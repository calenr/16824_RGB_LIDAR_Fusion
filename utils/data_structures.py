"""
This file contains custom data structures that we need
"""
import torch


class Bbox3D:
    def __init__(self, x1):
        # Use third_party.Objectron.objectron.dataset.box
        self.x1 = x1

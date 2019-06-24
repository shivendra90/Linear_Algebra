#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:42:51 2019

@author: Shiv
"""
# analysis:ignore W293, W391

import numpy as np
from numpy import array
from fractions import Fraction
import math
import copy


class LinearSystem:
    """Linear system class."""

    def __init__(self, unknowns):
        self.unknowns = list(unknowns)
        self.rows = len(self.unknowns)
        self.columns = len(self.unknowns[0])

    def form_matrix(self):
        """Forms m by n matrix."""
        mat = self.unknowns
        return array(mat, dtype='float')

    def gauss_jordan(self, b=array([1, 0, 0])):  # b default to 3 dimensions
        """Returns an identity matrix."""
        a = np.array(self.unknowns, dtype='float')
        b = np.array(b, dtype='float')
        Ab = np.hstack([a, b.reshape(-1, 1)])

        m = len(Ab)
        n = len(Ab[0])

        Ab = list(Ab)

        for row, col in enumerate(Ab):
            if col[row] != 0:
                divisor = col[row]
            else:
                raise ValueError("Diagonal elements cannot be zero.")

            for ind, term in enumerate(col):
                col[ind] = term / divisor

            for i in range(m):
                if i != row:
                    inv = -1*Ab[i][row]
                    for j in range(n):
                        Ab[i][j] += inv*Ab[row][j]

        if self.rows < self.columns or self.rows > self.columns:
            print("System has one or more free variables.")
        else:
            print("System has a unique soltuion.")

        return np.array(Ab)

    def __str__(self):
        """Returns array form."""
        return self.unknowns

    def __len__(self):
        """Returns dimensions of the matrix."""
        rows, columns = self.rows, self.columns
        return f"{rows} by {columns} array."


def set_module(module):
    """Custom decorator for modules."""
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return module


@set_module('LinearSystem')
class LinearError(Exception):

    def _raise_linerror_singular(err, indicator):
        raise LinearError("Singular Matrix.")

    def _raise_linerror_incorr_dims(err, indicator):
        raise LinearError("Incorrect number of dimensions.")

    def _raise_linerror_incorr_inp(err, indicator):
        raise LinearError("One or more incorrect inputs.")

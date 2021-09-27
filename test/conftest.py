#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy.random
from pyadjoint import set_working_tape, Tape


def pytest_runtest_setup(item):
    """ Hook function which is called before every test """
    set_working_tape(Tape())

    # Fix the seed to avoid random test failures due to slight tolerance variations
    numpy.random.seed(21)



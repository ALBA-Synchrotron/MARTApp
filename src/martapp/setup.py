#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

import versioneer

version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()


if __name__ == "__main__":
    setup(cmdclass=cmdclass, version=version)

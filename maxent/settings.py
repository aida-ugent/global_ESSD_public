# -*- coding: utf-8 -*-
# @Author: Paolo Simeone
# @Date:   2018-06-28 12:18:08
# @Last Modified by:   UGENT\psimeone
# @Last Modified time: 2018-09-18 11:05:52
import os
cwd = os.getcwd()
# if os.name == 'nt':
test_path = os.path.join(cwd, *['tests'])
lib_path = os.path.join(cwd, *['baseclass'])
examples_path = os.path.join(cwd, *['examples'])

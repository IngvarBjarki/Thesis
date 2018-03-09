# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:53:39 2018

@author: helga
"""

import tensorflow as tf

hello = tf.constant('Hello, tensorflow')
sess = tf.Session()
print(sess.run(hello))
import os
import sys
import time
import json

"""Print Format"""
def print_(text=''):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), text)
    
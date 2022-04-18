# https://youtu.be/UPFbkMWEbMU
"""
!pip install pyautogui
"""

import pyautogui
print(pyautogui.size())

pyautogui.moveTo(100, 100, duration = 1)

import numpy as np
w = np.random.randint(0, 1919)
h = np.random.randint(0, 1079)

pyautogui.moveTo(w, h, duration = 1)

import time

while True:
    w = np.random.randint(0, 1919)
    h = np.random.randint(0, 1079)
    pyautogui.moveTo(w, h, duration = 1)
    time.sleep(5)  #time in seconds
    

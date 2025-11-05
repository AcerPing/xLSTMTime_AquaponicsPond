# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:45:01 2025

@author: HoChePing
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 標籤與數據
ponds = ['IoTpond 2', 'IoTpond 3', 'IoTpond 4', 'IoTpond 1']
categories = ['處理後', 'Train', 'Test']
values = [
    [82099, 65679, 16420],  # IoTpond 2
    [63170, 50536, 12634],  # IoTpond 3
    [45784, 36627, 9157],   # IoTpond 4
    [41726, 8345, 33381],   # IoTpond 1
]
values = np.array(values).T  # 轉置以便依類別分組

# 畫圖
x = np.arange(len(ponds))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 6))

for i, (cat, val) in enumerate(zip(categories, values)):
    ax.bar(x + i * width, val, width, label=cat)

# 標籤與格式
ax.set_xlabel('池塘名稱')
ax.set_ylabel('資料筆數')
ax.set_title('各池塘資料分布（處理後/訓練/測試）')
ax.set_xticks(x + width)
ax.set_xticklabels(ponds)
ax.legend(title='資料類型')
plt.tight_layout()
plt.show()
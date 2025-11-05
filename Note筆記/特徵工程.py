# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:24:54 2025

@author: HoChePing
"""

import matplotlib.pyplot as plt
import pandas as pd

# 根據圖中數據建立 DataFrame
data = {
    'Pond1': [0.002051, 0.001076, 0.993699, 0.002553, 0.000301, 0.000321],
    'Pond2': [0.016854, 0.004858, 0.968696, 0.005342, 0.000372, 0.003878],
    'Pond3': [0.043248, 0.028165, 0.017039, 0.792273, 0.002193, 0.117081],
    'Pond4': [0.056267, 0.00785, 0.045221, 0.101674, 0.001452, 0.787535]
}
features = ['turbidity', 'temperature', 'ph', 'nitrate', 'disolved_oxg', 'ammonia']
df = pd.DataFrame(data, index=features)

# 繪製橫條圖
ax = df.plot(kind='barh', figsize=(10, 6))
ax.invert_yaxis()  # 重點：反轉 y 軸順序
plt.xlabel('Correlation Value')
plt.ylabel('Feature')
plt.title('Correlation Between Water Quality Features (per Pond)')
plt.legend(title='Pond')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


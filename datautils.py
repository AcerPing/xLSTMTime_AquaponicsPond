import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'aquaponics'] # 替換不同資料集。

# 1. ettm1 -> ETT 系列資料（電力需求、負載）
# 2. aquaponics -> 〔養殖〕魚菜共生數據及

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}" # 檢查設定的 params.dset 是否在允許的資料集列表 DSETS 中。
    if not hasattr(params,'use_time_features'): params.use_time_features = True

    if params.dset == 'ettm1': #  判斷目前指定的資料集是否為 'ettm1'
        root_path = 'datasets/ETT-small/' # 資料的資料夾路徑，表示原始的 ETTm1.csv 放在 datasets/ETT-small/ 裡。
        size = [params.context_points, 0, params.target_points] # size 定義輸入輸出長度
                                                                # context_points：輸入的歷史步數，例如過去 336 分鐘。
                                                                # 0：預留（目前沒使用，通常是預測前的空窗）預設不使用，即模型直接根據過去的資料預測未來資料。
                                                                # target_points：模型要預測未來幾點，例如未來 96 點。
                                                                #  xLSTM 這種結構可以直接輸入 context → 預測 target，因此中間 label 可省略。
        dls = DataLoaders( # 建立資料加載器
                datasetCls=Dataset_ETT_minute, # ETT 分鐘級資料專用
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features, # 'M' 表示多變量（multivariate）
                'scale': True, # 是否標準化
                'size': size, # 對應的輸入/輸出長度
                'use_time_features': params.use_time_features # 是否加上時間欄位（例如週期性特徵）
                },
                batch_size=params.batch_size, # 批次大小
                workers=params.num_workers, # 執行緒數
                ) # 給定 Dataset 所需的參數，包括資料檔名、標準化、是否加入時間特徵（如小時、週期）、資料切分長度（size），最後交由 DataLoaders() 包裝成 PyTorch 用的訓練與測試資料迭代器。


    elif params.dset == 'aquaponics':
        root_path = 'datasets/aquaponics/'
        size = [params.context_points, 0, params.target_points] # 參考過去 1440 筆數據來預測下一筆資料。 # * context_points=1440, target_points=1
        dls = DataLoaders(
                datasetCls=Dataset_Aquaponics, 
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'cleaned_IoTPond2.csv',
                'features': params.features, # * MS
                'scale': True,
                'size': size, # * [1440, 0, 1]
                'use_time_features': params.use_time_features # * flase
                },
                batch_size=params.batch_size, # * 656
                workers=params.num_workers,
                )
 
 
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points # dls.vars → 特徵數量（features）
                                                                                # dls.len → 輸入長度（context）
    dls.c = dls.train.dataset[0][1].shape[0] # 預測值的特徵數（通常與 vars 相同）
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    #for i, batch in enumerate(dls.valid):
    #    print(i, len(batch), batch[0].shape, batch[1].shape)
    #breakpoint()


"""
① 判斷資料集類型，讀取資料集。
② 切分資料
③ 做前處理
④ 建立 DataLoader
"""
# TODO: 在 初始化 scaler 部分改成以下這段

# 只對訓練集 fit scaler
train_data_x = df_data_x[:num_train]
train_data_y = df_data_y[:num_train]

self.feature_scaler = MinMaxScaler().fit(train_data_x.values)
self.target_scaler = MinMaxScaler().fit(train_data_y.values)

data_x = self.feature_scaler.transform(df_data_x.values)
data_y = self.target_scaler.transform(df_data_y.values)

self.data_x = data_x[border1:border2]
self.data_y = data_y[border1:border2]
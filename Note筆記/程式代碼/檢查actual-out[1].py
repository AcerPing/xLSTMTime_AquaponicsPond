# TODO: 在main-test測試流程檢查核對out[1]是否輸出正確

# 顯示部分內容（前幾筆）
# print('=== Predicted Values ===')
# print(out[0])  # 預測值前10筆
# print('=== Actual Values (前10筆) ===')
# print(out[1])  # 實際值前10筆
df = pd.DataFrame({
    'Actual': out[1].flatten(),
    # 'Predicted': pred_values.flatten()
})
df.to_csv('actual_vs_predicted.csv', index=False)

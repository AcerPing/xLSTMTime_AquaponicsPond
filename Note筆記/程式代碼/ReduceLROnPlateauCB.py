
class ReduceLROnPlateauCB(TrackerCB):
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0,
                 factor=0.1, patience=5, min_lr=1e-7, verbose=True):
        """
        Args:
            -- monitor (str): 要監控的指標名稱（如 'valid_loss')。
            -- comp (func): 比較函數（例如 np.less), 如果 None, 會自動從指標名稱推斷。
            -- min_delta (float): 新最佳結果的最小變化幅度（小於這個值會被忽略）
            -- factor (float): 每次降低學習率的倍率（例如 0.1 → 降低90%)
            -- patience (int): 容忍多少個 epoch 沒有進步才調整 lr
            -- min_lr (float): 學習率的下限，不會繼續降低到這之下
            -- verbose (bool): 是否印出 log
        """
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

    def before_fit(self):
        super().before_fit()  # 仍需要呼叫父類 TrackerCB 的初始化
        self.num_bad_epochs = 0

    def after_epoch(self):
        super().after_epoch()
        if self.new_best:
            self.num_bad_epochs = 0  # reset if improved
        else:
            # 如果驗證集的監控指標（如 valid_loss）連續超過 patience 個 epoch 沒有改善 → 降低學習率（lr）。
            # 連續 N 個 epoch 沒進步 → 減少學習率 → 重新觀察模型是否能更精細地收斂。
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience: # 判斷是否超過容忍次數（patience）。
                # old_lr = self.learner.opt.hypers[-1]['lr'] # 取得當前 optimizer 使用的學習率。
                # new_lr = max(old_lr * self.factor, self.min_lr) # 計算新的學習率，但不會低於 min_lr。
                # if old_lr > new_lr + 1e-8:  # only update if really lower （確保新 lr 實際上比舊的低（用 +1e-8 避免浮點數誤差））。
                #     self.learner.opt.hypers[-1]['lr'] = new_lr #  將新的學習率寫回 optimizer。
                #     if self.verbose:
                #         print(f'ReduceLROnPlateauCB: reducing lr from {old_lr:.6f} to {new_lr:.6f}')
                # self.num_bad_epochs = 0  # reset after reducing
                for i, param_group in enumerate(self.learner.opt.param_groups):
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if old_lr > new_lr + 1e-8: # only update if really lower
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'ReduceLROnPlateauCB (group {i}): reducing lr from {old_lr:.6f} to {new_lr:.6f}')
                    self.num_bad_epochs = 0  # reset after reducing


# TODO: ReduceLROnPlateauCB(monitor='valid_loss', factor=0.5, patience=5, min_lr=1e-7),
# TODO: learn.fit(n_epochs=args.n_epochs, lr=lr)

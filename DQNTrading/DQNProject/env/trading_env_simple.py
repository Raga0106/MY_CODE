import pandas as pd
import numpy as np

class TradingEnv:
    def __init__(self, data, asset_name="BTC", state_window_size=0):
        # === 資料設定 ===
        self.data = data.copy() # 使用副本避免修改原始資料
        self.asset_name = asset_name
        self.state_window_size = state_window_size # 用於計算 state 的回看窗口 (改為 20)

        # === 環境參數設定 ===
        self.initial_cash = 1000
        self.cash = self.initial_cash
        self.holding = 0 # 持有數量
        self.position = 0  # 1=多單, -1=空單, 0=無持倉
        self.entry_price = None
        self.stop_loss_price = None

        # === 手續費與滑點 (保持移除狀態) ===
        self.commission_rate = 0.0
        self.slippage_rate = 0.0

        # === 風險控管參數 (保持放寬狀態) ===
        self.single_trade_risk_pct = 0.10
        self.total_risk_pct = 0.05          # 總風險暴露（例如：所有持倉的潛在虧損總和不超過總資產的 5%）

        # === 持有時間處罰設定 (重新引入空倉懲罰) ===
        self.holding_penalty_rate = 0.0 # 虧損持倉懲罰不再需要，由 unrealized_reward 處理
        self.profitable_holding_penalty_rate = 0.0 # 盈利持倉懲罰保持移除
        self.idle_penalty_rate = 0.0005 # (修改) 重新引入空倉懲罰

        # === (修改) 獎勵函數參數 ===
        self.realized_reward_multiplier = 5 # (修改) 已實現 PnL 的獎勵放大倍數 (原 30 -> 5)
        self.open_reward = 0.5                # (修改) 降低開倉獎勵

        # === (修改) 績效調整設定 (放寬) ===
        self.win_rate_window = 10       # 計算勝率的回看交易次數
        self.min_win_rate = 0.0         # (修改) 低於此勝率則減半倉位 (原 0.2 -> 0.0, 實際禁用)

        # === 每日虧損控制 (VaR-like) ===
        self.var_control_enabled = True
        self.max_daily_loss_pct = 0.03  # 單日最大虧損比例

        # === 冷卻期設定 (禁用) ===
        self.cooldown_enabled = False # (修改) 交易後的冷卻 K 棒數 (原 True -> False)
        self.cooldown_steps_required = 2 # (保持, 但因 enabled=False 而無效)
        self.cooldown_steps_left = 0

        # === 內部追蹤變數 ===
        self.current_step = 0
        self.total_asset = self.initial_cash
        self.holding_steps = 0 # 持倉 K 棒數
        self.trade_log = [] # 新格式: [(step, type, price, amount, pnl_or_cash_after_open)]

        # === 績效追蹤 ===
        self.cumulative_loss_today = 0.0 # 當日累計虧損金額
        self.current_day = None # 當前日期，用於判斷是否換日
        self.win_flags = [] # 最近 N 筆交易的勝負紀錄 (1=win, 0=loss)
        self.position_size_multiplier = 1.0 # 倉位大小乘數 (1.0 或 0.5)

        # === State 維度 ===
        # 計算一次 state 來確定維度
        # 注意：這裡假設第一次 get_state 能拿到有效值，如果 window_size 很大可能需要調整
        # 或者直接手動指定維度
        # temp_state = self._get_state()
        # self.state_dim = temp_state.shape[0] if temp_state is not None else 12 # 手動指定為 12
        self.state_dim = 12 # 手動指定: close, ema7, ema20, ema60, ema200, rsi, macd, macd_signal, obv, volume_ma7, position, cash_ratio

        # === 預計算指標 (可選，加速) ===
        # self._precompute_indicators()

        self.just_opened = False              # 新增：當前步驟是否剛開倉

        # === 指標需求視窗設定 ===
        self.min_indicator_window = max(200, 14+1, 26, 7, 60)  # 最大所需歷史期數
        self.max_indicator_window = 250                         # 切片最大長度


    # def _precompute_indicators(self):
    #     """預先計算所有時間點的技術指標，避免在 step 中重複計算"""
    #     close = self.data['close'].values
    #     volume = self.data['volume'].values
    #     self.data['ema7'] = self._ema(close, 7)
    #     self.data['ema20'] = self._ema(close, 20)
    #     self.data['ema60'] = self._ema(close, 60)
    #     self.data['ema200'] = self._ema(close, 200)
    #     self.data['rsi'] = self._rsi(close, 14)
    #     macd_line, macd_signal = self._macd(close)
    #     self.data['macd'] = macd_line
    #     self.data['macd_signal'] = macd_signal
    #     self.data['obv'] = self._obv(close, volume)
    #     self.data['volume_ma7'] = self._sma(volume, 7)


    def reset(self):
        """重置環境，開始新的一輪交易"""
        self.cash = self.initial_cash
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None

        self.current_step = 0 # 從頭開始
        self.total_asset = self.initial_cash
        self.holding_steps = 0
        self.trade_log = []

        self.cumulative_loss_today = 0.0
        # 初始化 current_day 為第一筆資料的日期 (假設 timestamp 欄位存在且為可轉換格式)
        if 'timestamp' in self.data.columns and len(self.data) > 0:
             try:
                 self.current_day = pd.to_datetime(self.data.iloc[0]['timestamp']).date()
             except Exception:
                 self.current_day = 0 # 如果轉換失敗，給個預設值
        else:
            self.current_day = 0 # 沒有 timestamp 或 data 為空

        self.win_flags = []
        self.position_size_multiplier = 1.0 # 重置倉位乘數
        self.cooldown_steps_left = 0

        # 返回初始狀態 (需要至少 state_window_size 的數據)
        return self._get_state()

    def step(self, action):
        """
        執行一步動作
        action: 0=持有/不做, 1=買入/做多, 2=賣出/做空
        """
        done = False
        info = {'trade_executed': False, 'reason': ''} # 額外資訊

        current_price = self.data.iloc[self.current_step]['close']
        current_high = self.data.iloc[self.current_step]['high']
        current_low = self.data.iloc[self.current_step]['low']

        self.just_opened = False  # 重置開倉標記

        # === 更新冷卻期 ===
        if self.cooldown_steps_left > 0:
            self.cooldown_steps_left -= 1

        # === 檢查每日是否切換，重設 cumulative_loss_today ===
        if 'timestamp' in self.data.columns:
            try:
                today = pd.to_datetime(self.data.iloc[self.current_step]['timestamp']).date()
                if today != self.current_day:
                    # print(f"Day changed from {self.current_day} to {today} at step {self.current_step}. Resetting daily loss.")
                    self.current_day = today
                    self.cumulative_loss_today = 0.0 # 重置當日虧損
            except Exception:
                pass # Timestamp 格式錯誤或其他問題

        # === 檢查停損 ===
        stop_loss_pnl = 0
        realized_pnl_this_step = 0 # 初始化 realized_pnl_this_step
        trade_executed_by_stoploss = False # 新增標記
        if (self.position == 1 and self.stop_loss_price is not None and current_low <= self.stop_loss_price):
            trigger_price = self.stop_loss_price  # <--- 儲存觸發價格
            closed_holding = self.holding # 記錄被平倉的數量
            stop_loss_pnl = self._close_position(trigger_price) # 使用儲存的價格平倉
            info['trade_executed'] = True
            info['reason'] = f'Long Stop Loss triggered at {trigger_price:.4f}' # <--- 使用儲存的價格格式化
            self._update_performance_tracker(stop_loss_pnl) # <--- 更新績效
            realized_pnl_this_step = stop_loss_pnl
            trade_executed_by_stoploss = True
            # --- 補上停損的交易日誌 ---
            self.trade_log.append((self.current_step, 'Stop Loss (Long)', trigger_price, closed_holding, stop_loss_pnl))
            # --------------------------
        elif (self.position == -1 and self.stop_loss_price is not None and current_high >= self.stop_loss_price):
            trigger_price = self.stop_loss_price  # <--- 儲存觸發價格
            closed_holding = self.holding # 記錄被平倉的數量
            stop_loss_pnl = self._close_position(trigger_price) # 使用儲存的價格平倉
            info['trade_executed'] = True
            info['reason'] = f'Short Stop Loss triggered at {trigger_price:.4f}' # <--- 使用儲存的價格格式化
            self._update_performance_tracker(stop_loss_pnl) # <--- 更新績效
            realized_pnl_this_step = stop_loss_pnl
            trade_executed_by_stoploss = True
             # --- 補上停損的交易日誌 ---
            self.trade_log.append((self.current_step, 'Stop Loss (Short)', trigger_price, closed_holding, stop_loss_pnl))
            # --------------------------

        # === 根據 action 執行交易 (如果未被停損觸發) ===
        action_pnl = None # 初始化 action 產生的 PnL
        if not trade_executed_by_stoploss: # 新邏輯
            action_pnl = self._execute_action(action, current_price)
            if action_pnl is not None: # 如果有執行交易 (平倉)
                info['trade_executed'] = True
                self._update_performance_tracker(action_pnl) # <--- 更新績效
                realized_pnl_this_step = action_pnl # 記錄動作產生的 PnL
            elif self.position != 0 and action == 0: # 如果是持有不動
                 info['reason'] = 'Holding position.'
            elif self.position == 0 and action != 0: # 如果是嘗試開倉但失敗 (例如 kill switch)
                 if not self._can_open_new_position():
                     info['reason'] = 'Open condition not met.'
                 else:
                     info['reason'] = 'Open action executed.' # 假設開倉成功
            else: # 其他情況 (例如無倉位時 action=0)
                 info['reason'] = 'No action taken.'

        # === 更新持有狀態和總資產 ===
        self._update_holding_and_asset(current_price)

        # === 計算 reward ===
        # 將這一步驟中所有實現的 PnL (來自停損或 action) 傳遞給 reward 函數
        reward = self._calculate_reward_v2(realized_pnl_this_step)

        # === 判斷是否 done ===
        # 1. 資料走完
        if self.current_step >= len(self.data) - 1:
            done = True
            info['reason'] = 'End of data reached.'
        # 2. 總資產過低 (例如低於初始資金的 50%)
        if self.total_asset < self.initial_cash * 0.5:
            done = True
            info['reason'] = f'Total asset too low ({self.total_asset:.2f}).'
        # 3. Kill Switch 觸發 (如果啟用) - 可以在 _can_open_new_position 裡檢查，或者在這裡強制結束
        # if self.kill_switch_enabled and not self._check_kill_switch_ok():
        #    done = True
        #    info['reason'] = 'Kill switch triggered.'

        # === 前進一步 ===
        self.current_step += 1

        return self._get_state(), reward, done, info

    def _execute_action(self, action, current_price):
        """
        根據 agent 的動作執行交易邏輯
        action: 0=持有/不做, 1=買入/做多, 2=賣出/做空
        返回: 該交易實現的盈虧 (pnl)，如果沒有交易則返回 None
        """
        executed_pnl = None

        # --- 平倉邏輯 ---
        # 做多時收到賣出信號 -> 平多倉
        if self.position == 1 and action == 2:
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Long', current_price, self.holding, executed_pnl))
        # 做空時收到買入信號 -> 平空倉
        elif self.position == -1 and action == 1:
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Short', current_price, self.holding, executed_pnl))

        # --- 開倉邏輯 (僅在當前無持倉時) ---
        elif self.position == 0:
            # 在決定開倉前，先根據勝率更新倉位大小乘數
            self._update_position_size_multiplier()

            if action == 1: # 開多倉
                if self._can_open_new_position():
                    # risk_per_unit = current_price * self.single_trade_risk_pct # 停損計算移到下方
                    self.stop_loss_price = current_price * (1 - self.single_trade_risk_pct) # 簡易停損
                    # 計算基礎倉位大小
                    base_amount = (self.cash * 0.5) / current_price # (修改) 簡易：用 50% 現金買入 (原 0.9)
                    # 應用乘數調整倉位大小
                    amount_to_buy = base_amount * self.position_size_multiplier # <--- 應用乘數
                    cost = amount_to_buy * current_price * (1 + self.commission_rate + self.slippage_rate)
                    if self.cash >= cost and amount_to_buy > 1e-9: # 確保有足夠現金且倉位不為零
                        self.cash -= cost
                        self.holding = amount_to_buy
                        self.position = 1
                        self.entry_price = current_price
                        self.holding_steps = 0 # 重置持倉計數
                        self.trade_log.append((self.current_step, 'Open Long', current_price, amount_to_buy, self.cash))
                        self.just_opened = True  # 標記開倉
            elif action == 2: # 開空倉
                 if self._can_open_new_position():
                    # risk_per_unit = current_price * self.single_trade_risk_pct # 停損計算移到下方
                    self.stop_loss_price = current_price * (1 + self.single_trade_risk_pct) # 簡易停損
                    # 計算基礎倉位大小
                    base_amount = (self.cash * 0.5) / current_price # (修改) 簡易：用 50% 現金開空 (保證金概念) (原 0.9)
                    # 應用乘數調整倉位大小
                    amount_to_sell = base_amount * self.position_size_multiplier # <--- 應用乘數
                    margin_required = amount_to_sell * current_price # 簡化保證金
                    commission_cost = amount_to_sell * current_price * (self.commission_rate + self.slippage_rate)
                    if self.cash >= margin_required + commission_cost and amount_to_sell > 1e-9: # 確保有足夠現金且倉位不為零
                        self.cash -= commission_cost
                        self.holding = amount_to_sell
                        self.position = -1
                        self.entry_price = current_price
                        self.holding_steps = 0 # 重置持倉計數
                        self.trade_log.append((self.current_step, 'Open Short', current_price, amount_to_sell, self.cash))
                        self.just_opened = True  # 標記開倉

        return executed_pnl # 返回這筆交易的盈虧

    def _close_position(self, close_price):
        """
        平掉目前的倉位，計算並返回盈虧
        close_price: 平倉價格
        """
        pnl = 0
        if self.position == 1:
            # 多單平倉：賣出所得 - 初始成本 (已在開倉時扣除)
            revenue = self.holding * close_price * (1 - self.commission_rate - self.slippage_rate)
            initial_cost = self.holding * self.entry_price # 開倉時的名義成本 (用於計算 PnL)
            pnl = revenue - initial_cost
            self.cash += revenue # 現金增加賣出所得
        elif self.position == -1:
            # 空單平倉：(開倉價 - 平倉價) * 數量 - 交易成本
            # 現金變化 = 初始保證金(近似 entry*holding) + PnL
            price_diff = self.entry_price - close_price
            gross_pnl = price_diff * self.holding
            # 假設平倉時也要支付費用
            commission_cost = self.holding * close_price * (self.commission_rate + self.slippage_rate)
            pnl = gross_pnl - commission_cost # 扣除平倉費用
            # --- 修正現金更新邏輯 ---
            self.cash += pnl # <--- 正確邏輯：現金只增減 PnL
            # --------------------------

        # print(f"Step {self.current_step}: Closing {'Long' if self.position==1 else 'Short'} at {close_price:.4f}. Entry: {self.entry_price:.4f}. Amount: {self.holding:.4f}. PnL: {pnl:.4f}. Cash: {self.cash:.2f}")

        # 重置倉位信息
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None
        self.cooldown_steps_left = self.cooldown_steps_required # 進入冷卻期

        return pnl

    def _update_holding_and_asset(self, current_price):
        """更新持倉天數和當前總資產"""
        if self.position != 0:
            self.holding_steps += 1
        else:
            self.holding_steps = 0

        # 計算當前總資產 = 現金 + 持倉市值
        self.total_asset = self.cash
        if self.position == 1:
            self.total_asset += self.holding * current_price
        elif self.position == -1:
            # 空單市值 = 開倉時的名義價值 + 未實現盈虧
            unrealized_pnl = (self.entry_price - current_price) * self.holding
            # position_value = (self.entry_price * self.holding) + unrealized_pnl # 近似於保證金+浮動盈虧 # 舊的不精確計算
            # 修正：總資產 = 現金 + 空頭部位的未實現盈虧
            self.total_asset = self.cash + unrealized_pnl

    def _calculate_reward_v1(self):
        """計算 reward v1：(當前總資產 - 上一步總資產) / 上一步總資產 - 持有懲罰"""
        # 這個版本在剛平倉時可能會有問題，因為資產變化巨大
        reward = 0
        if self.current_step > 0:
            # 需要獲取上一步的總資產
            # 這需要在 step 開始時記錄，或者重新計算
            # 為了簡化，我們可能需要一個 self.last_total_asset 變數
            # reward = (self.total_asset - self.last_total_asset) / self.last_total_asset
            pass # 暫時不實現這個版本

        # === 計算持有時間懲罰 ===
        holding_penalty = 0
        if self.position != 0: # 只要有持倉就懲罰
            holding_penalty = self.holding_penalty_rate # 可以乘以 holding_steps 讓懲罰隨時間增加

        # reward -= holding_penalty
        return reward # 返回計算出的 reward

    def _calculate_reward_v2(self, realized_pnl):
        """
        計算 reward v2：基於已實現盈虧 + 未實現盈虧變化 - 空倉懲罰 + 開倉獎勵
        持有虧損倉位的獎懲由 unrealized_reward 隱含處理
        realized_pnl: 這一步驟中交易實現的盈虧
        """
        # 1. 已實現盈虧 Reward (使用更新後的 multiplier)
        realized_reward = realized_pnl / self.initial_cash * self.realized_reward_multiplier

        # 2. 未實現盈虧變化 Reward (同時計算 current_unrealized_pnl)
        unrealized_reward = 0
        current_unrealized_pnl = 0 # 初始化
        if self.position != 0 and self.current_step > 0 and self.current_step < len(self.data):
            current_price = self.data.iloc[self.current_step]['close']
            last_price = self.data.iloc[self.current_step - 1]['close']
            price_change = current_price - last_price

            if self.position == 1: # 多單
                unrealized_pnl_change = self.holding * price_change
                current_unrealized_pnl = self.holding * (current_price - self.entry_price)
            elif self.position == -1: # 空單
                unrealized_pnl_change = self.holding * (-price_change)
                current_unrealized_pnl = self.holding * (self.entry_price - current_price)

            unrealized_reward = unrealized_pnl_change / self.initial_cash # 標準化

        # 3. 懲罰 (僅空倉時)
        penalty = 0
        if self.position == 0: # 無持倉 (空倉)
            penalty = self.idle_penalty_rate
        # 持倉時的獎懲已包含在 unrealized_reward 中，無需額外 penalty

        # 4. 總 Reward (不含開倉獎勵)
        reward = realized_reward + unrealized_reward - penalty

        # 5. 開倉即時獎勵
        if self.just_opened:
            reward += self.open_reward

        # === 更新每日虧損累計（給 VaR 控制用）===
        # 這個應該在計算完總資產後更新
        if self.current_step > 0:
             # 需要上一步的 total_asset
             # 可以在 step 開始時保存 last_total_asset = self.total_asset
             # daily_pnl = self.total_asset - self.last_total_asset
             # if daily_pnl < 0:
             #    self.cumulative_loss_today += abs(daily_pnl)
             pass # 暫時不實現精確的每日虧損更新

        return reward


    def _update_performance_tracker(self, pnl):
        """當一筆交易完成時，更新績效追蹤指標 (只更新勝率)"""
        if pnl > 0:
            self.win_flags.append(1)
        elif pnl < 0:
            self.win_flags.append(0)
        # else: pnl == 0 不記錄或視為輸

        # 維護勝率計算窗口
        if len(self.win_flags) > self.win_rate_window:
            self.win_flags.pop(0)

    def _update_position_size_multiplier(self):
        """根據最近的勝率更新倉位大小乘數"""
        if len(self.win_flags) < self.win_rate_window:
            # 交易次數不足，使用預設倉位
            self.position_size_multiplier = 1.0
            return

        current_win_rate = sum(self.win_flags) / len(self.win_flags)
        if current_win_rate < self.min_win_rate:
            # 勝率低於門檻，減半倉位
            self.position_size_multiplier = 0.5
            # print(f"Step {self.current_step}: Win rate {current_win_rate:.2f} < {self.min_win_rate}. Reducing position size multiplier to 0.5.")
        else:
            # 勝率達標或回升，恢復正常倉位
            self.position_size_multiplier = 1.0
            # print(f"Step {self.current_step}: Win rate {current_win_rate:.2f} >= {self.min_win_rate}. Setting position size multiplier to 1.0.")

    def _can_open_new_position(self):
        """檢查是否滿足所有開倉條件 (移除 Kill Switch 檢查)"""
        # 1. 檢查冷卻期
        if self.cooldown_enabled and self.cooldown_steps_left > 0:
            return False

        # 2. (移除) 檢查 Kill Switch
        # if not self._check_kill_switch_ok():
        #     return False

        # 3. 檢查每日虧損限制 (VaR) - 保持不變 (如果需要實現)
        # if self.var_control_enabled and self.cumulative_loss_today >= self.initial_cash * self.max_daily_loss_pct:
        #     return False

        # 4. 檢查總風險暴露 - 保持不變 (如果需要實現)
        # if not self._check_total_risk():
        #     return False

        return True

    def _get_state(self):
        idx = self.current_step
        # 若資料不足以計算最基本的指標 (例如需要 2 期計算 diff)，回傳零向量
        # 將門檻稍微放寬，至少要有幾根 K 棒
        min_required_data = 2 # 至少需要 2 筆資料才能計算變化
        if idx < min_required_data:
            return np.zeros(self.state_dim)

        # 擷取最大切片範圍，以涵蓋所有指標最長所需資料
        # 但確保 start_idx 不為負
        start_idx = max(0, idx - self.max_indicator_window)
        # 確保切片至少有 min_required_data 筆，否則 state 維度可能不符或計算出錯
        if idx - start_idx < min_required_data:
             # 如果 idx 靠近開頭，切片可能不足 min_required_data
             # 這種情況下，雖然 idx >= min_required_data，但可用數據仍太少
             # 為了安全起見，也返回零向量
             # 或者，可以嘗試用 idx+1 來切片，確保包含當前步驟數據
             data_slice = self.data.iloc[start_idx : idx + 1]
             if len(data_slice) < min_required_data:
                 return np.zeros(self.state_dim)
        else:
             data_slice = self.data.iloc[start_idx : idx + 1] # 包含當前 step 的數據

        close_prices = data_slice['close'].values
        volumes = data_slice['volume'].values

        # 檢查是否有足夠數據進行基本計算
        if len(close_prices) < min_required_data:
             return np.zeros(self.state_dim)

        # 逐一計算指標，若資料不足則設為 np.nan
        latest_close = close_prices[-1]
        latest_ema7    = self._ema(close_prices, 7)[-1]    # _ema 內部已處理不足情況返回 nan
        latest_ema20   = self._ema(close_prices, 20)[-1]
        latest_ema60   = self._ema(close_prices, 60)[-1]
        latest_ema200  = self._ema(close_prices, 200)[-1]
        latest_rsi     = self._rsi(close_prices, 14)[-1]   # _rsi 內部已處理不足情況返回 nan
        macd_line, macd_signal = self._macd(close_prices) # _macd 內部已處理不足情況返回 nan
        latest_macd         = macd_line[-1] if macd_line is not None and len(macd_line) > 0 else np.nan
        latest_macd_signal  = macd_signal[-1] if macd_signal is not None and len(macd_signal) > 0 else np.nan
        latest_obv      = self._obv(close_prices, volumes)[-1] # _obv 內部已處理不足情況返回 nan
        latest_volume_ma7 = self._sma(volumes, 7)[-1]        # _sma 內部已處理不足情況返回 nan

        # === 特徵集合 ===
        cash_ratio = self.cash / (self.total_asset if self.total_asset != 0 else 1e-8) # 避免除以零
        features = np.array([
            latest_close, latest_ema7, latest_ema20, latest_ema60, latest_ema200,
            latest_rsi, latest_macd, latest_macd_signal, latest_obv, latest_volume_ma7,
            float(self.position), cash_ratio # 確保 position 是 float
        ], dtype=np.float32)

        # === 正規化 (處理 NaN) ===
        # 對 close 價格進行 Z-score 正規化 (忽略 NaN)
        close_mean = np.nanmean(close_prices)
        close_std = np.nanstd(close_prices)
        if close_std > 1e-8: # 避免除以零或極小值
            features[0] = (features[0] - close_mean) / close_std
        else:
            features[0] = 0.0 # 如果標準差為零，則設為 0

        # 對 RSI 正規化 (範圍 [-1, 1])
        # 如果 latest_rsi 是 NaN，這裡會保持 NaN
        features[5] = (features[5] - 50.0) / 50.0

        # 其他指標 (EMA, MACD, OBV, VolMA) 可以考慮是否需要正規化
        # 例如，可以對 EMA 相對於 Close 的比例進行正規化
        # features[1] = (features[1] - features[0]) if not np.isnan(features[1]) and not np.isnan(features[0]) else np.nan # EMA7 vs Close (正規化後)
        # ... 其他 EMA ...
        # 對於 MACD，可以考慮除以 Close 的一個比例或標準差
        # 對於 OBV 和 Volume MA，可以考慮對數轉換或與近期均值比較

        # === 新版 特徵集合：相對/標準化指標 ===
        # price 已用 Z-score 正規化後存於 features[0]
        close_norm = features[0]
        # 相對指標 (EMA, MACD) 相對於最新收盤價
        ema7_rel   = (latest_ema7   - latest_close) / latest_close if not np.isnan(latest_ema7) else 0.0
        ema20_rel  = (latest_ema20  - latest_close) / latest_close if not np.isnan(latest_ema20) else 0.0
        ema60_rel  = (latest_ema60  - latest_close) / latest_close if not np.isnan(latest_ema60) else 0.0
        ema200_rel = (latest_ema200 - latest_close) / latest_close if not np.isnan(latest_ema200) else 0.0
        macd_rel         = latest_macd        / latest_close if not np.isnan(latest_macd) else 0.0
        macd_signal_rel  = latest_macd_signal / latest_close if not np.isnan(latest_macd_signal) else 0.0
        # OBV 與 Volume MA7 按平均交易量標準化
        vol_mean = np.nanmean(volumes) if len(volumes)>0 else 1.0
        obv_norm    = latest_obv      / vol_mean if not np.isnan(latest_obv) else 0.0
        vol_ma7_norm= latest_volume_ma7 / vol_mean if not np.isnan(latest_volume_ma7) else 0.0

        # 組合最終特徵向量
        features = np.array([
            close_norm, ema7_rel, ema20_rel, ema60_rel, ema200_rel,
            features[5],             # RSI (已正規化)
            macd_rel, macd_signal_rel,
            obv_norm, vol_ma7_norm,
            float(self.position), cash_ratio
        ], dtype=np.float32)

        # === 最後一步：將所有 NaN 替換為 0.0 ===
        # 這樣無法計算的指標或正規化中產生的 NaN 都會變成 0
        final_features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 確保返回的維度正確
        if final_features.shape[0] != self.state_dim:
             # print(f"Warning: State dimension mismatch. Expected {self.state_dim}, got {final_features.shape[0]}. Returning zeros.")
             return np.zeros(self.state_dim)

        return final_features

    # --- 技術指標計算函數 ---
    # 確保這些函數在數據不足時返回 np.nan 或包含 np.nan 的陣列

    def _ema(self, data, period):
        """指數移動平均 (EMA)"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float) # 返回 float 類型的 nan
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def _sma(self, data, period):
        """簡單移動平均 (SMA)"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        return pd.Series(data).rolling(window=period).mean().values

    def _rsi(self, data, period=14):
        """相對強弱指數 (RSI)"""
        if len(data) < period + 1:
            return np.full_like(data, np.nan, dtype=float)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # 使用 EMA 計算平均增益和損失
        # 將 com 改為 alpha=1/period 以匹配常見定義
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_full = np.full_like(data, np.nan, dtype=float)
        # diff 會減少一個元素，ewm 可能會產生前導 NaN，確保索引正確
        # 確保 rsi 陣列長度足夠填充
        if len(rsi) >= period:
             rsi_full[period:] = rsi[period-1:] # 從第一個有效值開始填充
        elif len(rsi) > 0:
             # 如果 rsi 長度不足 period 但大於 0，從最後一個可用值填充
             rsi_full[-len(rsi):] = rsi
        return rsi_full

    def _macd(self, data, fast=12, slow=26, signal=9):
        """MACD 指標"""
        if len(data) < slow: # 需要足夠數據計算慢線
             # 返回兩個充滿 NaN 的陣列
             return np.full_like(data, np.nan, dtype=float), np.full_like(data, np.nan, dtype=float)
        ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        # 確保 macd_line 有足夠數據計算 signal line
        if len(macd_line.dropna()) < signal:
             macd_signal_line = pd.Series(np.full_like(data, np.nan, dtype=float), index=macd_line.index)
        else:
             macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.values, macd_signal_line.values # 返回 NumPy 陣列

    def _obv(self, close, volume):
        """能量潮指標 (OBV)"""
        # OBV 從 0 開始累加，即使數據很少也能計算，但意義不大
        # 如果嚴格要求，可以設定一個最小長度，例如 2
        if len(close) < 2:
             # 返回與 close 同形狀的 NaN 陣列
             return np.full_like(close, np.nan, dtype=float)

        # 確保 volume 長度匹配 close
        if len(volume) != len(close):
             # print("Warning: OBV calculation encountered mismatched close/volume lengths.")
             # 嘗試截斷或填充 volume，或者直接返回 NaN
             return np.full_like(close, np.nan, dtype=float)

        obv = np.zeros_like(close, dtype=float) # <--- 修改 np.zeros像 為 np.zeros_like
        # 迭代計算 OBV
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    def render(self, mode='human'):
        """可選：實現一個簡單的渲染方法，打印當前狀態"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Holding: {self.holding:.4f}, Position: {self.position}, Total Asset: {self.total_asset:.2f}, Daily Loss: {self.cumulative_loss_today:.2f}")
        else:
            super(TradingEnv, self).render(mode=mode) # or raise NotImplementedError

    def close(self):
        """可選：清理資源"""
        pass

# === Example Usage (for testing) ===
if __name__ == '__main__':
    # 創建假數據
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.rand(500) * 100 + 1000,
        'high': lambda x: x['open'] + np.random.rand(500) * 10,
        'low': lambda x: x['open'] - np.random.rand(500) * 10,
        'close': lambda x: x['open'] + np.random.randn(500) * 5,
        'volume': np.random.rand(500) * 1000 + 100
    })
    # 確保 high >= open, close and low <= open, close
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)


    env = TradingEnv(data, state_window_size=50) # 測試時用小一點的 window
    state = env.reset()
    print("Initial State Dim:", env.state_dim)
    print("Initial State (first few):", state[:5])
    print("Initial Total Asset:", env.total_asset)

    done = False
    total_reward = 0
    steps = 0
    while not done and steps < len(data) -1 : # 確保不超出索引
        action = np.random.randint(0, 3) # 隨機動作: 0=hold, 1=buy, 2=sell
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if steps % 50 == 0 or info.get('trade_executed'):
             env.render()
             print(f"Action: {action}, Reward: {reward:.4f}, Done: {done}, Info: {info}")
             print("-" * 20)

    print(f"\nFinished after {steps} steps.")
    print(f"Final Total Asset: {env.total_asset:.2f}")
    print(f"Total Reward: {total_reward:.4f}")
    print("Trade Log:", env.trade_log[-5:]) # 顯示最後幾筆交易
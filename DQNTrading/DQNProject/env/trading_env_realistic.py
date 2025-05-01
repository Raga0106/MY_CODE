import pandas as pd
import numpy as np
import warnings # 用於更優雅的警告

class TradingEnv:
    def __init__(self, data, asset_name="BTC", state_window_size=100, initial_cash=10000, # 增加初始資金
                 commission_rate=0.0005, slippage_rate=0.0005, # 調整滑點
                 single_trade_risk_pct=0.02, max_daily_loss_pct=0.03,
                 holding_penalty_rate_profit=0.00005, holding_penalty_rate_loss=0.0002,
                 no_position_penalty_rate=0.0001, cooldown_steps_required=5,
                 kill_switch_enabled=True, win_rate_window=20, min_win_rate=0.3, max_losing_streak=7, # 調整 Kill Switch 參數
                 var_control_enabled=True, state_dim=12): # 允許外部傳入 state_dim
        # === 資料設定 ===
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        self.data = data.copy().reset_index(drop=True) # 確保索引是從 0 開始的連續整數
        self.asset_name = asset_name
        self.state_window_size = state_window_size

        # === 環境參數設定 ===
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # === 風險控管參數 ===
        self.single_trade_risk_pct = single_trade_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct

        # === 持有時間處罰設定 ===
        self.holding_penalty_rate_profit = holding_penalty_rate_profit
        self.holding_penalty_rate_loss = holding_penalty_rate_loss
        self.no_position_penalty_rate = no_position_penalty_rate

        # === 績效停單設定 (Kill Switch) ===
        self.kill_switch_enabled = kill_switch_enabled
        self.win_rate_window = win_rate_window
        self.min_win_rate = min_win_rate
        self.max_losing_streak = max_losing_streak

        # === 每日虧損控制 (VaR-like) ===
        self.var_control_enabled = var_control_enabled

        # === 冷卻期設定 ===
        self.cooldown_enabled = cooldown_steps_required > 0
        self.cooldown_steps_required = cooldown_steps_required

        # === State 維度 ===
        self.state_dim = state_dim # 由外部傳入或保持預設

        # === 內部追蹤變數 (在 reset 中初始化) ===
        self.cash = 0
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None
        self.current_step = 0
        self.total_asset = 0
        self.last_total_asset = 0
        self.holding_steps = 0
        self.trade_log = []
        self.cumulative_loss_today = 0.0
        self.current_day = None
        self.win_flags = []
        self.loss_streak = 0
        self.cooldown_steps_left = 0

        # === 初始化內部狀態 ===
        self.reset()

    def reset(self):
        """重置環境，開始新的一輪交易"""
        self.cash = self.initial_cash
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None

        self.current_step = 0 # 從頭開始
        self.total_asset = self.initial_cash
        self.last_total_asset = self.initial_cash
        self.holding_steps = 0
        self.trade_log = []

        self.cumulative_loss_today = 0.0
        # 初始化 current_day (更健壯)
        try:
            self.current_day = pd.to_datetime(self.data.iloc[0]['timestamp']).date()
        except (IndexError, KeyError, TypeError, ValueError):
            self.current_day = 0 # 如果無法獲取日期，設為 0

        self.win_flags = []
        self.loss_streak = 0
        self.cooldown_steps_left = 0

        # 返回初始狀態，確保有足夠數據
        return self._get_state(self.current_step) # 使用索引 0 獲取初始狀態

    def step(self, action):
        """執行一步動作"""
        done = False
        info = {'trade_executed': False, 'reason': '', 'pnl': 0.0} # 初始化 info
        realized_pnl_this_step = 0.0

        # --- 邊界檢查 ---
        if self.current_step >= len(self.data):
            warnings.warn(f"Step called with invalid current_step {self.current_step}. Returning zero state and done=True.")
            return np.zeros(self.state_dim), 0.0, True, {'reason': 'Invalid step index'}

        # --- 記錄上一步資產 ---
        self.last_total_asset = self.total_asset

        # --- 獲取當前 K 線數據 ---
        current_kline = self.data.iloc[self.current_step]
        current_price = current_kline['close']
        current_high = current_kline['high']
        current_low = current_kline['low']

        # --- 更新冷卻期 ---
        if self.cooldown_steps_left > 0:
            self.cooldown_steps_left -= 1

        # --- 更新日期與每日虧損 ---
        try:
            today = pd.to_datetime(current_kline['timestamp']).date()
            if today != self.current_day:
                self.current_day = today
                self.cumulative_loss_today = 0.0
        except (KeyError, TypeError, ValueError):
            pass # 無法處理日期則忽略

        # --- 處理停損 ---
        stop_loss_triggered = False
        if self.position == 1 and self.stop_loss_price is not None and current_low <= self.stop_loss_price:
            trigger_price = self.stop_loss_price
            realized_pnl_this_step = self._close_position(trigger_price)
            info['trade_executed'] = True
            info['reason'] = f'Long SL @{trigger_price:.4f}'
            info['pnl'] = realized_pnl_this_step
            stop_loss_triggered = True
            self._update_performance_tracker(realized_pnl_this_step)
        elif self.position == -1 and self.stop_loss_price is not None and current_high >= self.stop_loss_price:
            trigger_price = self.stop_loss_price
            realized_pnl_this_step = self._close_position(trigger_price)
            info['trade_executed'] = True
            info['reason'] = f'Short SL @{trigger_price:.4f}'
            info['pnl'] = realized_pnl_this_step
            stop_loss_triggered = True
            self._update_performance_tracker(realized_pnl_this_step)

        # --- 執行智能體動作 (如果未觸發停損) ---
        if not stop_loss_triggered:
            action_pnl = self._execute_action(action, current_price)
            if action_pnl is not None: # 平倉動作被執行
                realized_pnl_this_step = action_pnl
                info['trade_executed'] = True
                # reason 已在 _execute_action 的 trade_log 中記錄，這裡簡化
                info['reason'] = f'Action Close {"Long" if self.position == 0 else "Short"}' # 記錄是哪個動作導致平倉
                info['pnl'] = realized_pnl_this_step
                self._update_performance_tracker(realized_pnl_this_step)
            else: # 未執行平倉 (可能是開倉、持有或無法開倉)
                if self.position != 0 and action == 0:
                    info['reason'] = 'Hold'
                elif self.position == 0 and action != 0:
                    # 檢查是否真的開倉了 (position 狀態改變)
                    # _execute_action 內部會處理開倉邏輯
                    # 我們可以在這裡檢查 self.position 是否改變來判斷
                    # 但 _execute_action 已經記錄了 trade_log，所以這裡可以簡化
                    if not self._can_open_new_position():
                         info['reason'] = 'Cannot Open (Condition)'
                    # else: # 假設 _execute_action 成功開倉 (如果失敗會在內部處理)
                    #     info['reason'] = f'Action Open {"Long" if action == 1 else "Short"}'
                elif self.position == 0 and action == 0:
                    info['reason'] = 'No Position, No Action'

        # --- 更新持有狀態和總資產 ---
        self._update_holding_and_asset(current_price)

        # --- 計算獎勵 ---
        reward = self._calculate_reward_v2(realized_pnl_this_step)

        # --- 判斷是否結束 ---
        # 1. 資料走完
        if self.current_step >= len(self.data) - 1:
            done = True
            info['reason'] += ' EndOfData'
        # 2. 總資產過低
        if self.total_asset < self.initial_cash * 0.2: # 更嚴格的破產線
            done = True
            info['reason'] += ' Bankrupt'
        # 3. Kill Switch 觸發 (可選，讓環境結束)
        # if self.kill_switch_enabled and not self._check_kill_switch_ok():
        #    done = True
        #    info['reason'] += ' KillSwitch'

        # --- 獲取下一步狀態 ---
        next_step_index = self.current_step + 1
        next_state = self._get_state(next_step_index)

        # --- 前進到下一步 ---
        self.current_step += 1

        return next_state, reward, done, info

    def _execute_action(self, action, current_price):
        """執行交易邏輯，返回實現的盈虧 (僅平倉時)"""
        executed_pnl = None
        action_executed = False # 標記是否有動作執行

        # --- 平倉邏輯 ---
        if self.position == 1 and action == 2: # 平多倉
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Long', current_price, self.holding, executed_pnl))
            action_executed = True
        elif self.position == -1 and action == 1: # 平空倉
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Short', current_price, self.holding, executed_pnl))
            action_executed = True

        # --- 開倉邏輯 ---
        elif self.position == 0 and action != 0: # 嘗試開倉
            if self._can_open_new_position():
                # 計算止損價格
                stop_loss_price = 0
                if action == 1: # 多倉
                    stop_loss_price = current_price * (1 - self.single_trade_risk_pct)
                elif action == 2: # 空倉
                    stop_loss_price = current_price * (1 + self.single_trade_risk_pct)

                # 計算基於風險的倉位大小
                risk_per_unit = abs(current_price - stop_loss_price)
                if risk_per_unit > 1e-9: # 避免除以零
                    max_loss_amount = self.total_asset * self.single_trade_risk_pct
                    position_size = max_loss_amount / risk_per_unit

                    # 檢查現金/保證金是否足夠
                    cost_or_margin = 0
                    if action == 1: # 多倉成本
                        cost_or_margin = position_size * current_price * (1 + self.commission_rate + self.slippage_rate)
                    elif action == 2: # 空倉所需現金 (簡化)
                        cost_or_margin = position_size * current_price * (self.commission_rate + self.slippage_rate) # 僅費用
                        # 更真實的檢查應考慮保證金要求

                    if self.cash >= cost_or_margin and position_size > 1e-9: # 確保倉位 > 0
                        # 再次檢查最大可負擔數量
                        max_affordable_size = float('inf')
                        if action == 1:
                             max_affordable_size = self.cash / (current_price * (1 + self.commission_rate + self.slippage_rate))
                        # 空倉的最大可負擔量較難精確，暫不限制

                        final_position_size = min(position_size, max_affordable_size)

                        if final_position_size > 1e-9:
                            # --- 執行開倉 ---
                            self.holding = final_position_size
                            self.entry_price = current_price
                            self.stop_loss_price = stop_loss_price
                            self.holding_steps = 0

                            if action == 1: # 開多
                                self.position = 1
                                final_cost = final_position_size * current_price * (1 + self.commission_rate + self.slippage_rate)
                                self.cash -= final_cost
                                self.trade_log.append((self.current_step, 'Open Long', current_price, final_position_size, 0))
                                action_executed = True
                            elif action == 2: # 開空
                                self.position = -1
                                final_commission_cost = final_position_size * current_price * (self.commission_rate + self.slippage_rate)
                                self.cash -= final_commission_cost # 僅扣除費用 (簡化)
                                self.trade_log.append((self.current_step, 'Open Short', current_price, final_position_size, 0))
                                action_executed = True
                # else: # risk_per_unit 為 0，無法開倉
                #     pass
            # else: # 無法開倉 (條件不滿足)
            #     pass

        # 如果沒有執行任何交易動作 (例如持有或無法開倉)，返回 None
        return executed_pnl

    def _close_position(self, close_price):
        """平倉，計算盈虧，返回 PnL"""
        pnl = 0
        if self.position == 1: # 平多
            revenue = self.holding * close_price * (1 - self.commission_rate - self.slippage_rate)
            # 成本基於入場價，不含手續費，因為手續費已在開倉時扣除
            initial_cost = self.holding * self.entry_price
            pnl = revenue - initial_cost
            self.cash += revenue
        elif self.position == -1: # 平空
            price_diff = self.entry_price - close_price
            gross_pnl = price_diff * self.holding
            # 平倉費用
            commission_cost = self.holding * close_price * (self.commission_rate + self.slippage_rate)
            pnl = gross_pnl - commission_cost
            # 現金變化：拿回開倉時的名義價值 + PnL (開倉費用已扣)
            cash_change = (self.holding * self.entry_price) + pnl
            self.cash += cash_change

        # 重置倉位
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None
        if self.cooldown_enabled:
            self.cooldown_steps_left = self.cooldown_steps_required

        return pnl

    def _update_holding_and_asset(self, current_price):
        """更新持倉步數和總資產"""
        if self.position != 0:
            self.holding_steps += 1
        else:
            self.holding_steps = 0

        self.total_asset = self.cash
        if self.position == 1:
            # 預計平倉價值
            position_value = self.holding * current_price * (1 - self.commission_rate - self.slippage_rate)
            self.total_asset += position_value
        elif self.position == -1:
            # 空單權益 = 開倉名義價值 + 未實現盈虧 - 預計平倉費用
            unrealized_pnl = (self.entry_price - current_price) * self.holding
            estimated_close_cost = self.holding * current_price * (self.commission_rate + self.slippage_rate)
            # 空單權益代表平倉後能拿回的現金增量（相對於開倉時鎖定的名義價值）
            position_equity = (self.holding * self.entry_price) + unrealized_pnl - estimated_close_cost
            # 總資產 = 現金 + 空單權益 (這裡的 cash 已經扣了開倉費用)
            self.total_asset += position_equity # 這裡的計算需要仔細核對，但概念上是這樣

    def _calculate_reward_v2(self, realized_pnl):
        """計算獎勵 v2"""
        # 1. 已實現盈虧 Reward
        realized_reward_factor = 10.0 # 增加已實現獎勵的權重
        realized_reward_component = (realized_pnl / self.initial_cash) * realized_reward_factor

        # 2. 未實現盈虧變化 Reward
        unrealized_reward = 0
        if self.position != 0 and realized_pnl == 0 and self.current_step > 0:
            # 確保 current_step 和 current_step - 1 都在有效範圍內
            if self.current_step < len(self.data):
                 current_price = self.data.iloc[self.current_step]['close']
                 if self.current_step - 1 >= 0:
                     last_price = self.data.iloc[self.current_step - 1]['close']
                     price_change = current_price - last_price
                     unrealized_pnl_change = 0
                     if self.position == 1:
                         unrealized_pnl_change = self.holding * price_change
                     elif self.position == -1:
                         unrealized_pnl_change = self.holding * (-price_change)

                     unrealized_reward_factor = 1.0 # 未實現獎勵權重相對較小
                     unrealized_reward = (unrealized_pnl_change / self.initial_cash) * unrealized_reward_factor

        # 3. 持有懲罰
        holding_penalty = 0
        if self.position != 0:
             if self.current_step < len(self.data) and self.entry_price is not None: # 確保能獲取價格和入場價
                 current_price = self.data.iloc[self.current_step]['close']
                 is_profitable = (self.position == 1 and current_price > self.entry_price) or \
                                 (self.position == -1 and current_price < self.entry_price)
                 penalty_rate = self.holding_penalty_rate_profit if is_profitable else self.holding_penalty_rate_loss
                 holding_penalty = penalty_rate * self.holding_steps
        else:
            holding_penalty = self.no_position_penalty_rate # 無倉位懲罰

        # 4. 總 Reward
        reward = realized_reward_component + unrealized_reward - holding_penalty

        # === 更新每日虧損累計 ===
        asset_change = self.total_asset - self.last_total_asset
        if asset_change < 0:
            self.cumulative_loss_today += abs(asset_change)

        # --- Reward Clipping (可選) ---
        # reward = np.clip(reward, -0.5, 0.5) # 限制獎勵範圍

        return reward

    def _update_performance_tracker(self, pnl):
        """更新績效追蹤指標"""
        if abs(pnl) > 1e-9: # 只有顯著的盈虧才計入
            if pnl > 0:
                self.win_flags.append(1)
                self.loss_streak = 0
            else:
                self.win_flags.append(0)
                self.loss_streak += 1
            # 維護窗口
            if len(self.win_flags) > self.win_rate_window:
                self.win_flags.pop(0)

    def _check_kill_switch_ok(self):
        """檢查 Kill Switch 條件"""
        if not self.kill_switch_enabled: return True
        # 檢查勝率
        if len(self.win_flags) >= self.win_rate_window:
            current_win_rate = sum(self.win_flags) / len(self.win_flags)
            if current_win_rate < self.min_win_rate: return False
        # 檢查連敗
        if self.loss_streak >= self.max_losing_streak: return False
        return True

    def _can_open_new_position(self):
        """檢查是否滿足所有開倉條件"""
        if self.cooldown_enabled and self.cooldown_steps_left > 0: return False
        if not self._check_kill_switch_ok(): return False
        if self.var_control_enabled and self.cumulative_loss_today >= self.initial_cash * self.max_daily_loss_pct: return False
        # TODO: 實現總風險暴露檢查 (_check_total_risk)
        return True

    def _get_state(self, step_index): # 接受 step_index
        """整理 observation"""
        idx = step_index

        # === 邊界與數據充足性檢查 ===
        if idx >= len(self.data):
            # 這是預期的情況，當 step 請求下一步狀態但已到結尾
            # print(f"Debug: Requested state index {idx} is out of bounds. Returning zero state.") # Debug 用
            return np.zeros(self.state_dim)
        if idx < self.state_window_size - 1:
            # print(f"Debug: Not enough data for state window at index {idx}. Returning zero state.") # Debug 用
            return np.zeros(self.state_dim)

        # === 抓取數據窗口 ===
        start_idx = idx - self.state_window_size + 1
        indicator_start_idx = max(0, idx - 250) # 指標計算需要更長數據
        data_slice = self.data.iloc[indicator_start_idx : idx + 1]

        if data_slice.empty or len(data_slice) < 2: # 需要至少2點計算指標
            # print(f"Debug: Insufficient data in slice at index {idx}. Returning zero state.") # Debug 用
            return np.zeros(self.state_dim)

        close_prices = data_slice['close'].values
        volumes = data_slice['volume'].values

        # === 計算指標 (增加錯誤處理) ===
        try:
            ema7 = self._ema(close_prices, 7)
            ema20 = self._ema(close_prices, 20)
            ema60 = self._ema(close_prices, 60)
            ema200 = self._ema(close_prices, 200)
            rsi = self._rsi(close_prices, 14)
            macd_line, macd_signal = self._macd(close_prices)
            obv = self._obv(close_prices, volumes)
            volume_ma7 = self._sma(volumes, 7)

            # 檢查指標陣列是否為空或長度不足
            indicators = [ema7, ema20, ema60, ema200, rsi, macd_line, macd_signal, obv, volume_ma7]
            if any(ind is None or len(ind) == 0 for ind in indicators):
                 raise ValueError("Indicator calculation resulted in empty array.")

            # 取最新值，處理 NaN
            latest_close = close_prices[-1]
            latest_ema7 = ema7[-1] if not np.isnan(ema7[-1]) else 0
            latest_ema20 = ema20[-1] if not np.isnan(ema20[-1]) else 0
            latest_ema60 = ema60[-1] if not np.isnan(ema60[-1]) else 0
            latest_ema200 = ema200[-1] if not np.isnan(ema200[-1]) else 0
            latest_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
            latest_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
            latest_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            latest_obv = obv[-1] if len(obv) > 0 else 0
            latest_volume_ma7 = volume_ma7[-1] if not np.isnan(volume_ma7[-1]) else 0

        except Exception as e:
            # print(f"Error calculating indicators at index {idx}: {e}. Returning zero state.") # Debug 用
            return np.zeros(self.state_dim)

        # === 特徵集合 ===
        safe_total_asset = max(self.total_asset, 1e-6) # 避免除以零
        cash_ratio = self.cash / safe_total_asset

        features = np.array([
            latest_close, latest_ema7, latest_ema20, latest_ema60, latest_ema200,
            latest_rsi, latest_macd, latest_macd_signal, latest_obv, latest_volume_ma7,
            float(self.position), cash_ratio # 確保 position 是 float
        ], dtype=np.float32)

        # === 正規化 ===
        norm_window_data = self.data.iloc[start_idx : idx + 1]
        if norm_window_data.empty or len(norm_window_data) <= 1:
            # print(f"Debug: Not enough data for normalization at index {idx}. Returning non-normalized features.") # Debug 用
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) # 處理 NaN
            # 確保維度
            if len(features) != self.state_dim:
                 padded_features = np.zeros(self.state_dim)
                 l = min(len(features), self.state_dim)
                 padded_features[:l] = features[:l]
                 return padded_features
            return features

        # --- 執行正規化 (與之前邏輯類似，確保健壯性) ---
        try:
            norm_closes = norm_window_data['close'].values
            norm_volumes = norm_window_data['volume'].values

            # 價格相關 Z-score
            mean_close = np.mean(norm_closes)
            std_close = np.std(norm_closes) + 1e-8
            features[0:5] = (features[0:5] - mean_close) / std_close

            # OBV Z-score
            window_obv = self._obv(norm_closes, norm_volumes)
            if len(window_obv) > 1:
                mean_obv = np.mean(window_obv)
                std_obv = np.std(window_obv) + 1e-8
                features[8] = (features[8] - mean_obv) / std_obv
            else: features[8] = 0

            # Volume MA Z-score
            window_vol_ma = self._sma(norm_volumes, 7)
            valid_vol_ma = window_vol_ma[~np.isnan(window_vol_ma)]
            if len(valid_vol_ma) > 1:
                mean_vol_ma = np.mean(valid_vol_ma)
                std_vol_ma = np.std(valid_vol_ma) + 1e-8
                features[9] = (features[9] - mean_vol_ma) / std_vol_ma if not np.isnan(features[9]) else 0
            else: features[9] = 0

            # RSI [-1, 1]
            features[5] = (features[5] - 50) / 50

            # MACD/Signal Z-score
            window_macd, window_signal = self._macd(norm_closes)
            valid_macd = window_macd[~np.isnan(window_macd)]
            valid_signal = window_signal[~np.isnan(window_signal)]
            if len(valid_macd) > 1:
                mean_macd = np.mean(valid_macd)
                std_macd = np.std(valid_macd) + 1e-8
                features[6] = (features[6] - mean_macd) / std_macd if not np.isnan(features[6]) else 0
            else: features[6] = 0
            if len(valid_signal) > 1:
                mean_signal = np.mean(valid_signal)
                std_signal = np.std(valid_signal) + 1e-8
                features[7] = (features[7] - mean_signal) / std_signal if not np.isnan(features[7]) else 0
            else: features[7] = 0

        except Exception as e:
            # print(f"Error during normalization at index {idx}: {e}. Returning zero state.") # Debug 用
            return np.zeros(self.state_dim)

        # 最後處理 NaN/Inf 並確保維度
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if len(features) != self.state_dim:
             padded_features = np.zeros(self.state_dim)
             l = min(len(features), self.state_dim)
             padded_features[:l] = features[:l]
             return padded_features

        return features

    # --- 技術指標計算函數 (保持不變) ---
    def _ema(self, data, period):
        if len(data) < period: return np.full_like(data, np.nan, dtype=float)
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def _sma(self, data, period):
        if len(data) < period: return np.full_like(data, np.nan, dtype=float)
        return pd.Series(data).rolling(window=period).mean().values

    def _rsi(self, data, period=14):
        if len(data) < period + 1: return np.full_like(data, np.nan, dtype=float)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # 使用 ewm 計算移動平均更穩定
        avg_gain = pd.Series(gain).ewm(com=period - 1, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(com=period - 1, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        # 填充頭部 NaN
        rsi_full = np.full_like(data, np.nan, dtype=float)
        rsi_full[period:] = rsi[period-1:] # 從第 period 個元素開始填充
        return rsi_full


    def _macd(self, data, fast=12, slow=26, signal=9):
        if len(data) < slow:
             return np.full_like(data, np.nan, dtype=float), np.full_like(data, np.nan, dtype=float)
        ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.values, macd_signal.values

    def _obv(self, close, volume):
        # 確保長度一致
        min_len = min(len(close), len(volume))
        close = close[:min_len]
        volume = volume[:min_len]
        if min_len < 2: return np.zeros(min_len, dtype=float)

        obv = np.zeros(min_len, dtype=float)
        obv_diff = np.diff(close)
        # 根據價格變化方向調整成交量符號
        vol_change = np.where(obv_diff > 0, volume[1:], np.where(obv_diff < 0, -volume[1:], 0))
        obv[1:] = np.cumsum(vol_change)
        return obv

    def render(self, mode='human'):
        """簡單渲染"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Asset: {self.total_asset:.2f}, Cash: {self.cash:.2f}, Holding: {self.holding:.4f}, Pos: {self.position}, Day Loss: {self.cumulative_loss_today:.2f}")

    def close(self):
        pass

# === Example Usage ===
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
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)

    env = TradingEnv(data, state_window_size=50)
    state = env.reset()
    print(f"Initial State Dim: {env.state_dim}, Asset: {env.total_asset:.2f}")

    done = False
    total_reward = 0
    steps = 0
    max_steps = len(data) - env.state_window_size - 1 # 確保 state 計算有足夠數據

    while not done and steps < max_steps :
        action = np.random.randint(0, 3) # 隨機動作
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
    print("Trade Log (last 5):", env.trade_log[-5:])
import pandas as pd
import numpy as np

class TradingEnv:
    def __init__(self, data, asset_name="BTC", state_window_size=100):
        # === 資料設定 ===
        self.data = data.copy() # 使用副本避免修改原始資料
        self.asset_name = asset_name
        self.state_window_size = state_window_size # 用於計算 state 的回看窗口

        # === 環境參數設定 ===
        self.initial_cash = 1000
        self.cash = self.initial_cash
        self.holding = 0 # 持有數量
        self.position = 0  # 1=多單, -1=空單, 0=無持倉
        self.entry_price = None
        self.stop_loss_price = None

        # === 手續費與滑點 ===
        self.commission_rate = 0.0005  # 0.05%
        self.slippage_rate = 0.002     # 0.2%

        # === 風險控管參數 ===
        self.single_trade_risk_pct = 0.02  # 單筆最大虧損佔總資產比例
        self.total_risk_pct = 0.05          # 總風險暴露（例如：所有持倉的潛在虧損總和不超過總資產的 5%）

        # === 持有時間處罰設定 ===
        # self.holding_penalty_rate = 0.0001 # 每步持有懲罰因子 (舊)
        self.holding_penalty_rate_profit = 0.00005 # 持有獲利部位的懲罰因子
        self.holding_penalty_rate_loss = 0.0002  # 持有虧損部位的懲罰因子
        self.no_position_penalty_rate = 0.0001 # 無倉位時的懲罰因子 (懲罰閒置資金)

        # === 績效停單設定 (Kill Switch) ===
        self.kill_switch_enabled = True
        self.win_rate_window = 10       # 計算勝率的回看交易次數
        self.min_win_rate = 0.3         # 最低可接受勝率
        self.max_losing_streak = 5      # 最大連續虧損次數
        # self.max_loss_pct = 0.10      # 最大總虧損比例 (相對於初始資金) - 這個比較適合全局控制，單日用下面的

        # === 每日虧損控制 (VaR-like) ===
        self.var_control_enabled = True
        self.max_daily_loss_pct = 0.03  # 單日最大虧損比例

        # === 冷卻期設定 ===
        self.cooldown_enabled = True
        self.cooldown_steps_required = 5 # 交易後的冷卻 K 棒數
        self.cooldown_steps_left = 0

        # === 內部追蹤變數 ===
        self.current_step = 0
        self.total_asset = self.initial_cash
        self.last_total_asset = self.initial_cash # 追蹤上一步的總資產
        self.holding_steps = 0 # 持倉 K 棒數
        self.trade_log = [] # 記錄交易 [(step, type, price, amount, profit/loss)]

        # === 績效追蹤 ===
        self.cumulative_loss_today = 0.0 # 當日累計虧損金額
        self.current_day = None # 當前日期，用於判斷是否換日
        self.win_flags = [] # 最近 N 筆交易的勝負紀錄 (1=win, 0=loss)
        self.loss_streak = 0 # 當前連續虧損次數

        # === State 維度 ===
        self.state_dim = 12 # 手動指定: close, ema7, ema20, ema60, ema200, rsi, macd, macd_signal, obv, volume_ma7, position, cash_ratio

        # === 預計算指標 (可選，加速) ===
        # self._precompute_indicators()


    # def _precompute_indicators(self):
    #     """預先計算所有時間點的技術指標，避免在 step 中重複計算"""
    #     # ... (implementation as before) ...


    def reset(self):
        """重置環境，開始新的一輪交易"""
        self.cash = self.initial_cash
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None

        self.current_step = 0 # 從頭開始
        self.total_asset = self.initial_cash
        self.last_total_asset = self.initial_cash # 重置上一步資產
        self.holding_steps = 0
        self.trade_log = []

        self.cumulative_loss_today = 0.0
        # 初始化 current_day
        if 'timestamp' in self.data.columns and len(self.data) > 0:
             try:
                 self.current_day = pd.to_datetime(self.data.iloc[0]['timestamp']).date()
             except Exception:
                 self.current_day = 0 # 轉換失敗
        else:
            self.current_day = 0 # 無 timestamp 或 data 為空

        self.win_flags = []
        self.loss_streak = 0
        self.cooldown_steps_left = 0

        return self._get_state()

    def step(self, action):
        """
        執行一步動作
        action: 0=持有/不做, 1=買入/做多, 2=賣出/做空
        """
        done = False
        info = {'trade_executed': False, 'reason': ''} # 額外資訊
        realized_pnl_this_step = 0 # 初始化這一步的已實現盈虧

        # 記錄上一步的總資產，用於計算變化
        self.last_total_asset = self.total_asset

        current_price = self.data.iloc[self.current_step]['close']
        current_high = self.data.iloc[self.current_step]['high']
        current_low = self.data.iloc[self.current_step]['low']

        # === 更新冷卻期 ===
        if self.cooldown_steps_left > 0:
            self.cooldown_steps_left -= 1

        # === 檢查每日是否切換，重設 cumulative_loss_today ===
        if 'timestamp' in self.data.columns:
            try:
                today = pd.to_datetime(self.data.iloc[self.current_step]['timestamp']).date()
                if today != self.current_day:
                    self.current_day = today
                    self.cumulative_loss_today = 0.0 # 重置當日虧損
            except Exception:
                pass # Timestamp 格式錯誤或其他問題

        # === 檢查停損 ===

            stop_loss_pnl = 0
            if self.position == 1 and self.stop_loss_price is not None and current_low <= self.stop_loss_price:
                trigger_price = self.stop_loss_price  # <--- 儲存觸發價格
                stop_loss_pnl = self._close_position(trigger_price) # 使用儲存的價格平倉
                info['trade_executed'] = True
                info['reason'] = f'Long Stop Loss triggered at {trigger_price:.4f}' # <--- 使用儲存的價格格式化
                self._update_performance_tracker(stop_loss_pnl)
                realized_pnl_this_step = stop_loss_pnl
            elif self.position == -1 and self.stop_loss_price is not None and current_high >= self.stop_loss_price:
                trigger_price = self.stop_loss_price  # <--- 儲存觸發價格
                stop_loss_pnl = self._close_position(trigger_price) # 使用儲存的價格平倉
                info['trade_executed'] = True
                info['reason'] = f'Short Stop Loss triggered at {trigger_price:.4f}' # <--- 使用儲存的價格格式化
                self._update_performance_tracker(stop_loss_pnl)
                realized_pnl_this_step = stop_loss_pnl


        # === 根據 action 執行交易 (如果未被停損觸發) ===
        action_pnl = None
        if not info['trade_executed']: # 只有在沒有觸發停損時才執行動作
            action_pnl = self._execute_action(action, current_price)
            if action_pnl is not None: # 如果有執行交易 (平倉)
                info['trade_executed'] = True
                # reason 會在 _execute_action 內部添加到 trade_log，這裡可以不用重複
                # info['reason'] = f'Action {action} executed closing position.'
                self._update_performance_tracker(action_pnl)
                realized_pnl_this_step = action_pnl # 記錄動作產生的 PnL
            elif self.position != 0 and action == 0: # 如果是持有不動
                 info['reason'] = 'Holding position.'
            elif self.position == 0 and action != 0: # 如果是嘗試開倉但失敗 (例如 kill switch)
                 # _execute_action 會返回 None，這裡可以加個原因
                 if not self._can_open_new_position():
                     info['reason'] = 'Open condition not met.'
                 else:
                     info['reason'] = 'Open action executed.' # 假設開倉成功
            else: # 其他情況 (例如無倉位時 action=0)
                 info['reason'] = 'No action taken.'


        # === 更新持有狀態和總資產 ===
        # 總資產計算現在依賴於正確的 self.position 和 self.holding
        self._update_holding_and_asset(current_price)

        # === 計算 reward ===
        # 將這一步驟中所有實現的 PnL 傳遞給 reward 函數
        reward = self._calculate_reward_v2(realized_pnl_this_step)

        # === 判斷是否 done ===
        # 1. 資料走完
        if self.current_step >= len(self.data) - 1:
            done = True
            info['reason'] += ' End of data reached.'
        # 2. 總資產過低
        if self.total_asset < self.initial_cash * 0.5:
            done = True
            info['reason'] += f' Total asset too low ({self.total_asset:.2f}).'
        # 3. Kill Switch 觸發 (可選，如果希望 Kill Switch 直接結束 episode)
        # if self.kill_switch_enabled and not self._check_kill_switch_ok():
        #    done = True
        #    info['reason'] += ' Kill switch triggered episode end.'

        # === 前進一步 ===
        self.current_step += 1

        return self._get_state(), reward, done, info

    def _execute_action(self, action, current_price):
        """
        根據 agent 的動作執行交易邏輯
        action: 0=持有/不做, 1=買入/做多, 2=賣出/做空
        返回: 該交易實現的盈虧 (pnl)，僅在平倉時有值，否則返回 None
        """
        executed_pnl = None

        # --- 平倉邏輯 ---
        if self.position == 1 and action == 2:
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Long', current_price, self.holding, executed_pnl))
        elif self.position == -1 and action == 1:
            executed_pnl = self._close_position(current_price)
            self.trade_log.append((self.current_step, 'Close Short', current_price, self.holding, executed_pnl))

        # --- 開倉邏輯 (僅在當前無持倉且滿足條件時) ---
        elif self.position == 0:
            if action == 1: # 開多倉
                if self._can_open_new_position():
                    # 計算止損價格
                    self.stop_loss_price = current_price * (1 - self.single_trade_risk_pct)
                    risk_per_unit = current_price - self.stop_loss_price
                    if risk_per_unit <= 0: # 避免除以零或負數
                        return None # 無法計算倉位大小

                    # 計算基於風險的倉位大小
                    max_loss_amount = self.total_asset * self.single_trade_risk_pct
                    position_size = max_loss_amount / risk_per_unit

                    # 計算實際成本並檢查現金是否足夠
                    cost = position_size * current_price * (1 + self.commission_rate + self.slippage_rate)
                    if self.cash >= cost and position_size > 0:
                        # 確保倉位大小不超過現金能買的量 (以防萬一)
                        max_affordable_size = self.cash / (current_price * (1 + self.commission_rate + self.slippage_rate))
                        final_position_size = min(position_size, max_affordable_size)

                        if final_position_size > 0:
                            final_cost = final_position_size * current_price * (1 + self.commission_rate + self.slippage_rate)
                            self.cash -= final_cost
                            self.holding = final_position_size
                            self.position = 1
                            self.entry_price = current_price
                            self.holding_steps = 0
                            # 重新計算止損價以匹配實際倉位 (可選，但更精確)
                            # self.stop_loss_price = current_price - (max_loss_amount / final_position_size)
                            self.trade_log.append((self.current_step, 'Open Long', current_price, final_position_size, 0))
            elif action == 2: # 開空倉
                 if self._can_open_new_position():
                    # 計算止損價格
                    self.stop_loss_price = current_price * (1 + self.single_trade_risk_pct)
                    risk_per_unit = self.stop_loss_price - current_price
                    if risk_per_unit <= 0:
                        return None

                    # 計算基於風險的倉位大小
                    max_loss_amount = self.total_asset * self.single_trade_risk_pct
                    position_size = max_loss_amount / risk_per_unit

                    # 計算保證金和費用
                    margin_required = position_size * current_price # 簡化保證金計算
                    commission_cost = position_size * current_price * (self.commission_rate + self.slippage_rate)
                    total_cash_needed = margin_required + commission_cost # 這裡假設保證金直接從現金扣除 (不完全準確，但作為簡化)

                    if self.cash >= total_cash_needed and position_size > 0:
                        # 確保倉位大小不超過現金能支持的量
                        # 這裡的檢查比較複雜，因為涉及保證金，暫時簡化
                        max_affordable_size = self.cash / (current_price * (1 + self.commission_rate + self.slippage_rate)) # 粗略估計
                        final_position_size = min(position_size, max_affordable_size)

                        if final_position_size > 0:
                            final_commission_cost = final_position_size * current_price * (self.commission_rate + self.slippage_rate)
                            # 實際操作中，保證金是鎖定，不是直接扣除，這裡為了簡化先扣費用
                            self.cash -= final_commission_cost
                            self.holding = final_position_size
                            self.position = -1
                            self.entry_price = current_price
                            self.holding_steps = 0
                            # 重新計算止損價
                            # self.stop_loss_price = current_price + (max_loss_amount / final_position_size)
                            self.trade_log.append((self.current_step, 'Open Short', current_price, final_position_size, 0))

        return executed_pnl # 只有平倉時返回 PnL

    def _close_position(self, close_price):
        """
        平掉目前的倉位，計算並返回盈虧
        close_price: 平倉價格
        """
        pnl = 0
        if self.position == 1:
            revenue = self.holding * close_price * (1 - self.commission_rate - self.slippage_rate)
            initial_cost = self.holding * self.entry_price # 用於計算 PnL
            pnl = revenue - initial_cost
            self.cash += revenue
        elif self.position == -1:
            price_diff = self.entry_price - close_price
            gross_pnl = price_diff * self.holding
            commission_cost = self.holding * close_price * (self.commission_rate + self.slippage_rate)
            pnl = gross_pnl - commission_cost
            # 簡化現金返還：返還名義價值 + PnL
            cash_change = (self.holding * self.entry_price) + pnl
            self.cash += cash_change

        # 重置倉位信息
        self.holding = 0
        self.position = 0
        self.entry_price = None
        self.stop_loss_price = None
        if self.cooldown_enabled:
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
            self.total_asset += self.holding * current_price * (1 - self.commission_rate) # 考慮平倉手續費的潛在價值
        elif self.position == -1:
            # 空單市值 = 開倉名義價值 + 未實現盈虧 - 預計平倉費用
            unrealized_pnl = (self.entry_price - current_price) * self.holding
            estimated_close_commission = self.holding * current_price * self.commission_rate
            position_value = (self.entry_price * self.holding) + unrealized_pnl - estimated_close_commission
            self.total_asset += position_value

    # 移除 _calculate_reward_v1

    def _calculate_reward_v2(self, realized_pnl):
        """
        計算 reward v2：基於已實現盈虧 + 未實現盈虧變化 - 持有懲罰
        realized_pnl: 這一步驟中交易實現的盈虧
        """
        # 1. 已實現盈虧 Reward
        realized_reward = realized_pnl / self.initial_cash * 10 # 放大 PnL 影響

        # 2. 未實現盈虧變化 Reward (基於上一步到這一步的資產變化)
        # 使用 total_asset 的變化來反映未實現盈虧變化和現金變化
        # 注意：這裡的 total_asset 已經包含了 realized_pnl 的影響，所以需要調整
        # asset_change = self.total_asset - self.last_total_asset
        # unrealized_reward = (asset_change - realized_pnl) / self.initial_cash # 僅考慮非交易導致的資產變化

        # 或者，直接計算未實現盈虧變化（如果倉位未變）
        unrealized_reward = 0
        if self.position != 0 and realized_pnl == 0 and self.current_step > 0: # 僅在持有且未交易時計算
            current_price = self.data.iloc[self.current_step]['close']
            last_price = self.data.iloc[self.current_step - 1]['close']
            price_change = current_price - last_price
            if self.position == 1:
                unrealized_pnl_change = self.holding * price_change
            elif self.position == -1:
                unrealized_pnl_change = self.holding * (-price_change)
            unrealized_reward = unrealized_pnl_change / self.initial_cash

        # 3. 持有懲罰
        holding_penalty = 0
        if self.position != 0:
            holding_penalty = self.holding_penalty_rate * self.holding_steps

        # 4. 總 Reward
        reward = realized_reward + unrealized_reward - holding_penalty

        # === 更新每日虧損累計（給 VaR 控制用）===
        # 計算自上次換日以來的總資產變化
        daily_pnl = self.total_asset - self.last_total_asset # 這一步的資產變化
        if daily_pnl < 0:
            self.cumulative_loss_today += abs(daily_pnl)

        return reward


    def _update_performance_tracker(self, pnl):
        """當一筆交易完成時，更新績效追蹤指標"""
        if pnl > 0:
            self.win_flags.append(1)
            self.loss_streak = 0 # 中斷連敗
        elif pnl < 0: # 只記錄虧損交易
            self.win_flags.append(0)
            self.loss_streak += 1 # 增加連敗次數
        # pnl == 0 (例如手續費剛好抵消) 不計入勝負

        # 維護勝率計算窗口
        if len(self.win_flags) > self.win_rate_window:
            self.win_flags.pop(0)

    def _check_kill_switch_ok(self):
        """檢查是否觸發 Kill Switch 條件"""
        if not self.kill_switch_enabled:
            return True

        # 1. 檢查勝率
        if len(self.win_flags) >= self.win_rate_window:
            current_win_rate = sum(self.win_flags) / len(self.win_flags)
            if current_win_rate < self.min_win_rate:
                return False

        # 2. 檢查最大連敗次數
        if self.loss_streak >= self.max_losing_streak:
            return False

        return True

    def _can_open_new_position(self):
        """檢查是否滿足所有開倉條件"""
        # 1. 檢查冷卻期
        if self.cooldown_enabled and self.cooldown_steps_left > 0:
            return False

        # 2. 檢查 Kill Switch
        if not self._check_kill_switch_ok():
            return False

        # 3. 檢查每日虧損限制 (VaR)
        if self.var_control_enabled and self.cumulative_loss_today >= self.initial_cash * self.max_daily_loss_pct:
            # print(f"Cannot open: Daily loss limit reached ({self.cumulative_loss_today:.2f})")
            return False

        # 4. 檢查總風險暴露 (如果需要實現)
        # if not self._check_total_risk():
        #     return False

        return True

    # def _calculate_position_size(self, entry_price, stop_loss_price):
    #     # ... (implementation as before) ...

    # def _check_total_risk(self):
    #     # ... (implementation as before) ...

    def _get_state(self):
        """整理一份 observation，給 agent 使用"""
        idx = self.current_step

        if idx < self.state_window_size:
            return np.zeros(self.state_dim) # 資料不足

        # === 抓取數據 & 計算指標 ===
        start_idx = max(0, idx - self.state_window_size)
        indicator_start_idx = max(0, idx - 250) # 給指標計算留足夠空間
        data_slice = self.data.iloc[indicator_start_idx:idx+1] # 包含當前 step 的數據

        close_prices = data_slice['close'].values
        volumes = data_slice['volume'].values

        ema7 = self._ema(close_prices, 7)
        ema20 = self._ema(close_prices, 20)
        ema60 = self._ema(close_prices, 60)
        ema200 = self._ema(close_prices, 200)
        rsi = self._rsi(close_prices, 14)
        macd_line, macd_signal = self._macd(close_prices)
        obv = self._obv(close_prices, volumes)
        volume_ma7 = self._sma(volumes, 7)

        # === 取當前時間點的最新指標值 ===
        latest_close = close_prices[-1]
        latest_ema7 = ema7[-1] if not np.isnan(ema7[-1]) else 0
        latest_ema20 = ema20[-1] if not np.isnan(ema20[-1]) else 0
        latest_ema60 = ema60[-1] if not np.isnan(ema60[-1]) else 0
        latest_ema200 = ema200[-1] if not np.isnan(ema200[-1]) else 0
        latest_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50 # RSI 中值
        latest_macd = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        latest_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
        latest_obv = obv[-1]
        latest_volume_ma7 = volume_ma7[-1] if not np.isnan(volume_ma7[-1]) else 0

        # === 特徵集合 ===
        safe_total_asset = self.total_asset if self.total_asset != 0 else 1e-6
        cash_ratio = self.cash / safe_total_asset

        features = np.array([
            latest_close, latest_ema7, latest_ema20, latest_ema60, latest_ema200,
            latest_rsi, latest_macd, latest_macd_signal, latest_obv, latest_volume_ma7,
            self.position, cash_ratio
        ], dtype=np.float32)

        # === 正規化 ===
        norm_window_data = self.data.iloc[start_idx:idx+1] # 包含當前 step
        norm_closes = norm_window_data['close'].values

        if len(norm_closes) > 1:
            mean_close = np.mean(norm_closes)
            std_close = np.std(norm_closes) + 1e-8
            # 正規化價格相關特徵 (indices 0-4)
            features[0:5] = (features[0:5] - mean_close) / std_close
            # 正規化 OBV (index 8) - 可以用 Z-score
            window_obv = self._obv(norm_closes, norm_window_data['volume'].values)
            mean_obv = np.mean(window_obv)
            std_obv = np.std(window_obv) + 1e-8
            features[8] = (features[8] - mean_obv) / std_obv
            # 正規化 Volume MA (index 9) - 可以用 Z-score
            window_vol_ma = self._sma(norm_window_data['volume'].values, 7)
            mean_vol_ma = np.nanmean(window_vol_ma) # 使用 nanmean
            std_vol_ma = np.nanstd(window_vol_ma) + 1e-8
            features[9] = (features[9] - mean_vol_ma) / std_vol_ma

        # 正規化 RSI (index 5) 到 [-1, 1]
        features[5] = (features[5] - 50) / 50
        # 正規化 MACD/Signal (indices 6, 7) - 可以用窗口 Z-score
        window_macd, window_signal = self._macd(norm_closes)
        mean_macd = np.nanmean(window_macd)
        std_macd = np.nanstd(window_macd) + 1e-8
        mean_signal = np.nanmean(window_signal)
        std_signal = np.nanstd(window_signal) + 1e-8
        features[6] = (features[6] - mean_macd) / std_macd
        features[7] = (features[7] - mean_signal) / std_signal

        # Position (10) 和 Cash Ratio (11) 通常不需要額外正規化

        # 處理 NaN 或 Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # 確保維度正確
        if len(features) != self.state_dim:
             padded_features = np.zeros(self.state_dim)
             l = min(len(features), self.state_dim)
             padded_features[:l] = features[:l]
             return padded_features

        return features

    # --- 技術指標計算函數 ---
    # (EMA, SMA, RSI, MACD, OBV, ATR - implementation as before)
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
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_full = np.full_like(data, np.nan, dtype=float)
        rsi_full[period:] = rsi[period-1:]
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
        obv = np.zeros_like(close, dtype=float)
        if len(close) < 2: return obv
        # 確保 volume 和 close 長度一致
        min_len = min(len(close), len(volume))
        close = close[:min_len]
        volume = volume[:min_len]
        obv = obv[:min_len]

        # 計算 OBV
        obv_diff = np.diff(close)
        vol_change = np.zeros_like(volume, dtype=float)
        vol_change[1:][obv_diff > 0] = volume[1:][obv_diff > 0]
        vol_change[1:][obv_diff < 0] = -volume[1:][obv_diff < 0]
        obv = np.cumsum(vol_change)

        # 如果原始數據長度不同，需要處理返回值的長度
        # 這裡假設返回與 close 同長度的 OBV
        if len(obv) < len(close):
            obv_full = np.zeros_like(close, dtype=float)
            obv_full[:len(obv)] = obv
            return obv_full
        return obv


    def _calculate_atr(self, high, low, close, period=14):
        if len(high) < period + 1: return np.full_like(high, np.nan, dtype=float)
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([high_low, high_close, low_close]) # 使用 reduce
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        return atr

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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析引擎模块 - 负责对数字货币数据进行分析和计算技术指标
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Analyzer:
    """
    分析引擎 - 计算技术指标和进行市场分析
    """
    
    def __init__(self, database):
        """
        初始化分析引擎
        
        Args:
            database: 数据库实例，用于获取和存储数据
        """
        self.db = database
    
    def calculate_indicators(self, exchange, symbol, timeframe='1d'):
        """
        计算指定交易对的技术指标
        
        Args:
            exchange: 交易所名称
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            计算成功返回True，否则返回False
        """
        try:
            # 获取K线数据
            ohlcv_data = self.db.get_ohlcv_data(exchange, symbol, timeframe, limit=200)
            
            if not ohlcv_data:
                logger.warning(f"没有找到{exchange}交易所{symbol}的{timeframe}周期数据")
                return False
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp')
            
            # 计算移动平均线
            self._calculate_ma(df, exchange, symbol, timeframe)
            
            # 计算MACD
            self._calculate_macd(df, exchange, symbol, timeframe)
            
            # 计算RSI
            self._calculate_rsi(df, exchange, symbol, timeframe)
            
            # 计算布林带
            self._calculate_bollinger_bands(df, exchange, symbol, timeframe)
            
            # 计算成交量指标
            self._calculate_volume_indicators(df, exchange, symbol, timeframe)
            
            # 计算KDJ指标
            self._calculate_kdj(df, exchange, symbol, timeframe)
            
            # 计算趋势强度指标
            self._calculate_trend_strength(df, exchange, symbol, timeframe)
            
            # 检测MACD交叉信号
            self._detect_macd_crossover(df, exchange, symbol, timeframe)
            
            logger.info(f"已计算{exchange}交易所{symbol}的{timeframe}周期技术指标")
            return True
            
        except Exception as e:
            logger.error(f"计算技术指标出错: {str(e)}")
            return False
    
    def _calculate_ma(self, df, exchange, symbol, timeframe):
        """
        计算移动平均线
        """
        # 计算不同周期的MA
        for period in [5, 10, 20, 30, 60]:
            ma_column = f'ma_{period}'
            df[ma_column] = df['close'].rolling(window=period).mean()
            
            # 保存到数据库
            for idx, row in df.iterrows():
                if pd.notna(row[ma_column]):
                    indicator_data = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': row['timestamp'],
                        'indicator_name': f'MA{period}',
                        'value': row[ma_column],
                        'parameters': {'period': period}
                    }
                    self.db.save_indicator(indicator_data)
    
    def _calculate_macd(self, df, exchange, symbol, timeframe):
        """
        计算MACD指标
        """
        # 计算EMA
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # 计算MACD线和信号线
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        # 保存到数据库
        for idx, row in df.iterrows():
            if pd.notna(row['macd']) and pd.notna(row['signal']) and pd.notna(row['histogram']):
                # 保存MACD线
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'MACD',
                    'value': row['macd'],
                    'parameters': {'type': 'line'}
                })
                
                # 保存信号线
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'MACD',
                    'value': row['signal'],
                    'parameters': {'type': 'signal'}
                })
                
                # 保存柱状图
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'MACD',
                    'value': row['histogram'],
                    'parameters': {'type': 'histogram'}
                })
    
    def _calculate_rsi(self, df, exchange, symbol, timeframe):
        """
        计算RSI指标
        """
        for period in [6, 14, 21]:
            # 计算价格变化
            delta = df['close'].diff()
            
            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 计算平均上涨和下跌
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # 计算相对强度
            rs = avg_gain / avg_loss
            
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
            
            # 保存到数据库
            for idx, row in df.iterrows():
                if pd.notna(rsi.iloc[idx]):
                    self.db.save_indicator({
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': row['timestamp'],
                        'indicator_name': 'RSI',
                        'value': rsi.iloc[idx],
                        'parameters': {'period': period}
                    })
    
    def _calculate_bollinger_bands(self, df, exchange, symbol, timeframe):
        """
        计算布林带指标
        """
        period = 20
        std_dev = 2
        
        # 计算移动平均线
        df['middle_band'] = df['close'].rolling(window=period).mean()
        
        # 计算标准差
        df['std'] = df['close'].rolling(window=period).std()
        
        # 计算上下轨
        df['upper_band'] = df['middle_band'] + (df['std'] * std_dev)
        df['lower_band'] = df['middle_band'] - (df['std'] * std_dev)
        
        # 保存到数据库
        for idx, row in df.iterrows():
            if pd.notna(row['middle_band']):
                # 保存中轨
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'BOLL',
                    'value': row['middle_band'],
                    'parameters': {'type': 'middle', 'period': period, 'std_dev': std_dev}
                })
                
                # 保存上轨
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'BOLL',
                    'value': row['upper_band'],
                    'parameters': {'type': 'upper', 'period': period, 'std_dev': std_dev}
                })
                
                # 保存下轨
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'BOLL',
                    'value': row['lower_band'],
                    'parameters': {'type': 'lower', 'period': period, 'std_dev': std_dev}
                })
    
    def _calculate_volume_indicators(self, df, exchange, symbol, timeframe):
        """
        计算成交量相关指标
        """
        # 计算成交量移动平均
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # 保存到数据库
        for idx, row in df.iterrows():
            if pd.notna(row['volume_ma']):
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'VOLUME_MA',
                    'value': row['volume_ma'],
                    'parameters': {'period': 20}
                })
    
    def _calculate_kdj(self, df, exchange, symbol, timeframe):
        """
        计算KDJ指标
        """
        # 设置KDJ参数
        period = 9
        k_period = 3
        d_period = 3
        
        # 计算最高价和最低价的N日内最高值和最低值
        df['low_min'] = df['low'].rolling(window=period).min()
        df['high_max'] = df['high'].rolling(window=period).max()
        
        # 计算RSV值
        df['rsv'] = 100 * ((df['close'] - df['low_min']) / 
                         (df['high_max'] - df['low_min']).replace(0, 0.0001))
        
        # 计算K值、D值和J值
        df['k'] = df['rsv'].rolling(window=k_period).mean()
        df['d'] = df['k'].rolling(window=d_period).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # 保存到数据库
        for idx, row in df.iterrows():
            if pd.notna(row['k']) and pd.notna(row['d']) and pd.notna(row['j']):
                # 保存K值
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'KDJ',
                    'value': row['k'],
                    'parameters': {'type': 'K', 'period': period}
                })
                
                # 保存D值
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'KDJ',
                    'value': row['d'],
                    'parameters': {'type': 'D', 'period': period}
                })
                
                # 保存J值
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'KDJ',
                    'value': row['j'],
                    'parameters': {'type': 'J', 'period': period}
                })
    
    def _detect_macd_crossover(self, df, exchange, symbol, timeframe):
        """
        检测MACD交叉信号
        """
        # 确保MACD已经计算
        if 'macd' not in df.columns or 'signal' not in df.columns:
            # 计算EMA
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # 计算MACD线和信号线
            df['macd'] = df['ema12'] - df['ema26']
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 创建交叉信号列
        df['macd_cross'] = 0
        
        # 检测金叉（MACD线从下方穿过信号线）
        golden_cross = (df['macd'].shift(1) < df['signal'].shift(1)) & (df['macd'] > df['signal'])
        df.loc[golden_cross, 'macd_cross'] = 1
        
        # 检测死叉（MACD线从上方穿过信号线）
        death_cross = (df['macd'].shift(1) > df['signal'].shift(1)) & (df['macd'] < df['signal'])
        df.loc[death_cross, 'macd_cross'] = -1
        
        # 保存交叉信号到数据库
        for idx, row in df.iterrows():
            if row['macd_cross'] != 0:
                signal_type = 'golden_cross' if row['macd_cross'] == 1 else 'death_cross'
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'MACD_CROSS',
                    'value': row['macd_cross'],
                    'parameters': {'type': signal_type}
                })
    
    def _calculate_trend_strength(self, df, exchange, symbol, timeframe):
        """
        计算趋势强度指标
        """
        # 计算价格变化率
        df['price_change'] = df['close'].pct_change()
        
        # 计算方向移动指标 (DMI)
        period = 14
        
        # 计算真实波幅 (TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算方向移动 (DM)
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        # 计算正方向移动 (+DM) 和负方向移动 (-DM)
        df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # 计算平滑值
        df['+di'] = 100 * (df['+dm'].rolling(window=period).sum() / df['tr'].rolling(window=period).sum())
        df['-di'] = 100 * (df['-dm'].rolling(window=period).sum() / df['tr'].rolling(window=period).sum())
        
        # 计算方向指数 (DX)
        df['dx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']).replace(0, 0.0001))
        
        # 计算平均方向指数 (ADX)
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # 保存到数据库
        for idx, row in df.iterrows():
            if pd.notna(row['adx']):
                # 保存ADX
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'ADX',
                    'value': row['adx'],
                    'parameters': {'period': period}
                })
                
                # 保存+DI
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'DI',
                    'value': row['+di'],
                    'parameters': {'type': 'positive', 'period': period}
                })
                
                # 保存-DI
                self.db.save_indicator({
                    'exchange': exchange,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['timestamp'],
                    'indicator_name': 'DI',
                    'value': row['-di'],
                    'parameters': {'type': 'negative', 'period': period}
                })
    
    def get_market_trend(self, exchange, symbol, timeframe='1d'):
        """
        获取市场趋势分析
        
        Args:
            exchange: 交易所名称
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            趋势分析结果字典
        """
        try:
            # 获取最近的K线数据
            ohlcv_data = self.db.get_ohlcv_data(exchange, symbol, timeframe, limit=30)
            
            if not ohlcv_data:
                return {'trend': 'unknown', 'strength': 0, 'message': '没有足够的数据'}
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp')
            
            # 获取最近的MA指标
            ma20_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'MA20', limit=30
            )
            
            if not ma20_data:
                # 如果没有计算好的指标，临时计算
                df['ma20'] = df['close'].rolling(window=20).mean()
            else:
                # 使用已计算的指标
                ma_df = pd.DataFrame(ma20_data)
                ma_df = ma_df.sort_values('timestamp')
                df['ma20'] = ma_df['value'].values
            
            # 获取最近的RSI指标
            rsi_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'RSI', limit=30
            )
            
            if rsi_data:
                rsi_df = pd.DataFrame(rsi_data)
                rsi_df = rsi_df.sort_values('timestamp')
                df['rsi'] = rsi_df['value'].values
            
            # 分析趋势
            trend = self._analyze_trend(df)
            
            return trend
            
        except Exception as e:
            logger.error(f"获取市场趋势分析出错: {str(e)}")
            return {'trend': 'unknown', 'strength': 0, 'message': f'分析出错: {str(e)}'}
    
    def get_market_sentiment(self, exchange, symbol, timeframe='1d'):
        """
        获取市场情绪分析
        
        Args:
            exchange: 交易所名称
            symbol: 交易对
            timeframe: 时间周期
            
        Returns:
            市场情绪分析结果字典
        """
        try:
            # 获取最近的K线数据
            ohlcv_data = self.db.get_ohlcv_data(exchange, symbol, timeframe, limit=30)
            
            if not ohlcv_data or len(ohlcv_data) < 10:
                return {'sentiment': 'unknown', 'score': 50, 'message': '没有足够的数据'}
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp')
            
            # 获取各种技术指标
            indicators = {}
            
            # 获取RSI指标
            rsi_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'RSI', 
                parameters={'period': 14}, limit=10
            )
            if rsi_data:
                indicators['rsi'] = rsi_data[-1]['value']
            
            # 获取MACD指标
            macd_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'MACD', 
                parameters={'type': 'histogram'}, limit=10
            )
            if macd_data:
                indicators['macd_histogram'] = [item['value'] for item in macd_data]
            
            # 获取KDJ指标
            kdj_j_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'KDJ', 
                parameters={'type': 'J'}, limit=10
            )
            if kdj_j_data:
                indicators['kdj_j'] = kdj_j_data[-1]['value']
            
            # 获取布林带指标
            boll_middle_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'BOLL', 
                parameters={'type': 'middle'}, limit=1
            )
            boll_upper_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'BOLL', 
                parameters={'type': 'upper'}, limit=1
            )
            boll_lower_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'BOLL', 
                parameters={'type': 'lower'}, limit=1
            )
            
            if boll_middle_data and boll_upper_data and boll_lower_data:
                indicators['boll_middle'] = boll_middle_data[0]['value']
                indicators['boll_upper'] = boll_upper_data[0]['value']
                indicators['boll_lower'] = boll_lower_data[0]['value']
            
            # 获取ADX指标
            adx_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'ADX', limit=1
            )
            if adx_data:
                indicators['adx'] = adx_data[0]['value']
            
            # 获取MACD交叉信号
            macd_cross_data = self.db.get_indicator_data(
                exchange, symbol, timeframe, 'MACD_CROSS', limit=5
            )
            if macd_cross_data:
                indicators['macd_cross'] = [item['value'] for item in macd_cross_data]
            
            # 分析市场情绪
            sentiment = self._analyze_market_sentiment(df, indicators)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"获取市场情绪分析出错: {str(e)}")
            return {'sentiment': 'unknown', 'score': 50, 'message': f'分析出错: {str(e)}'}
    
    def _analyze_trend(self, df):
        """
        分析价格趋势
        
        Args:
            df: 包含价格和指标数据的DataFrame
            
        Returns:
            趋势分析结果字典
        """
        result = {'trend': 'neutral', 'strength': 0, 'signals': []}
        
        # 检查数据是否足够
        if len(df) < 10:
            result['message'] = '数据不足，无法进行可靠分析'
            return result
        
        # 获取最近的收盘价
        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # 计算短期价格变化
        price_change = (latest_close - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
        
        # 检查价格与MA的关系
        if 'ma20' in df.columns and not pd.isna(df['ma20'].iloc[-1]):
            ma20 = df['ma20'].iloc[-1]
            if latest_close > ma20:
                result['signals'].append('价格位于20日均线上方')
                result['strength'] += 1
            else:
                result['signals'].append('价格位于20日均线下方')
                result['strength'] -= 1
        
        # 检查RSI指标
        if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
            rsi = df['rsi'].iloc[-1]
            if rsi > 70:
                result['signals'].append('RSI超买(>70)')
                result['strength'] -= 1
            elif rsi < 30:
                result['signals'].append('RSI超卖(<30)')
                result['strength'] += 1
        
        # 检查价格趋势
        if price_change > 5:
            result['signals'].append(f'5日价格上涨{price_change:.2f}%')
            result['strength'] += 2
        elif price_change < -5:
            result['signals'].append(f'5日价格下跌{abs(price_change):.2f}%')
            result['strength'] -= 2
        
        # 确定整体趋势
        if result['strength'] >= 2:
            result['trend'] = 'bullish'
            result['message'] = '市场呈现上涨趋势'
        elif result['strength'] <= -2:
            result['trend'] = 'bearish'
            result['message'] = '市场呈现下跌趋势'
        else:
            result['trend'] = 'neutral'
            result['message'] = '市场处于盘整状态'
        
        return result
    
    def _analyze_market_sentiment(self, df, indicators):
        """
        分析市场情绪
        
        Args:
            df: 包含价格数据的DataFrame
            indicators: 包含各种技术指标的字典
            
        Returns:
            市场情绪分析结果字典
        """
        # 初始化结果
        result = {
            'sentiment': 'neutral',  # 情绪: 极度恐慌/恐慌/中性/贪婪/极度贪婪
            'score': 50,  # 情绪得分: 0-100, 0表示极度恐慌，100表示极度贪婪
            'signals': [],  # 信号列表
            'message': ''  # 分析信息
        }
        
        # 计算情绪得分
        score_components = []
        
        # 1. RSI指标评分 (0-100)
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            # RSI评分: RSI直接作为得分组件之一
            score_components.append(rsi)
            
            # 添加RSI信号
            if rsi > 70:
                result['signals'].append(f'RSI超买({rsi:.2f})')
            elif rsi < 30:
                result['signals'].append(f'RSI超卖({rsi:.2f})')
            else:
                result['signals'].append(f'RSI中性({rsi:.2f})')
        
        # 2. MACD柱状图评分
        if 'macd_histogram' in indicators and len(indicators['macd_histogram']) > 0:
            # 获取最近的MACD柱状图值
            recent_hist = indicators['macd_histogram'][-5:]
            
            # 计算柱状图趋势
            hist_trend = sum(1 for h in recent_hist if h > 0) / len(recent_hist) * 100
            score_components.append(hist_trend)
            
            # 添加MACD信号
            if hist_trend > 60:
                result['signals'].append('MACD柱状图多头趋势')
            elif hist_trend < 40:
                result['signals'].append('MACD柱状图空头趋势')
        
        # 3. KDJ指标评分
        if 'kdj_j' in indicators:
            j_value = indicators['kdj_j']
            # J值评分: 0-100之间，直接使用
            j_score = max(0, min(100, j_value))
            score_components.append(j_score)
            
            # 添加KDJ信号
            if j_value > 80:
                result['signals'].append(f'KDJ超买(J={j_value:.2f})')
            elif j_value < 20:
                result['signals'].append(f'KDJ超卖(J={j_value:.2f})')
        
        # 4. 布林带位置评分
        if all(k in indicators for k in ['boll_middle', 'boll_upper', 'boll_lower']):
            # 获取最新收盘价
            latest_close = df['close'].iloc[-1]
            
            # 计算价格在布林带中的位置 (0-100)
            band_width = indicators['boll_upper'] - indicators['boll_lower']
            if band_width > 0:
                position = (latest_close - indicators['boll_lower']) / band_width * 100
                position = max(0, min(100, position))
                score_components.append(position)
                
                # 添加布林带信号
                if position > 80:
                    result['signals'].append('价格接近布林带上轨')
                elif position < 20:
                    result['signals'].append('价格接近布林带下轨')
        
        # 5. ADX趋势强度评分
        if 'adx' in indicators:
            adx = indicators['adx']
            # ADX评分: 将ADX值映射到0-100
            adx_score = min(100, adx)
            score_components.append(adx_score)
            
            # 添加ADX信号
            if adx > 25:
                result['signals'].append(f'强趋势市场(ADX={adx:.2f})')
            else:
                result['signals'].append(f'弱趋势市场(ADX={adx:.2f})')
        
        # 6. MACD交叉信号评分
        if 'macd_cross' in indicators and len(indicators['macd_cross']) > 0:
            # 检查最近是否有金叉或死叉
            recent_cross = indicators['macd_cross'][-1]
            if recent_cross == 1:
                result['signals'].append('MACD最近形成金叉')
                score_components.append(75)  # 金叉给予较高的贪婪得分
            elif recent_cross == -1:
                result['signals'].append('MACD最近形成死叉')
                score_components.append(25)  # 死叉给予较低的恐慌得分
        
        # 7. 价格趋势评分
        if len(df) >= 10:
            # 计算10日价格变化率
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
            
            # 价格变化率映射到0-100的评分
            price_score = 50 + price_change * 2  # 每1%的变化对应2分
            price_score = max(0, min(100, price_score))
            score_components.append(price_score)
            
            # 添加价格趋势信号
            if price_change > 10:
                result['signals'].append(f'10日价格大幅上涨({price_change:.2f}%)')
            elif price_change > 5:
                result['signals'].append(f'10日价格上涨({price_change:.2f}%)')
            elif price_change < -10:
                result['signals'].append(f'10日价格大幅下跌({abs(price_change):.2f}%)')
            elif price_change < -5:
                result['signals'].append(f'10日价格下跌({abs(price_change):.2f}%)')
        
        # 计算最终情绪得分 (各组件的平均值)
        if score_components:
            result['score'] = sum(score_components) / len(score_components)
        
        # 根据得分确定情绪状态
        if result['score'] >= 80:
            result['sentiment'] = '极度贪婪'
            result['message'] = '市场情绪极度乐观，可能存在过热风险'
        elif result['score'] >= 60:
            result['sentiment'] = '贪婪'
            result['message'] = '市场情绪偏向乐观'
        elif result['score'] <= 20:
            result['sentiment'] = '极度恐慌'
            result['message'] = '市场情绪极度悲观，可能存在超卖机会'
        elif result['score'] <= 40:
            result['sentiment'] = '恐慌'
            result['message'] = '市场情绪偏向悲观'
        else:
            result['sentiment'] = '中性'
            result['message'] = '市场情绪中性平衡'
        
        return result
    
    def get_price_prediction(self, exchange, symbol, timeframe='1d', days=7, model_type='ensemble'):
        """
        价格预测功能
        
        Args:
            exchange: 交易所名称
            symbol: 交易对
            timeframe: 时间周期
            days: 预测天数
            model_type: 预测模型类型 (linear, arima, prophet, ensemble)
            
        Returns:
            预测结果字典
        """
        try:
            # 获取历史K线数据
            ohlcv_data = self.db.get_ohlcv_data(exchange, symbol, timeframe, limit=120)
            
            if not ohlcv_data or len(ohlcv_data) < 60:
                return {'status': 'error', 'message': '没有足够的历史数据进行预测'}
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv_data)
            df = df.sort_values('timestamp')
            
            # 根据选择的模型类型进行预测
            if model_type == 'linear':
                predictions, confidence_scores = self._predict_with_linear_regression(df, days)
            elif model_type == 'arima':
                predictions, confidence_scores = self._predict_with_arima(df, days)
            elif model_type == 'prophet':
                predictions, confidence_scores = self._predict_with_prophet(df, days, timeframe)
            else:  # ensemble - 默认使用集成方法
                predictions, confidence_scores = self._predict_with_ensemble(df, days, timeframe)
            
            # 生成预测日期
            last_date = pd.to_datetime(df['timestamp'].iloc[-1])
            future_dates = []
            for i in range(1, days + 1):
                if timeframe == '1d':
                    future_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))
                elif timeframe == '1h':
                    future_dates.append((last_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M'))
                else:  # 其他时间周期
                    future_dates.append(f"T+{i}")
            
            # 构建结果
            prediction_data = []
            for i in range(days):
                prediction_data.append({
                    'date': future_dates[i],
                    'predicted_price': float(predictions[i]),
                    'confidence': float(confidence_scores[i]),
                    'upper_bound': float(predictions[i] * (1 + confidence_scores[i] * 0.1)),
                    'lower_bound': float(predictions[i] * (1 - confidence_scores[i] * 0.1))
                })
            
            # 计算预测趋势
            if len(predictions) > 1:
                trend = 'up' if predictions[-1] > predictions[0] else 'down' if predictions[-1] < predictions[0] else 'sideways'
                trend_strength = abs((predictions[-1] - predictions[0]) / predictions[0]) * 100
            else:
                trend = 'unknown'
                trend_strength = 0
            
            return {
                'status': 'success',
                'current_price': float(df['close'].iloc[-1]),
                'predictions': prediction_data,
                'trend': trend,
                'trend_strength': float(trend_strength),
                'model_used': model_type,
                'message': f'基于{self._get_model_name(model_type)}的预测结果，仅供参考，不构成投资建议'
            }
            
        except Exception as e:
            logger.error(f"价格预测出错: {str(e)}")
            return {'status': 'error', 'message': f'预测出错: {str(e)}'}
    
    def _get_model_name(self, model_type):
        """
        获取模型的中文名称
        """
        model_names = {
            'linear': '线性回归',
            'arima': 'ARIMA时间序列',
            'prophet': 'Prophet预测',
            'ensemble': '集成模型'
        }
        return model_names.get(model_type, '未知模型')
    
    def _predict_with_linear_regression(self, df, days):
        """
        使用线性回归进行预测
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        # 准备特征
        df['index'] = np.arange(len(df))
        
        # 添加一些技术指标作为特征
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # 删除NaN值
        df = df.dropna()
        
        # 准备特征和目标变量
        features = ['index', 'ma5', 'ma10', 'ma20', 'price_change', 'volume_change']
        X = df[features].values
        y = df['close'].values
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 评估模型
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        confidence = (train_score + test_score) / 2
        
        # 预测未来价格
        last_index = df['index'].iloc[-1]
        future_indices = np.arange(last_index + 1, last_index + days + 1)
        
        # 为未来数据创建特征
        future_features = []
        for idx in future_indices:
            # 使用最后的MA和变化率作为简单预测
            future_features.append([idx, 
                                   df['ma5'].iloc[-1], 
                                   df['ma10'].iloc[-1], 
                                   df['ma20'].iloc[-1],
                                   df['price_change'].iloc[-1],
                                   df['volume_change'].iloc[-1]])
        
        future_features = np.array(future_features)
        predictions = model.predict(future_features)
        
        # 生成置信度分数
        confidence_scores = [confidence] * days
        
        return predictions, confidence_scores
    
    def _predict_with_arima(self, df, days):
        """
        使用ARIMA模型进行预测
        """
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error
        import math
        
        # 准备时间序列数据
        ts = df['close'].values
        
        # 尝试不同的ARIMA参数
        best_mse = float('inf')
        best_order = (1, 1, 0)  # 默认参数
        
        # 简单的参数搜索
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        # 使用最后30个数据点进行测试
                        train_size = len(ts) - 10
                        train, test = ts[0:train_size], ts[train_size:]
                        
                        # 拟合模型
                        model = ARIMA(train, order=(p, d, q))
                        model_fit = model.fit()
                        
                        # 预测
                        predictions = model_fit.forecast(steps=len(test))
                        
                        # 计算MSE
                        mse = mean_squared_error(test, predictions)
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_order = (p, d, q)
                    except:
                        continue
        
        # 使用最佳参数拟合模型
        model = ARIMA(ts, order=best_order)
        model_fit = model.fit()
        
        # 预测未来价格
        predictions = model_fit.forecast(steps=days)
        
        # 计算置信度 (基于RMSE)
        rmse = math.sqrt(best_mse)
        confidence_base = max(0, 1 - (rmse / df['close'].mean()))
        
        # 置信度随着预测天数的增加而降低
        confidence_scores = [max(0.1, confidence_base * (1 - i * 0.05)) for i in range(days)]
        
        return predictions, confidence_scores
    
    def _predict_with_prophet(self, df, days, timeframe):
        """
        使用Facebook Prophet模型进行预测
        """
        try:
            from prophet import Prophet
        except ImportError:
            # 如果没有安装Prophet，回退到线性回归
            logger.warning("Prophet库未安装，回退到线性回归模型")
            return self._predict_with_linear_regression(df, days)
        
        # 准备Prophet所需的数据格式
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df['timestamp'])
        prophet_df['y'] = df['close']
        
        # 创建和拟合模型
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_df)
        
        # 创建未来数据框
        if timeframe == '1d':
            future = model.make_future_dataframe(periods=days)
        elif timeframe == '1h':
            future = model.make_future_dataframe(periods=days, freq='H')
        else:
            # 对于其他时间周期，使用天为单位
            future = model.make_future_dataframe(periods=days)
        
        # 预测
        forecast = model.predict(future)
        
        # 提取预测结果
        predictions = forecast['yhat'].iloc[-days:].values
        
        # 提取置信区间
        lower_bound = forecast['yhat_lower'].iloc[-days:].values
        upper_bound = forecast['yhat_upper'].iloc[-days:].values
        
        # 计算置信度分数 (基于置信区间的宽度)
        confidence_scores = []
        for i in range(days):
            interval_width = upper_bound[i] - lower_bound[i]
            relative_width = interval_width / predictions[i]
            confidence = max(0.1, 1 - min(1, relative_width / 2))
            confidence_scores.append(confidence)
        
        return predictions, confidence_scores
    
    def _predict_with_ensemble(self, df, days, timeframe):
        """
        使用集成方法进行预测
        """
        # 获取各个模型的预测结果
        linear_pred, linear_conf = self._predict_with_linear_regression(df, days)
        arima_pred, arima_conf = self._predict_with_arima(df, days)
        
        try:
            prophet_pred, prophet_conf = self._predict_with_prophet(df, days, timeframe)
            # 集成三个模型的结果
            predictions = []
            confidence_scores = []
            
            for i in range(days):
                # 加权平均预测值
                weighted_pred = (linear_pred[i] * linear_conf[i] + 
                                arima_pred[i] * arima_conf[i] + 
                                prophet_pred[i] * prophet_conf[i]) / \
                               (linear_conf[i] + arima_conf[i] + prophet_conf[i])
                
                # 平均置信度
                avg_conf = (linear_conf[i] + arima_conf[i] + prophet_conf[i]) / 3
                
                predictions.append(weighted_pred)
                confidence_scores.append(avg_conf)
        except:
            # 如果Prophet预测失败，只集成线性回归和ARIMA
            predictions = []
            confidence_scores = []
            
            for i in range(days):
                # 加权平均预测值
                weighted_pred = (linear_pred[i] * linear_conf[i] + 
                                arima_pred[i] * arima_conf[i]) / \
                               (linear_conf[i] + arima_conf[i])
                
                # 平均置信度
                avg_conf = (linear_conf[i] + arima_conf[i]) / 2
                
                predictions.append(weighted_pred)
                confidence_scores.append(avg_conf)
        
        return predictions, confidence_scores
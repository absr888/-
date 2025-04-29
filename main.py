#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数字货币分析系统主程序
"""

import os
import sys
import logging
from dotenv import load_dotenv

# 导入各个模块
from data_collector.collector import DataCollector
from data_storage.database import Database
from analysis_engine.analyzer import Analyzer
from ui.app import start_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


def main():
    """
    主程序入口
    """
    try:
        logger.info("启动数字货币分析系统...")
        
        # 初始化数据库
        db = Database()
        db.initialize()
        
        # 初始化数据采集器
        collector = DataCollector(db)
        
        # 初始化分析引擎
        analyzer = Analyzer(db)
        
        # 启动数据采集
        collector.start_collection()
        
        # 启动Web界面
        start_app(db, analyzer)
        
    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
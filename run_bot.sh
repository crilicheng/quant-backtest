#!/bin/bash
cd /Users/liujincheng/quant_backtest
source venv/bin/activate
echo "╔══════════════════════════════════════╗"
echo "║  Bitget 量化机器人 - 5币并发/10x杠杆 ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  今天有信号就会自动开仓"
echo "  Ctrl+C 随时停止"
echo ""
python bitget_bot.py --dry-run

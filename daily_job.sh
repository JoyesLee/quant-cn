#!/usr/bin/env bash
# quant-cn 每日自动任务：更新行情 → 虚拟盘调仓 → 生成日报(含 vibe-trading AI 点评)
# 由 cron 在每个交易日 16:20 (A股/港股收盘后) 触发；任务幂等，重复运行会跳过已执行日期
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
VT=/home/ben/.local/share/pipx/venvs/vibe-trading-ai/bin/python
mkdir -p "$DIR/logs"
echo "=== $(date '+%F %T') daily job start ==="
cd "$DIR"
"$VT" main.py daily --ai
rc=$?
echo "=== $(date '+%F %T') daily job end rc=$rc ==="
exit $rc

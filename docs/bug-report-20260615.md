# Bug 复盘：2026-06-15 实盘巨亏事件

## 概况

- **时间**：2026-06-14 22:35 ~ 2026-06-15 09:00
- **交易**：50 笔（3 赢 / 47 输）
- **净亏损**：-$29
- **根因**：两个代码 bug 叠加，导致 TP/SL 挂单失败→紧急平仓→信号反复→循环开平

## 错误分类

| 类型 | 笔数 | 净利 | 说明 |
|------|------|------|------|
| 正常止盈 | 3 | +$43 | RSI 策略正确执行 |
| 正常止损 | 4 | -$20 | 方向判断错误，正常止损 |
| **Bug 秒平** | **43** | **-$52** | TP/SL 挂不上，紧急平仓 |

去掉 bug 的话，7 笔正常交易净利 +$23，胜率 43%，与回测预期一致。

## 根因

### Bug 1：价格精度错误

`update_trailing_stop()` 中 `new_sl` 统一取 6 位小数：

```python
new_sl = round(new_best * (1 + trail_dist), 6)  # 错误
```

但 Bitget 要求按币种精度：
- SOLUSDT 最多 3 位小数 → 6 位被拒（错误码 45115）
- DOGEUSDT 最多 5 位小数

### Bug 2：holdSide 格式错误

TP/SL 挂单时 `holdSide` 传了 `"buy"/"sell"`，但 Bitget API 要求 `"long"/"short"`：

```python
"holdSide":"buy" if hold_side == "long" else "sell"  # 错误
```

### 叠加效果

1. `modify-tpsl-order` 因精度错误失败（45115）
2. `place-tpsl-order` 因 holdSide 格式错误偶尔被拒（45122）
3. TP/SL 挂不上 → 触发 `_emergency_close` 立即平仓
4. 60 秒后信号仍在 → bot 重开仓位
5. 重复 1-4 → 43 单循环送手续费

### 附带 Bug：日亏 20% 停机失效

`start_equity_for_dd` 取了 Bitget API 的 `equity` 字段（恒为 0），导致日亏检查永远不触发。

## 修复

| Bug | 修复 |
|------|------|
| 精度 | `new_sl = round(new_best * (1 + trail_dist), PREC_MAP.get(symbol, 2))` |
| holdSide | `"holdSide": hold_side` — 直接用 API 返回的 `"long"/"short"` |
| 日亏 | 改用 `available` 余额，每天 0 点自动刷新基准 |
| 通知 | 每小时自动检查 bot 状态，停机/报错即时推送 |

## 教训

1. **新增代码必须复用已有常量**（PREC_MAP），不能硬编码
2. **API 参数格式需要对照文档验证**，不能凭记忆猜
3. **风控开关要实际验证**——日亏熔断失效了半个月才被发现
4. **重要改动后应小仓观察**，不应直接大仓

---

> 2026-06-15 记录

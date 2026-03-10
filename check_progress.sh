#!/bin/bash
# 监控后台任务进度

echo "=== DLC项目后台任务监控 ==="
echo "时间: $(date)"
echo ""

echo "1. 训练时间测量 (PID: 3700193)"
if ps -p 3700193 > /dev/null 2>&1; then
    echo "   状态: 运行中"
    echo "   最新日志:"
    tail -5 training_time_measurement.log 2>/dev/null | sed 's/^/   /'
else
    echo "   状态: 已完成或未运行"
    if [ -f results/final/benchmark/dlc_training_time_results.csv ]; then
        echo "   ✓ 结果已生成"
    fi
fi

echo ""
echo "2. 性能复现验证 (PID: 3701322)"
if ps -p 3701322 > /dev/null 2>&1; then
    echo "   状态: 运行中"
    echo "   最新日志:"
    tail -5 reproducibility_check.log 2>/dev/null | sed 's/^/   /'
else
    echo "   状态: 已完成或未运行"
fi

echo ""
echo "查看完整日志:"
echo "  训练时间: tail -f training_time_measurement.log"
echo "  性能验证: tail -f reproducibility_check.log"

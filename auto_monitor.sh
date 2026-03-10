#!/bin/bash
# Auto-monitor background tasks and report when complete

TRAINING_PID=3700193
VERIFY_PID=3701322

echo "开始监控后台任务..."
echo "训练时间测量 PID: $TRAINING_PID"
echo "性能验证 PID: $VERIFY_PID"
echo ""

training_done=false
verify_done=false

while true; do
    if ! $training_done && ! ps -p $TRAINING_PID > /dev/null 2>&1; then
        training_done=true
        echo "[$(date)] ✓ 训练时间测量已完成"
        
        if [ -f results/final/benchmark/dlc_training_time_results.csv ]; then
            echo "结果文件已生成:"
            cat results/final/benchmark/dlc_training_time_results.csv
            echo ""
            if [ -f results/final/benchmark/dlc_training_time_notes.md ]; then
                echo "说明文档:"
                head -20 results/final/benchmark/dlc_training_time_notes.md
            fi
        else
            echo "⚠ 结果文件未找到,检查日志:"
            tail -20 training_time_measurement.log
        fi
        echo ""
    fi
    
    if ! $verify_done && ! ps -p $VERIFY_PID > /dev/null 2>&1; then
        verify_done=true
        echo "[$(date)] ✓ 性能验证已完成"
        echo "验证结果:"
        tail -15 reproducibility_check.log
        echo ""
    fi
    
    if $training_done && $verify_done; then
        echo "=========================================="
        echo "所有后台任务已完成!"
        echo "=========================================="
        break
    fi
    
    sleep 30
done

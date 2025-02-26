import os
os.environ["CUDNN_ALGORITHM_DETERMINISTIC"] = "1"  # 固定随机种子
print("环境随机性锁定时戳：2025-03-01T09:00:00Z")
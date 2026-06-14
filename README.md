# Neural Network and Deep Learning - Project 2

本仓库是课程 `Neural Network and Deep Learning` 的 Project 2 代码实现，主题为：

- CIFAR-10 图像分类
- Batch Normalization 对优化过程的影响分析

说明：

- 本仓库只存放代码、报告源码和必要说明文件。
- 训练结果图片、表格、模型权重、数据集副本不放在 GitHub。
- 模型权重与完整实验输出请放在单独的网盘或 ModelScope 链接中，并在报告中给出公开访问地址。

## 项目结构

```text
codes/VGG_BatchNorm/
  data/                  # 本地数据目录（不上传）
  models/                # 网络结构
  utils/                 # 训练、评估、可视化工具
  run_project2.py        # 主实验入口
  requirements.txt       # Python 依赖

report/
  Project2_Report.tex    # 最终 LaTeX 报告源码

setup_project2_env.sh          # 环境配置脚本
run_project2_experiments.sh    # 实验运行脚本
SERVER_RUN_GUIDE.md            # 服务器运行说明
```

## 已完成实验

代码覆盖了以下方向：

- A 组：结构矩阵（TinyCNN / BasicCNN / VGG / BN / Dropout / Residual / SmallResNet / WideSmallResNet）
- B 组：filters / width ablation
- C 组：loss + regularization（label smoothing / focal / dropout / augmentation / mixup / cutmix）
- D 组：activation ablation
- E 组：optimizer / scheduler / manual implementation
- F 组：Batch Normalization 分析（loss landscape / gradient difference / prediction error）
- G 组：可视化与解释
- H 组：最终高精度搜索

## 环境安装

推荐在 Linux 服务器上：

```bash
bash setup_project2_env.sh
```

或者在 `codes/VGG_BatchNorm` 下手动安装：

```bash
pip install -r requirements.txt
```

## 运行实验

进入主代码目录后运行：

```bash
cd codes/VGG_BatchNorm
python run_project2.py --output-root outputs_search --data-root data --device cuda --resume --skip-completed
```

如果只想快速检查流程，可适当减小样本数或 epoch scale。

## 结果说明

完整实验输出默认保存在：

```text
codes/VGG_BatchNorm/outputs_search/
```

但这些内容已被 `.gitignore` 忽略，不会上传到 GitHub。

## 报告

最终报告源码位于：

```text
report/Project2_Report.tex
```

编译建议使用 `xelatex`。


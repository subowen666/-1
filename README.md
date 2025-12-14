# Iris 数据集机器学习分类可视化项目 README

## 项目概述

本项目基于经典的 **Iris（鸢尾花）数据集**，使用 Python 进行数据探索、特征可视化和机器学习分类模型的决策边界与概率分布可视化。项目展示了从二维到三维的多种可视化技术，帮助直观理解分类模型（如逻辑回归）在不同特征空间下的决策过程。

主要使用库：
- pandas、numpy
- matplotlib、seaborn
- plotly（交互式图表）
- scikit-learn（模型训练）
- skimage（3D 等值面提取）

项目分为数据预览、特征分布分析，以及四个核心可视化任务。

## 数据集介绍

- 来源：scikit-learn 内置 `load_iris()`
- 样本数：150 条
- 特征：4 个（sepal length, sepal width, petal length, petal width，单位：cm）
- 类别：3 类（setosa, versicolor, virginica），每类 50 条
- 本项目中部分任务仅使用前两类进行二分类演示（setosa vs versicolor）

## 运行环境要求

- Python 3.8+
- 必要库：
  ```bash
  pip install pandas seaborn numpy matplotlib plotly scikit-learn scikit-image
  ```

注意：在无图形界面（如服务器）的环境中运行时，`plt.show()` 和 `fig.show()` 不会显示图像，但代码可正常执行并输出文本信息。

## 项目结构与内容说明

### 1. 数据加载与预览
- 加载 Iris 数据集并转换为 DataFrame
- 添加物种名称列
- 打印第 50-100 行数据（versicolor 类样本）

### 2. 特征分布可视化（箱线图）
- 使用 seaborn 绘制 4 个特征在三种物种间的箱线图
- 直观展示各特征的分布差异（尤其是 petal length 和 petal width 的区分度最高）

### 3. 交互式散点图（特征对）
- 使用 plotly.express 生成 6 组二维散点图（所有特征对组合）
- 支持交互缩放、悬停查看详情
- 颜色区分三种物种

### 任务一：二特征三分类决策边界与概率图
- 特征：petal length + petal width（最易区分的两个特征）
- 模型：多分类逻辑回归
- 可视化：
  - 整体决策边界（contourf）
  - 每个类别的预测概率热力图（分别显示）
- 模型在测试集准确率：**1.000**（完美分类）

### 任务二：三特征二分类 3D 决策边界可视化
- 特征：sepal length + sepal width + petal length（前三个特征）
- 类别：仅 setosa 与 versicolor（二分类）
- 模型：逻辑回归
- 使用 `marching_cubes` 算法提取决策面（概率=0.5 的等值面）
- 3D 散点图 + 半透明决策平面
- 模型在测试集准确率：**1.000**

### 任务三：三特征二分类 3D 概率图可视化
- 同上数据与模型
- 在多个固定 petal length 平面上绘制概率 contourf 切片
- 使用颜色条显示属于正类的概率强度

### 任务四：综合可视化（3D 边界 + 概率图）
- 并列展示任务二（决策边界）和任务三（概率切片）的效果
- 更全面地呈现模型在三维空间中的分类行为

## 运行结果示例（文本输出）

运行代码时会输出：
- 数据预览表
- 多分类模型准确率：1.000
- 二分类模型准确率：1.000

在支持图形界面的环境中，将弹出多个 matplotlib 和 plotly 窗口展示精美图表。

## 项目意义

- 适合机器学习入门者理解分类决策边界与概率的概念
- 演示从 2D 到 3D 可视化的进阶技巧
- 展示线性模型（如逻辑回归）在 Iris 数据集上的强大表现


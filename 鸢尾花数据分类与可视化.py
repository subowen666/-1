import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================
# 1. 数据加载与预览
# ====================
print("正在加载 Iris 数据集...")
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("数据集预览（行 50-100）：")
print(df.iloc[50:100])

# 数据预处理
df['species_code'] = df['species_name'].astype('category').cat.codes

# ====================
# 2. 特征分布可视化（箱线图）
# ====================
print("\n正在生成箱线图...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(x='species_name', y='sepal length (cm)', data=df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Sepal Length by Species', fontsize=12, fontweight='bold')

sns.boxplot(x='species_name', y='sepal width (cm)', data=df, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Sepal Width by Species', fontsize=12, fontweight='bold')

sns.boxplot(x='species_name', y='petal length (cm)', data=df, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Petal Length by Species', fontsize=12, fontweight='bold')

sns.boxplot(x='species_name', y='petal width (cm)', data=df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Petal Width by Species', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('01_boxplot_features.png', dpi=150, bbox_inches='tight')
print("✓ 已保存: 01_boxplot_features.png")
plt.show()

# ====================
# 3. 交互式散点图（特征对）
# ====================
print("\n正在生成交互式散点图...")

# 使用静态图代替交互式图（便于保存）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

feature_pairs = [
    ('sepal length (cm)', 'sepal width (cm)'),
    ('sepal length (cm)', 'petal length (cm)'),
    ('sepal length (cm)', 'petal width (cm)'),
    ('sepal width (cm)', 'petal length (cm)'),
    ('sepal width (cm)', 'petal width (cm)'),
    ('petal length (cm)', 'petal width (cm)')
]

for idx, (feat1, feat2) in enumerate(feature_pairs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    for species_name, color in zip(['setosa', 'versicolor', 'virginica'], ['red', 'green', 'blue']):
        subset = df[df['species_name'] == species_name]
        ax.scatter(subset[feat1], subset[feat2], label=species_name, 
                  alpha=0.7, s=50, c=color, edgecolors='k', linewidths=0.5)
    
    ax.set_xlabel(feat1, fontsize=10)
    ax.set_ylabel(feat2, fontsize=10)
    ax.set_title(f'{feat1} vs {feat2}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_scatter_feature_pairs.png', dpi=150, bbox_inches='tight')
print("✓ 已保存: 02_scatter_feature_pairs.png")
plt.show()

# ====================
# 任务一：二特征三分类决策边界与概率图
# ====================
print("\n任务一：二特征三分类可视化")
X = iris.data[:, 2:]  # 使用 petal length 和 petal width
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"模型准确率: {accuracy:.3f}")

# 网格生成
h = 0.02  # 步长
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 预测概率
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
probs = probs.reshape(xx.shape[0], xx.shape[1], 3)

# 绘图
class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
fig, axs = plt.subplots(1, 4, figsize=(22, 5))

# 整体决策边界
axs[0].contourf(xx, yy, Z, alpha=0.4, cmap=mcolors.ListedColormap(class_colors), levels=2)
for i, (species_name, color) in enumerate(zip(['setosa', 'versicolor', 'virginica'], class_colors)):
    mask = y == i
    axs[0].scatter(X[mask, 0], X[mask, 1], c=color, edgecolors='black', 
                   marker='o', s=80, linewidths=1.5, label=species_name, alpha=0.8)
axs[0].set_title('Overall Decision Boundaries', fontsize=13, fontweight='bold')
axs[0].set_xlabel('Petal Length (cm)', fontsize=11)
axs[0].set_ylabel('Petal Width (cm)', fontsize=11)
axs[0].legend(loc='upper left')
axs[0].grid(alpha=0.3)

# 每个类别的概率图
for i in range(3):
    ax = axs[i + 1]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['white', class_colors[i]], N=256)
    contour = ax.contourf(xx, yy, probs[:, :, i], levels=15, alpha=0.8, cmap=cmap)
    
    for j, (species_name, color) in enumerate(zip(['setosa', 'versicolor', 'virginica'], class_colors)):
        mask = y == j
        ax.scatter(X[mask, 0], X[mask, 1], c=color, edgecolors='black', 
                  marker='o', s=60, linewidths=1, alpha=0.7)
    
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Class {iris.target_names[i].capitalize()} Probability', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Petal Length (cm)', fontsize=11)
    ax.set_ylabel('Petal Width (cm)', fontsize=11)
    ax.grid(alpha=0.3)

plt.suptitle(f'Task 1: 2D Decision Boundaries & Probability Maps (Accuracy: {accuracy:.2%})', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('03_task1_2d_decision_boundaries.png', dpi=150, bbox_inches='tight')
print("✓ 已保存: 03_task1_2d_decision_boundaries.png")
plt.show()

# ====================
# 任务二：三特征二分类 3D 边界可视化
# ====================
print("\n任务二：三特征二分类 3D 边界可视化")

# 只使用两类：setosa 和 versicolor
mask_binary = iris.target != 2
X_bin = iris.data[mask_binary, :3]  # 使用前三个特征
y_bin = iris.target[mask_binary]

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42)

model_bin = LogisticRegression(max_iter=200, random_state=42)
model_bin.fit(X_train_bin, y_train_bin)
accuracy_bin = model_bin.score(X_test_bin, y_test_bin)
print(f"二分类模型准确率: {accuracy_bin:.3f}")

# 生成网格
x_min, x_max = X_bin[:, 0].min() - 0.5, X_bin[:, 0].max() + 0.5
y_min, y_max = X_bin[:, 1].min() - 0.5, X_bin[:, 1].max() + 0.5
z_min, z_max = X_bin[:, 2].min() - 0.5, X_bin[:, 2].max() + 0.5

step = 0.1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step),
                         np.arange(z_min, z_max, step))

Z_3d = model_bin.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z_3d = Z_3d.reshape(xx.shape)

# 绘制3D决策边界
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors_bin = ['#FF6B6B', '#4ECDC4']
for i, (species_name, color) in enumerate(zip(['Setosa', 'Versicolor'], colors_bin)):
    mask = y_bin == i
    ax.scatter(X_bin[mask, 0], X_bin[mask, 1], X_bin[mask, 2], 
              c=color, edgecolor='black', s=80, alpha=0.8, 
              linewidths=1.5, label=species_name)

ax.set_xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
ax.set_title(f'Task 2: 3D Decision Boundary (Accuracy: {accuracy_bin:.2%})', 
            fontsize=14, fontweight='bold', pad=20)

# 绘制决策边界等值面
try:
    verts, faces, _, _ = measure.marching_cubes(Z_3d, level=0.5)
    verts = verts * np.array([step, step, step]) + np.array([x_min, y_min, z_min])
    mesh = Poly3DCollection(verts[faces], alpha=0.2, edgecolor='navy', 
                           facecolor='cyan', linewidths=0.3)
    ax.add_collection3d(mesh)
    print("✓ 成功生成3D决策边界等值面")
except Exception as e:
    print(f"⚠ 等值面生成失败: {e}")

ax.legend(loc='upper left', fontsize=11)
ax.view_init(elev=20, azim=45)
ax.grid(True, alpha=0.3)

plt.savefig('04_task2_3d_decision_boundary.png', dpi=150, bbox_inches='tight')
print("✓ 已保存: 04_task2_3d_decision_boundary.png")
plt.show()

# ====================
# 任务三：三特征二分类 3D 概率图可视化
# ====================
print("\n任务三：三特征二分类 3D 概率图可视化")

probs_3d = model_bin.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
probs_3d = probs_3d.reshape(xx.shape)

fig = plt.figure(figsize=(14, 11))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点
for i, (species_name, color) in enumerate(zip(['Setosa', 'Versicolor'], colors_bin)):
    mask = y_bin == i
    ax.scatter(X_bin[mask, 0], X_bin[mask, 1], X_bin[mask, 2], 
              c=color, edgecolor='black', s=80, alpha=0.9, 
              linewidths=1.5, label=species_name)

ax.set_xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Petal Length (cm)', fontsize=12, fontweight='bold')
ax.set_title('Task 3: 3D Probability Map (Binary Classification)', 
            fontsize=14, fontweight='bold', pad=20)

# 绘制概率等值面（多个层级）
num_slices = 6
z_levels = np.linspace(z_min, z_max, num_slices)

for z_level in z_levels:
    z_idx = np.argmin(np.abs(zz[0, 0, :] - z_level))
    ax.contourf(xx[:, :, z_idx], yy[:, :, z_idx
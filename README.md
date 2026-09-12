# 机器学习与强化学习研习资料库 (Machine Learning & RL Data)

本仓库用于系统整理与同步本人的机器学习、深度学习（D2L）及强化学习（Easy RL）的学习教材、理论精解笔记与实战代码。

---

---

## 📖 当前研习进度备忘
* **当前阅读进度**：已精读至《Easy RL：强化学习教程》（蘑菇书）**第 58 页**
* **当前知识板块**：第 3 章【表格型方法 (Tabular Methods)】
  * 核心涉及：时序差分学习 (TD)、$ 步时序差分、悬崖寻路 (Cliff Walking) 问题及免模型控制 (GPI)
* **下一阶段目标**：经典算法 **Sarsa** 与 **Q-Learning** 的核心机制与代码实战

## 📂 仓库目录结构

`	ext
资料/
├── 深度学习_D2L/
│   ├── 教材/
│   │   └── d2l-zh-pytorch.pdf           # 《动手学深度学习》电子教材 (Git LFS)
│   └── 代码/
│       ├── 01_张量操作基础.ipynb        # 张量初始化、切片、运算符与广播机制
│       ├── 02_数据预处理.ipynb          # Pandas 缺失值处理与张量格式转换
│       └── data/
│           └── house_tiny.csv           # 预处理章节配套数据集
│
├── 强化学习_EasyRL/
│   ├── 教材/
│   │   └── EasyRL_v1.0.6.pdf            # 《Easy RL 蘑菇书》电子教材 (Git LFS)
│   ├── 笔记/
│   │   ├── EasyRL_第一章_课后习题精解.md # 第一章 1-1 至 1-10 思考题精解
│   │   ├── 强化学习核心概念_价值函数与折扣因子.md # V/Q函数深度拆解及 AlphaGo 架构勘误
│   │   └── EasyRL_第二章_马尔可夫决策过程核心笔记.md # 马尔可夫性、三级跳模型与贝尔曼方程
│   └── 代码/
│       └── 01_Gym环境交互测试.ipynb     # Gymnasium / Gym 交互环境实战 (CartPole, MountainCar)
│
├── .gitattributes                       # Git LFS 大文件托管配置
├── .gitignore                           # 忽略缓存与中间文件
└── README.md                            # 仓库导航索引
`

---

## 🛠️ 环境依赖要求
* **Python**: 3.14+
* **深度学习框架**: 	orch (PyTorch)
* **强化学习环境**: gymnasium, gym, pygame-ce
* **数据与绘图**: 
umpy, pandas, matplotlib
* **交互式编辑器**: Jupyter Notebook\n

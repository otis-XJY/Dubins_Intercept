# ICRA 2026 Paper - LaTeX Project

## 论文信息
- **标题**: Dynamic Intercept Point Selection for Multi-UAV Interception in Large-Scale Obstacle Environments via Transformer-based MARL
- **格式**: ICRA 2026 (8 pages, double-column, IEEE format)
- **编译**: pdfLaTeX + BibTeX

## 文件结构
```
icra2026_paper/
├── main.tex          # 主论文文件
├── references.bib    # 参考文献
├── Makefile          # 编译脚本
├── figures/          # 图片目录（待添加）
└── README.md         # 本文件
```

## 编译方法

### 方法 1: Make (推荐)
```bash
cd docs/icra2026_paper
make
```

### 方法 2: 手动编译
```bash
cd docs/icra2026_paper
xelatex main
bibtex main
xelatex main
xelatex main
```

### 方法 3: Overleaf
1. 将 `main.tex`、`references.bib`、`IEEEtran.cls` 和 `IEEEtran.bst` 上传到 Overleaf
2. 设置编译器为 XeLaTeX
3. 编译即可

### 方法 4: VS Code LaTeX Workshop
1. 安装 LaTeX Workshop 扩展
2. 打开 `main.tex`
3. 按 Ctrl+Alt+B 编译（需配置 XeLaTeX）

### 方法 5: TeXstudio / TeXmaker
1. 打开 `main.tex`
2. 选项 → 设置 → 构建 → 默认编译器 → XeLaTeX
3. F5 编译

## 需要完成的工作

### 红色标记 (MARK-红色)
论文中所有 `\marktodo{}` 命令标记了待确认的内容：
- 地图参数 (X)
- 命名确认
- 实验结果
- 参考文献验证
- 图片插入
- 未来工作确认

### 参考文献
`references.bib` 中部分引用标记了 `\marktodo{}`，需要：
1. 补充完整的作者、标题、期刊信息
2. 验证 DOI/URL 的准确性
3. 确保无虚假引用

### 图片
需要在 `figures/` 目录中添加：
1. `framework.pdf` - 系统框架图
2. `network.pdf` - CQN 网络架构图
3. `scalability.pdf` - 可扩展性折线图
4. `trajectory.pdf` - 轨迹可视化
5. `replanning.pdf` - DWA 重规划过程图

## ICRA 2026 格式要求
- **页数**: 8 页（含参考文献）
- **格式**: 双栏 IEEE
- **字体**: 10pt
- **审稿**: 双盲（提交时需移除作者信息）
- **模板**: https://ras.papercept.net/conferences/support/support.php

## 提交前检查
- [ ] 移除所有 `\marktodo{}` 命令
- [ ] 移除作者信息（双盲审稿）
- [ ] 验证所有参考文献
- [ ] 插入所有图片
- [ ] 检查页数不超过 8 页
- [ ] 生成 PDF 并检查格式

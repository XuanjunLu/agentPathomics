# Pathomics Viewer（本项目）使用说明（Anaconda Prompt / CMD）

本说明文档面向 **Windows + Anaconda Prompt（CMD）**，不包含 PowerShell 命令。

---

## 1. 项目目录结构（默认约定）

- 项目目录：`Q:\PyPathomics_agent\seg_result\path_viewer_backup\`
- 默认切片目录：`path_viewer_backup\slides\`
  - 把 `.svs/.tif/.tiff/.ndpi` 放这里
- 默认结果目录：`Q:\PyPathomics_agent\seg_result\result\`
  - 例如：
    - `result\nuclei_seg\*_class.png`
    - `result\nuclei_seg_dzi\*.dzi` + `*_files\...`（可选，用于快速 nuclei overlay）
    - `result\tissue_mask\*.png`
    - `result\tissue_seg\full_images\*.png`

如果你想自定义路径，见下方“环境变量”。

---

## 2. 创建并进入 Conda 环境

打开 **Anaconda Prompt**，执行：

```bat
cd /d Q:\PyPathomics_agent\seg_result\path_viewer_backup
conda create -n path_viewer python=3.10 -y
conda activate path_viewer
```

---

## 3. 安装依赖（推荐方式）

### 3.1 安装 OpenSlide 运行库（解决 openslide DLL）

```bat
conda install -c conda-forge openslide -y
```

### 3.2 安装 Python 依赖

项目依赖在 `requirements.txt`：

```bat
pip install -r requirements.txt
```

如遇网络问题，可配置 pip 镜像后再装（按你单位/学校的镜像策略即可）。

---

## 4. 环境变量（可选，但推荐了解）

这些变量都可以 **只在当前 Anaconda Prompt 窗口生效**（用 `set`），不污染系统全局。

### 4.1 结果与切片路径

- **WSI_SLIDE_DIR**：切片目录（默认 `path_viewer_backup\slides`）
- **WSI_RESULT_DIR**：结果目录（默认 `path_viewer_backup` 的上一级的 `result/`）

示例（按你的实际路径改）：

```bat
set WSI_SLIDE_DIR=Q:\PyPathomics_agent\seg_result\path_viewer_backup\slides
set WSI_RESULT_DIR=Q:\PyPathomics_agent\seg_result\result
```

### 4.2 tissue_mask 叠加层颜色（后端染色）

后端会把 `tissue_mask` 的非 0 区域染成一个固定 RGBA 颜色。你可以用：

- **WSI_TISSUE_MASK_RGBA**：格式 `R,G,B,A`（0~255）

例如：绿色半透明（默认就是这个）：

```bat
set WSI_TISSUE_MASK_RGBA=0,255,0,220
```

> 提示：后端对 tissue_mask 预览图做了“颜色版本号”的缓存；改这个变量后重启服务即可生效。

### 4.3（可选）libvips 路径（用于生成 nuclei_seg 的 DZI）

如果你要把 `*_class.png` 转成 DeepZoom（DZI），需要 libvips：

- **VIPS_BIN**：`...\vips\bin`（包含 DLL）
- **VIPS_EXE**：`...\vips\bin\vips.exe`

示例（按你的实际解压路径改）：

```bat
set VIPS_BIN=Q:\PyPathomics_agent\seg_result\vips-dev-8.17\bin
set VIPS_EXE=Q:\PyPathomics_agent\seg_result\vips-dev-8.17\bin\vips.exe
set PATH=%VIPS_BIN%;%PATH%
```

---

## 5. 启动服务

在 Anaconda Prompt（已 `conda activate path_viewer`）里：

```bat
cd /d Q:\PyPathomics_agent\seg_result\path_viewer_backup
python app.py
```

当前代码默认监听：

- **地址**：`0.0.0.0`
- **端口**：`8366`

浏览器打开：

- 本机访问：`http://127.0.0.1:8366/`
- 局域网访问：`http://<你的局域网IP>:8366/`

查看局域网 IP：

```bat
ipconfig
```

### 5.1（可选）Windows 防火墙放行端口（CMD 方式）

如果你要让同网段其他机器访问，可能需要放行端口：

```bat
netsh advfirewall firewall add rule name="Pathomics Viewer 8366" dir=in action=allow protocol=TCP localport=8366
```

---

## 6. 使用方式（页面）

右侧“结果展示”中目前包含：

- **细胞核语义分割（nuclei_seg）**
  - 若存在 `result\nuclei_seg_dzi\...` 会优先用 DZI（更快）
  - 否则回退用后端生成的下采样预览 PNG
- **组织区域mask（tissue_mask）**
  - 从 `result\tissue_mask\` 读取并在后端下采样+着色
- **Tumor & Microenvironment（肿瘤及微环境）**
  - 直接读取 `result\tissue_seg\full_images\` 的 RGB 可视化图

每个叠加层都有开关与透明度滑条。

---

## 7.（可选）生成 nuclei_seg 的 DZI（推荐，用于快速加载）

目的：把巨大的 `*_class.png` 转成 `.dzi + _files/`，实现类似 SVS 的分块金字塔加载。

### 7.1 准备：确认 vips 可用

```bat
vips --version
```

如果提示找不到命令，先按上面“4.3 libvips 路径”设置 `VIPS_BIN/VIPS_EXE/PATH`。

### 7.2 一键自动从 `..\svs` 选取示例生成（常用）

在 `path_viewer_backup\` 下执行：

```bat
python -u tools\make_mask_dzi.py --auto --svs-dir "..\svs" --output-dir "..\result\nuclei_seg_dzi" --depth onepixel
```

如果你的 `*_class.png` 是 0..5 的标签图，并且在网页里“看起来发黑”，加 `--colorize`：

```bat
python -u tools\make_mask_dzi.py --auto --svs-dir "..\svs" --output-dir "..\result\nuclei_seg_dzi" --depth onepixel --colorize
```

> `--colorize` 需要 `pyvips`，并且需要 DLL 可见（设置 `VIPS_BIN` 并加到 `PATH` 通常就行）。

生成后会得到：

- `..\result\nuclei_seg_dzi\<name>.dzi`
- `..\result\nuclei_seg_dzi\<name>_files\...`

重启 `python app.py`，页面会自动优先走 DZI 叠加（加载更快）。

---

## 8. 常见问题（FAQ）

### 8.1 打开页面后看不到某个按钮/提示“No xxx found”

说明当前 slide 在结果目录里找不到对应文件：

- nuclei 语义分割：`result\nuclei_seg\<stem>_class.png` 或 `result\nuclei_seg_dzi\<stem>_class.dzi`
- tissue_mask：`result\tissue_mask\<stem>.png`
- full_images：`result\tissue_seg\full_images\<stem>.png`

其中 `<stem>` 是切片文件名去掉扩展名（如 `xxx.svs` → `xxx`）。

### 8.2 修改了叠加层颜色但浏览器还是旧颜色

按顺序做：

- 重启 `python app.py`
- 浏览器 `Ctrl+F5` 硬刷新

（后端已经对 tissue_mask 做了颜色版本化缓存，正常不会卡死在旧颜色上）


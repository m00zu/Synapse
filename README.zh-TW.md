<p align="center">
  <img src="synapse/icons/synapse_icon.png" alt="Synapse" width="128">
</p>

<h1 align="center">Synapse</h1>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-TW.md">繁體中文</a>
</p>

<p align="center">
  用於科學數據分析的工作流程編輯器
</p>

<p align="center">
  <a href="https://polyformproject.org/licenses/noncommercial/1.0.0"><img src="https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg" alt="Platform">
</p>

---

在畫布上連接各個分析節點來建立完整的分析流程。不用寫程式、不用切換軟體、不用在步驟之間轉換檔案格式。

## 功能

- **視覺化流程建構**：在畫布上連接節點，建立分析工作流程
- **可重現、可分享**：工作流程儲存為 `.json` 檔案，任何人都能開啟並重現執行結果
- **嚴格的埠類型檢查**：連線時強制檢查資料型別相容性；`MaskData` 輸出可向上轉型 (upcast) 連到 `ImageData` 輸入，但反方向不行；錯誤會即時顯示在狀態列
- **自動排版**：`Ctrl+L` 由左至右重新排列所有節點並避免重疊
- **執行所選節點 (含上游)**：`Ctrl+Shift+W` 執行選取的節點及其上游相依節點，並略過下游節點 — 適合在不觸發儲存/報告節點的情況下除錯
- **批次處理**：對整個資料夾進行迭代，並自動累積結果
- **外掛系統**：透過 `.py`、`.zip` 或 `.synpkg` 套件擴充客製化節點
- **兩種 AI 使用方式**：
  - **應用程式內 AI 對話面板** — 支援 8 種供應商，包含本地 **Ollama** 與 **llama.cpp** (無需 API 金鑰、無需網路)；雲端模型則為使用者自付費
  - **MCP 伺服器** — 從外部對話客戶端 (**Claude Code**、**Claude Desktop**、**Antigravity**、**Gemini CLI**) 透過你既有的訂閱方案來操作 Synapse；提供每個客戶端的一鍵設定
- **跨平台**：macOS、Windows、Linux

## 下載

獨立執行檔 (不需安裝 Python)：

| 作業系統 | 下載 |
|------|------|
| macOS (Apple Silicon) | [Synapse-macOS-arm64.dmg](https://github.com/m00zu/Synapse/releases/latest/download/Synapse-macOS-arm64.dmg) |
| Windows (64-bit) | [Synapse.exe](https://github.com/m00zu/Synapse/releases/latest/download/Synapse.exe) |

所有版本請參見 [Releases 頁面](https://github.com/m00zu/Synapse/releases)。

> **macOS 首次開啟**：macOS 可能會因為應用程式未簽署而封鎖。請在應用程式上按右鍵 → **打開** → 在對話框中點選 **打開**。或在終端機執行：
> ```bash
> xattr -cr /Applications/Synapse.app
> ```
> 此操作只需執行一次。

> **Windows 首次開啟**：Windows SmartScreen 可能會顯示警告。點選 **更多資訊** → **仍要執行**。此操作只需執行一次。

## 從原始碼安裝

測試環境：Python 3.13 與 3.14。

```bash
git clone https://github.com/m00zu/Synapse
cd Synapse
pip install .
```

建議安裝：預編譯的 Rust 擴充套件，可加速 OIR 檔案讀取與部分影像處理：

```bash
pip install oir_reader_rs image_process_rs --find-links https://github.com/m00zu/Synapse/releases/expanded_assets/rust-v0.1.1
```

執行：

```bash
synapse
```

## 範例工作流程

### CSV 分析

`Table Reader` > `Filter Table` > `Single Table Math` > `Aggregate Table` > `Data Table Node`

載入細胞測量的 CSV 檔案，過濾小物件 (`area > 100`)，計算圓度 (`4 * pi * area / perimeter^2`)，最後按分組計算 Control 與 Treatment 的平均值並顯示摘要。

<p align="center">
  <img src="docs/images/Example_1.png" alt="CSV 分析流程" width="800">
</p>

### 物件偵測與測量

`Image Reader` > `Gaussian Blur` > `Binary Threshold` > `Fill Holes` > `Watershed` > `Data Table Node`

載入硬幣影像，模糊處理以降低雜訊，二值化 (binarize)，填充孔洞，再用分水嶺演算法 (watershed) 分離相鄰物件，最終輸出每個偵測物件的面積、周長與圓度。

<p align="center">
  <img src="docs/images/Example_2.png" alt="影像物件偵測" width="800">
</p>

### 統計比較

`Table Reader` > `Filter Table` > `Pairwise Comparison` > `Bar Plot` > `Data Figure Node`

載入細胞測量數據，過濾碎屑，對 Control 與 Treatment 的 `intensity_mean` 進行比較，最終繪製帶顯著性標註的結果圖。

<p align="center">
  <img src="docs/images/Example_3.png" alt="統計比較" width="800">
</p>

### 批次 OIR 轉檔

```
Folder Iterator --> Image Reader  --> Data Saver
       └---------> Path Modifier -----↗
```

批次將 Olympus OIR 顯微鏡檔案轉換為 TIFF。Iterator 將每個 `.oir` 路徑同時傳給讀取器（解碼影像）和路徑修改器（將副檔名改為 `.tif` 並導向輸出資料夾），兩者都連接到儲存器。

<p align="center">
  <img src="docs/images/Example_4.png" alt="批次 OIR 轉檔" width="800">
</p>

### 批次多通道匯出（使用 Collection）

```
Folder Iterator --> OIR Reader --> Collect --> Scale Bar --> Split Collection --> Save Collection
       └---------> Path Modifier -----------------------------------------------↗
```

批次處理多個 channel 的 OIR 檔案。OIR Reader 將每個檔案拆分為個別 channel（ch1–ch4）及一個合成影像。Collect 將所有輸出打包成單一 Collection。Scale Bar 自動套用相同比例尺到每個通道。Split Collection 將合成影像與 ch1 分離出來，儲存到由 Path Modifier 指定的輸出資料夾與副檔名。

<p align="center">
  <img src="docs/images/Example_5.png" alt="批次多通道匯出" width="800">
</p>

### 膠原蛋白面積測量 (影片)

https://github.com/user-attachments/assets/a3772ee9-da64-4fe1-ad58-ee22ac6f41aa

<p align="center"><i>馬森三色染色 (Masson's trichrome stain) 的顏色反捲積 (color deconvolution)，接著對膠原蛋白 channel 進行二值化並測量面積。</i></p>

## 使用 AI

Synapse 提供**兩種 AI 使用方式** — 兩者都會操作相同的節點圖。請依你的成本考量與工作流程選擇。

### 應用程式內 AI 對話 (可搭配 Ollama 離線使用)

1. **View > AI Chat** 開啟對話面板。
2. 選擇供應商。最簡單且無需 API 金鑰的方式是安裝 [Ollama](https://ollama.com) 並下載模型 (`ollama pull gemma3:12b`)。
3. 輸入描述 — AI 助理會為你建構並編輯畫布。

支援 Claude、OpenAI、Gemini、Groq、OpenRouter、Ollama、llama.cpp 與 RunPod。API 金鑰透過作業系統的 keyring 儲存於本地；對話記錄僅保留於目前的工作階段。

### 從既有的對話客戶端透過 MCP 操作

Synapse 會在 `127.0.0.1:51780` 執行一個 MCP 伺服器，讓外部對話客戶端可以讀取、修改與執行你的圖形，並使用你既有的對話訂閱方案 (Synapse 不會額外計費)。

1. **Help > AI Connection (MCP)...** 開啟連線對話框。
2. 點選你所使用客戶端 (**Claude Code**、**Claude Desktop**、**Antigravity** 或 **Gemini CLI**) 對應的設定按鈕 — 系統會自動寫入正確的設定檔。
3. 開啟對話客戶端，並請它檢視、建構或修改工作流程。

完整功能參考請參閱 [AI 概覽](https://m00zu.github.io/Synapse/ai/)。

## 外掛

核心功能只包含資料 I/O 與顯示結果，而特定領域的節點以外掛形式發布。

### 安裝外掛

**從應用程式內的外掛管理器安裝（建議）：**

1. 在 Synapse 中，前往 **Plugins > Plugin Manager** 並開啟 **Browse Online** 分頁
2. 瀏覽可用的外掛，點選 **Install** 即可安裝
3. 外掛會自動下載並安裝，點選 **Plugins > Reload Plugins** 來載入新的節點。

**手動安裝：**

1. 從 [Synapse-Plugins Releases](https://github.com/m00zu/Synapse-Plugins/releases) 下載 `.synpkg` 檔案
2. 在 Synapse 中，前往 **Plugins > Install Plugin** 並選擇 `.synpkg` 檔案
3. 點選 **Plugins > Reload Plugins**，新節點會出現在 Node Explorer 中

也可以直接將 `.py` 檔案或解壓縮的外掛資料夾放入 `plugins/` 目錄。

### 可用外掛

| 外掛 | 說明 |
|------|------|
| Data Processing | 表格篩選、排序、數學欄位、彙總、合併、連接（預設安裝）|
| Image Analysis | 濾波、二值化、形態學、分割、測量、ROI |
| Statistical Analysis | t 檢定、ANOVA、迴歸、存活分析、PCA |
| Figure Plotting | 散佈圖、箱型圖、小提琴圖、熱圖、火山圖、迴歸圖、SVG 編輯器 |
| Machine Learning | scikit-learn 分類器、迴歸、分群、嵌入 (UMAP)、SHAP、訓練/測試集切分 |
| SAM2 & Cellpose | 點擊分割 (SAM2)、細胞/細胞核分割 (Cellpose)、影片追蹤 |
| Cheminformatics | RDKit 分子編輯、指紋、骨架、批次對接 (AutoDock Vina / GNINA)、蛋白質準備 |
| 3D Volume | Z-stack I/O、3D 形態學、體積渲染 |
| Filopodia | 細胞突起偵測與測量 (同 FiloQuant) |
| Report | 由工作流程輸出生成 Markdown / HTML 報告 |

## 使用說明

線上使用說明：[m00zu.github.io/Synapse](https://m00zu.github.io/Synapse/)，也可在應用程式內透過 **Help > Open Manual** 開啟。

## 授權

本程式採用 [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) 授權。你可以在任何非商業用途下使用、修改與散布 Synapse，包含個人專案、學術研究，以及非營利或政府組織內部使用。商業用途需另向著作權人取得授權。

"""
translations/zh_TW.py
=====================
Traditional Chinese (繁體中文) UI strings.

Keys   = the original English string used in source code.
Values = the Traditional Chinese equivalent.
"""

STRINGS: dict[str, str] = {

    # ── Node Explorer category labels ────────────────────────────────────────
    "Table Processing":     "表格處理",
    "Statistical Analysis": "統計分析",
    "Visualization":        "視覺化",
    "Plugins":              "外掛程式",
    "Confocal Analysis":    "共焦分析",
    "Input / Output":       "輸入 / 輸出",
    "Common Utilities":     "常用工具",
    "Collection":           "集合",
    "Display":              "顯示",
    "Image Processing":     "影像處理",
    "Color":                "色彩",
    "Adjust/Contrast":      "調整/對比",
    "Filters":              "濾鏡",
    "Thresholding":         "閾值化",
    "Morphology":           "形態學",
    "Math":                 "數學",
    "Filopodia Analysis":   "絲足分析",
    "Geometry":             "幾何",
    "Measure":              "測量",
    "Misc":                 "其他",
    "Segmentation":         "分割",
    "Video Analysis":       "影片分析",

    # ── Dock / toolbar titles ────────────────────────────────────────────────
    "Node Explorer":        "節點總管",
    "AI Assistant":         "AI 助理",
    "Node Help":            "節點說明",
    "Execution":            "執行",

    # ── Menu titles ──────────────────────────────────────────────────────────
    "&Edit":                "&編輯",
    "&Workflows":           "&工作流程",
    "&View":                "&檢視",
    "Open Recent":          "開啟最近",
    "&Examples":            "&範例",
    "Append Example":       "附加範例",
    "&Help":                "&說明",
    "Language":             "語言",

    # ── Menu / toolbar actions ───────────────────────────────────────────────
    "&Copy Nodes":              "&複製節點",
    "&Paste Nodes":             "&貼上節點",
    "&Save Workflow...":        "&儲存工作流程…",
    "&Open Workflow...":        "&開啟工作流程…",
    "Append Workflow...":       "附加工作流程…",
    "Autosave Now":             "立即自動儲存",
    "Reopen Last Workflow":     "重新開啟上次工作流程",
    "Install Plugin...":        "安裝外掛程式…",
    "Reload Plugins":           "重新載入外掛程式",
    "Plugin Manager...":        "外掛程式管理員…",
    "Pipe Style...":            "連線樣式…",
    "Select &All Nodes":        "選取全部節點(&A)",
    "&Focus Node Search":       "聚焦節點搜尋(&F)",
    "Minimap":                  "小地圖",
    "Open Manual":              "開啟手冊",
    "Node Help Panel":          "節點說明面板",
    "Run Graph":                "執行圖",
    "Batch Run":                "批次執行",
    "Stop":                     "停止",
    "Clear Selected Caches":    "清除所選的快取",
    "Clear All Caches":         "清除所有快取",
    "Light Mode":               "淺色模式",
    "Dark Mode":                "深色模式",
    "No recent workflows":      "無最近工作流程",
    "Clear Recent":             "清除最近記錄",

    # ── Common dialog buttons ────────────────────────────────────────────────
    "Save":     "儲存",
    "Discard":  "捨棄",
    "Cancel":   "取消",

    # ── Status bar (static messages only) ───────────────────────────────────
    "Ready":                            "就緒",
    "Executing graph...":               "正在執行圖形…",
    "Stopping execution...":            "正在停止執行…",
    "Execution failed.":                "執行失敗。",
    "No nodes selected to clear.":      "未選取任何節點以清除。",
    "All node caches cleared.":         "所有節點快取已清除。",
    "Autosave snapshot created.":       "自動儲存快照已建立。",
    "Autosave skipped (graph is empty).": "自動儲存已跳過（圖形為空）。",
    "Starting batch execution...":      "正在開始批次執行…",
    "Error: circular connection detected. Remove the cycle and try again.":
        "錯誤：偵測到循環連線。請移除迴圈後再試一次。",
    "Stop requested — waiting for current node to finish.":
        "已請求停止 — 等待目前節點完成。",
    "Batch stopped by user.":           "使用者已停止批次執行。",
    "Execution stopped by user.":       "使用者已停止執行。",
    "Script exported to":               "腳本已匯出至",
    "No examples found":                "找不到範例",

    # ── Dialog titles ────────────────────────────────────────────────────────
    "Reopen Last":              "重新開啟上次",
    "Recover Autosave":         "還原自動儲存",
    "Recovery Failed":          "還原失敗",
    "Unsaved Changes":          "未儲存的變更",
    "Execution Error":          "執行錯誤",
    "Save Workflow":            "儲存工作流程",
    "Save Error":               "儲存錯誤",
    "Open Workflow":            "開啟工作流程",
    "Append Workflow":          "附加工作流程",
    "Load Error":               "載入錯誤",
    "Append Error":             "附加錯誤",
    "Batch Completed with Errors": "批次執行完成（含錯誤）",
    "Install Plugin":           "安裝外掛程式",
    "Overwrite Plugin?":        "覆寫外掛程式？",
    "Install Failed":           "安裝失敗",
    "Manual Not Found":         "找不到手冊",
    "Export as Python Script":  "匯出為 Python 腳本",
    "Export Error":             "匯出錯誤",

    # ── Dialog body text (static) ────────────────────────────────────────────
    "No previous workflow found.":  "找不到先前的工作流程。",
    "opening another workflow":     "開啟另一個工作流程",
    "quitting":                     "結束程式",
    "The manual has not been built yet.\n\nRun 'mkdocs build' in the project directory to generate it.":
        "手冊尚未建置。\n\n請在專案目錄中執行 'mkdocs build' 來產生手冊。",
    "A 'Folder Iterator' node is in the graph.\n\nDid you mean to use Batch Run (Ctrl+B) to process all files?\n\nClick 'Yes' to switch to Batch Run, or 'No' to run once.":
        "圖形中有「資料夾迭代器」節點。\n\n您是否要使用批次執行 (Ctrl+B) 來處理所有檔案？\n\n點選「是」切換至批次執行，或點選「否」僅執行一次。",
    "No iterator node found in the graph (Folder Iterator, Video Iterator, etc.).":
        "圖形中找不到迭代器節點（資料夾迭代器、影片迭代器等）。",
    "Select a node to see its documentation.":
        "選取節點以檢視其說明文件。",

    # ── File dialog filters ──────────────────────────────────────────────────
    "JSON Files (*.json)":   "JSON 檔案 (*.json)",
    "Python Files (*.py)":   "Python 檔案 (*.py)",
    "Plugin Files (*.py *.zip *.synpkg)": "外掛程式檔案 (*.py *.zip *.synpkg)",

    # ── nodes/base.py widget labels ─────────────────────────────────────────
    "Select Dot Color":     "選取點的顏色",
    "Select file...":       "選取檔案…",
    "Select File":          "選取檔案",
    "Select directory...":  "選取目錄…",
    "Select Directory":     "選取目錄",
    "Progress":             "進度",
    "No Preview":           "無預覽",

    # ── Language menu items ──────────────────────────────────────────────────
    "English":              "English",
    "Traditional Chinese":  "繁體中文",
    "Restart required to apply language change.":
        "需要重新啟動才能套用語言變更。",
}

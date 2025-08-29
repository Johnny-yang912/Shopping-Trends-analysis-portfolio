# Shopping-Trends-analysis-portfolio
# 🛒 購物行為分析（Shopping Behavior Analysis）

## 📖 專案簡介
本專案以 **零售商視角** 出發，探討影響顧客行為的三大核心指標：  
1️⃣ **顧客評分**（Customer Rating）  
2️⃣ **會員加入意向**（Membership Subscription）  
3️⃣ **購物頻率**（Shopping Frequency）  

透過 **探索性資料分析（EDA）**、**統計檢定** 與 **機器學習建模**，挖掘顧客行為規律，進而提出 **行銷策略與營收優化建議**。

---
## 📊 報告展示（PDF）  
👉 [查看完整 PDF 報告](購物行為分析.pdf)  
👉 [查看 Jupyter Notebook](購物行為分析.ipynb)  

## 🧰本專案同時提供：
- **Notebook**：完整的分析過程與可視化
- **工具包 (`utils_shopping.py`)**：模組化工具，方便團隊快速重現與擴展分析流程
---

## 🔍 分析方法
- **資料來源**：Kaggle - Shopping Trends and Customer Behaviour Dataset  
- **分析流程**：  
  📊 資料探索 →  
  📑 假設檢定（ANOVA、卡方檢定） →  
  🤖 建模與參數調整（RandomizedSearchCV + GridSearchCV） →  
  📈 特徵重要性分析 →  
  🎯 策略制定與驗證  

---

## 📊 主要發現
- ⭐ **顧客評分**：受單一特徵影響有限，目前資料無法建立具商業價值的預測模型。  
- ⭐ **會員加入**：  
  - 折扣使用（>0.5）與性別（>0.1）是兩大關鍵因素。  
  - 隨機森林模型在 **Recall=1.0** 表現最佳，可有效捕捉潛在會員。  
- ⭐ **購物頻率**：多數特徵與其關聯性低，模型解釋力不足，顯示購物頻率可能由更複雜的行為動機驅動。  

---

## 🧾 策略建議
1. 🎯 **重新定義關鍵指標**：將顧客評分拆解為更細的滿意度維度（商品、服務、價格）。  
2. 🎁 **折扣與會員誘因設計**：設計入會即享優惠、節慶專屬折扣，搭配精準行銷。  
3. 👩 **女性客群調查**：補充女性消費習慣與痛點，改善入會障礙。  
4. 🧪 **模型應用與驗證**：透過 **A/B 測試** 驗證模型預測效果，持續追蹤轉化率。  

---

## 📂 檔案內容
| 檔名 | 說明 |
|------|------|
| `購物行為分析.ipynb` | 💻 Jupyter Notebook，完整程式碼與流程 |
| `購物行為分析.pdf` | 📑 分析報告簡報版，含圖表與結論 |
| `README.md` | 📘 專案說明文件 |
| `LICENSE` | 📜 授權條款 |
| `.gitignore` | ⚙️ Git 忽略設定 |

---

## ⚙️ 技術工具
- 🐍 **語言與套件**：Python (pandas, numpy, matplotlib, scipy, scikit-learn, xgboost)  
- 📐 **統計方法**：單因子 ANOVA、卡方檢定、Tukey HSD  
- 🤖 **機器學習**：線性迴歸、邏輯迴歸、隨機森林、XGBoost  
- 🔧 **調參方式**：RandomizedSearchCV、GridSearchCV、Cross Validation  

---

## 📌 限制與展望
- 🔎 本資料為 **模擬數據集**，部分結果僅能反映統計關聯，難以推論因果關係。  
- 📉 資料缺乏收入、生活型態、忠誠度等關鍵特徵，導致模型解釋力有限。  
- 🚀 未來可結合 **長期追蹤數據** 與 **顧客調查**，提升模型準確性，並驗證策略成效。  

---
## 🧰 工具包（utils_shopping.py）

- 自動前處理：One-Hot 類別欄、數值欄可選標準化
- 任務模式：classification（logreg / rf）、regression（linreg / rf）
- `tune=True`：RandomizedSearchCV 自動調參並印出最佳參數

### 安裝需求
```bash
pip install -r requirements.txt
```

### 使用範例（分類）
```
import importlib, utils_shopping
importlib.reload(utils_shopping)
from utils_shopping import load_and_clean, split_xy, train_and_eval

df = load_and_clean("Shopping Trends And Customer Behaviour Dataset.csv")
X, y = split_xy(df, target="Subscription Status")   # 二元或多分類都可以
res = train_and_eval(X, y, task="classification", model="rf", tune=True, n_iter=20)
print(res["metrics"])
```

### 使用範例(迴歸)
```
df = load_and_clean("Shopping Trends And Customer Behaviour Dataset.csv")
X, y = split_xy(df, target="Review Rating")   # 連續數值目標
res = train_and_eval(X, y, task="regression", model="rf", tune=True, n_iter=20)
print(res["metrics"])   # mae / rmse / r2
```

### 附件
🔧 工具包 [./utils_churn.py](./utils_churn.py)  
🔎 使用範例 [./test.ipynb](./test.ipynb)

## 📝 作者說明
💡 本作品為個人練習與求職作品，資料來源自 **Kaggle**，目的在於展示數據分析與機器學習應用能力。  
⚠️ **警告**：本專案數據與分析結果僅供學習參考，切勿用於任何實際商業或投資決策。  

---



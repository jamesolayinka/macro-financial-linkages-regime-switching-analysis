# macro-financial-linkages-regime-switching-analysis
Macro-Financial Linkages Between Equity and Commodity Markets: A Regime-Switching Econometric Analysis of Daily and Long-Term Strategies

A regime-switching econometric exploration of equity–commodity market dynamics and macro-financial linkages.
This project models daily and long-term interactions between equity and commodity markets using Markov-Switching VAR and GARCH frameworks, developing both tactical and strategic investment strategies informed by macroeconomic regimes.

## 💻 Developer Setup & Collaboration Guide

This section provides steps for **Emmanuel** (and any future collaborators) to set up the project locally and contribute effectively using **VS Code** and **Git**.

---

### 1. Set Up Your Development Environment

1. Install **[VS Code](https://code.visualstudio.com/)** and make sure **Git** is installed on your computer.  
2. Clone the repository by running:
   ```bash
   git clone https://github.com/jamesolayinka/macro-financial-linkages-regime-switching-analysis.git

3.	Open the cloned folder in VS Code.
4.	Create or activate the project environment:
```bash
conda env create -f environment.yml
conda activate macro_financial_env
(Alternatively, use pip install -r requirements.txt if you prefer pip.)

### 2. Working With Branches

To ensure smooth collaboration and avoid overwriting each other’s work, always use branches for major changes.
- Create a new branch before you start working on high-traffic or shared files:

```bash
git checkout -b your-branch-name

Example:
git checkout -b emmanuel-data-pipeline

- After making your changes:
git add .                      # or specify files explicitly
git commit -m "Add data cleaning functions in data_pipeline"
git push origin your-branch-name
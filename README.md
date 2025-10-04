# macro-financial-linkages-regime-switching-analysis
Macro-Financial Linkages Between Equity and Commodity Markets: A Regime-Switching Econometric Analysis of Daily and Long-Term Strategies

A regime-switching econometric exploration of equity–commodity market dynamics and macro-financial linkages.
This project models daily and long-term interactions between equity and commodity markets using Markov-Switching VAR and GARCH frameworks, developing both tactical and strategic investment strategies informed by macroeconomic regimes.

## 💻 Developer Setup & Collaboration Guide

This guide outlines how to set up and collaborate effectively on the project using **VS Code** and **Git**.

---

### 1. Set Up Your Development Environment

1. Install [**VS Code**](https://code.visualstudio.com/) and ensure **Git** is installed on your system.
2. Clone the repository by running:
   ```bash
   git clone https://github.com/jamesolayinka/macro-financial-linkages-regime-switching-analysis.git
   ```
3. Open the cloned folder in VS Code.
4. Create or activate the project environment:
   ```bash
   conda env create -f environment.yml
   conda activate macro_financial_env
   ```
   *(Alternatively, you can use pip if preferred:)*
   ```bash
   pip install -r requirements.txt
   ```

---

### 2. Working With Branches

To keep collaboration smooth and avoid overwriting each other’s work, always create a branch for major or shared edits.

- Create a new branch:

  ```bash
  git checkout -b your-branch-name
  ```

  **Example:**

  ```bash
  git checkout -b emmanuel-data-pipeline
  ```

- After finishing your work:

  ```bash
  git add .                      # or specify file names
  git commit -m "Add data cleaning functions in data_pipeline"
  git push origin your-branch-name
  ```

- Then go to GitHub and open a Pull Request (PR) from your branch into `main`.\
  James will review and merge it once confirmed.

---

### 3. Updating the Main Branch

If you’re working directly on `main` (only for minor, low-traffic updates):

```bash
git add .
git commit -m "Describe your update briefly"
git push origin main
```

⚠️ **Important:** Avoid making large edits directly on `main`.\
Always use a branch for major work (e.g., notebooks, model scripts) to ensure we understand and track each other’s changes.

---

### 4. Keeping Your Local Repo Updated

Before starting new work, make sure you have the latest version:

```bash
git checkout main
git pull origin main
```

This ensures your local files are up to date before making any new changes or creating a new branch.

---

### 🤝 Collaboration Principles

- Communicate before working on core files (e.g., modeling scripts or shared notebooks).
- Use clear, descriptive commit messages explaining what was changed and why.
- Review and approve each other’s Pull Requests before merging.
- Keep branches focused — one purpose per branch (e.g., feature addition, bug fix, data update).


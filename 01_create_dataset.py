# ============================================================
# Generate Synthetic Loan Dataset
# Author: Sanath | York University
# ============================================================
# I created this script to generate realistic synthetic loan
# application data. I chose synthetic data instead of real bank
# data because financial records are protected under PIPEDA in
# Canada. I used numpy's normal distribution to mimic real-world
# patterns — most credit scores cluster around 680 and most
# incomes around $60K, which matches actual population data.
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(42)
NUM_SAMPLES = 5000

print("🏦 Generating synthetic loan dataset...")
print(f"   Creating {NUM_SAMPLES} loan applications...\n")

# I'm generating each feature to match what a real loan application form looks like
age = np.random.randint(21, 66, NUM_SAMPLES)

# I used a bell curve (normal distribution) for income since most people earn
# around $60K with fewer at the extremes — this is more realistic than uniform random
income = np.random.normal(60000, 25000, NUM_SAMPLES).clip(20000, 200000).astype(int)

# Credit scores follow a similar bell curve centered around 680
credit_score = np.random.normal(680, 80, NUM_SAMPLES).clip(300, 850).astype(int)

employment_years = np.random.randint(0, 31, NUM_SAMPLES)
loan_amount = np.random.randint(1000, 50001, NUM_SAMPLES)

# DTI (debt-to-income) ratio — banks typically want this below 0.36
dti_ratio = np.random.uniform(0.05, 0.8, NUM_SAMPLES).round(2)

num_credit_lines = np.random.randint(1, 15, NUM_SAMPLES)

# 15% default rate — this ended up being the most important feature in my model
previous_default = np.random.choice([0, 1], NUM_SAMPLES, p=[0.85, 0.15])

purposes = ["debt_consolidation", "home_improvement", "business", "education", "medical", "other"]
loan_purpose = np.random.choice(purposes, NUM_SAMPLES)
home_ownership = np.random.choice(["rent", "own", "mortgage"], NUM_SAMPLES, p=[0.4, 0.2, 0.4])

# I built a realistic approval logic based on how banks actually evaluate loans
# instead of randomly assigning approved/denied — this way the model learns real patterns
print("   Calculating approval decisions based on realistic rules...\n")

approval_score = np.zeros(NUM_SAMPLES)
approval_score += (credit_score - 600) / 100
approval_score += np.clip(income / loan_amount - 2, -1, 2)
approval_score += (0.4 - dti_ratio) * 3
approval_score += employment_years / 15
approval_score -= previous_default * 3
approval_score -= np.clip(num_credit_lines - 8, 0, 5) / 2
approval_score += np.random.normal(0, 0.5, NUM_SAMPLES)

# Threshold gives ~61% approval rate which is realistic for the industry
approved = (approval_score > 0.5).astype(int)

print(f"   Approval rate: {approved.mean()*100:.1f}% ({approved.sum()} approved, {NUM_SAMPLES - approved.sum()} denied)")

df = pd.DataFrame({
    "age": age, "annual_income": income, "credit_score": credit_score,
    "employment_years": employment_years, "loan_amount": loan_amount,
    "dti_ratio": dti_ratio, "num_credit_lines": num_credit_lines,
    "previous_default": previous_default, "loan_purpose": loan_purpose,
    "home_ownership": home_ownership, "approved": approved
})

df.to_csv("data/loan_data.csv", index=False)
print(f"\n✅ Dataset saved to data/loan_data.csv")
print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("📋 First 5 rows:")
print(df.head().to_string())

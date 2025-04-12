#!/usr/bin/env python3
import numpy as np
from scipy.stats import bernoulli, norm, shapiro
from collections import defaultdict

def input_float(prompt, min_val=0, max_val=1e9):
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"请输入一个在 {min_val} 到 {max_val} 之间的数")
        except:
            print("输入无效，请重新输入。")

def input_int(prompt, min_val=1, max_val=1e9):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"请输入一个在 {min_val} 到 {max_val} 之间的整数")
        except:
            print("输入无效，请重新输入。")

def method1(n_students, difficulty, full_score, real_avg, n_questions, delta):
    print("\n=== 方法一：真实总分概率对比法（含分层） ===")

    real_total = real_avg * n_students
    expected_total = difficulty * full_score * n_students

    # 分层定义
    levels = [
        (0.4, min(difficulty * 4 / 3, 1.0)),
        (0.4, max(difficulty * 2 / 3, 0.0)),
        (0.2, difficulty)
    ]

    def simulate_total_score():
        total = 0
        for ratio, level_difficulty in levels:
            num = int(n_students * ratio)
            probs = [level_difficulty] * n_questions
            group_score = bernoulli.rvs(probs, size=(num, len(probs))).sum() * (full_score / n_questions)
            total += group_score
        return total

    simulations = 100000
    scores = [simulate_total_score() for _ in range(simulations)]
    scores = np.array(scores)

    # 计算真实总分和预期总分落在 delta 区间内的概率
    x_prob = np.mean(np.abs(scores - real_total) <= delta)
    y_prob = np.mean(np.abs(scores - expected_total) <= delta)

    real_possibility = 1 - abs(y_prob - x_prob) / y_prob if y_prob != 0 else 0

    print(f"真实总分：{real_total:.4f}")
    print(f"预期总分：{expected_total:.4f}")
    print(f"±{delta} 区间内真实总分的概率 x：{x_prob:.4f}")
    print(f"±{delta} 区间内预期总分的概率 y：{y_prob:.4f}")
    print(f"1 - |y - x| / y （真实可能性）：{real_possibility:.4f}")

    if real_possibility > 0.9:
        print("结论：难度系数极为符合学生水平。")
    elif real_possibility > 0.7:
        print("结论：难度系数较为符合学生水平。")
    else:
        print("结论：难度系数可能不太符合学生水平。")

def method2(n_students, difficulty, full_score, real_avg, n_questions):
    print("\n=== 方法二：可能难度系数与偏差概率差法 ===")
    est_difficulty = real_avg / full_score
    est_probs = [est_difficulty] * n_questions
    fixed_probs = [difficulty] * n_questions

    def sample_totals(probs):
        return bernoulli.rvs(probs, size=(n_students, len(probs))).sum(axis=1).mean()

    samples = 10000
    est_means = np.array([sample_totals(est_probs) for _ in range(samples)])
    fixed_means = np.array([sample_totals(fixed_probs) for _ in range(samples)])

    a = np.mean(fixed_means <= real_avg)
    b = np.mean(est_means >= real_avg)

    print(f"估计的可能难度系数：{est_difficulty:.4f}")
    print(f"P(估算均分 ≥ 实际均分) = {b:.4f}")
    print(f"P(设定均分 ≤ 实际均分) = {a:.4f}")
    print(f"b - a = {b - a:.4f}")

    if b - a > 0.5:
        print("结论：难度系数可能不太符合学生水平。")
    else:
        print("结论：难度系数可能符合学生水平。")

def method3(n_students, difficulty, full_score, real_avg, n_questions):
    print("\n=== 方法三：正态分布近似法 ===")
    probs = [difficulty] * n_questions
    sample_scores = [bernoulli.rvs(probs).sum() * full_score / n_questions for _ in range(n_students)]
    sample_scores = np.array(sample_scores)
    stat, p_value = shapiro(sample_scores)

    print(f"Shapiro-Wilk 正态性检验 p 值：{p_value:.4f}")
    if p_value > 0.05:
        print("结论：样本均值服从正态分布，可以使用正态模型分析。")
        mu = np.mean(sample_scores)
        sigma = np.std(sample_scores) / np.sqrt(n_students)
        z = (real_avg - mu) / sigma
        p = norm.cdf(z)
        print(f"Z = {z:.4f}, P(均值 ≤ 实际均值) = {p:.4f}")
    else:
        print("结论：样本均值不服从正态分布，不适合使用方法三。")

def main():
    print("=== 考试难度系数分析工具 ===")

    n_students = input_int("请输入考试人数: ")
    difficulty = input_float("请输入设定的难度系数 (如 0.6): ", 0, 1)
    full_score = input_float("请输入试卷满分: ", 1)
    real_avg = input_float("请输入实际平均分: ", 0, full_score)
    n_questions = input_int("请输入题目数量: ")
    delta = input_float("请输入误差容差 delta 值（如 1.0）用于方法一: ", 0)

    print("\n正在进行模拟计算，请稍候...")

    method1(n_students, difficulty, full_score, real_avg, n_questions, delta)
    method2(n_students, difficulty, full_score, real_avg, n_questions)
    method3(n_students, difficulty, full_score, real_avg, n_questions)

if __name__ == "__main__":
    main()
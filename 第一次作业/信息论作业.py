import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_entropy(probabilities):
    """计算熵"""
    return entropy(probabilities)

def calculate_joint_entropy(joint_probabilities):
    """计算联合熵"""
    return calculate_entropy(joint_probabilities)

def calculate_conditional_entropy(H_joint, H_cond):
    """计算条件熵 H(X|Y) 或 H(Y|X)"""
    return H_joint - H_cond

def calculate_mutual_information(H_X, H_Y, H_XY):
    """计算互信息 I(X; Y)"""
    return H_X + H_Y - H_XY

def main(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path, header=None)
    joint_probabilities = df.values.flatten()   # 获取联合概率分布
    joint_probabilities /= joint_probabilities.sum()  # 归一化

    # 计算H(X, Y)
    H_XY = calculate_joint_entropy(joint_probabilities)

    # 计算边际概率分布
    p_X = joint_probabilities.reshape(4, 4).sum(axis=1)  # 对每一行求和
    p_Y = joint_probabilities.reshape(4, 4).sum(axis=0)  # 对每一列求和

    # 计算H(X)和H(Y)
    H_X = calculate_entropy(p_X)
    H_Y = calculate_entropy(p_Y)
    
    # 计算条件熵 H(X|Y) 和 H(Y|X)
    H_X_given_Y = calculate_conditional_entropy(H_XY, H_Y)
    H_Y_given_X = calculate_conditional_entropy(H_XY, H_X)
    
    # 计算互信息 I(X; Y)
    I_XY = calculate_mutual_information(H_X, H_Y, H_XY)
    
    # 打印结果
    print("H(X) = -\\sum_{X}^{} \\sum_{Y}^{} p (x_{i} y_{j} )log_{2} p(x_{i} ) = "f"{H_X}")
    print("H(Y) = -\\sum_{X}^{} \\sum_{Y}^{} p (x_{i} y_{j} )log_{2} p(y_{j} ) = "f"{H_Y}")
    print("H(XY) = -\\sum_{X}^{} \\sum_{Y}^{} p (x_{i} y_{j} )log_{2} p(x_{i} y_{j} ) = "f"{H_XY}")
    print("H(X|Y) = -\\sum_{X}^{} \\sum_{Y}^{} p (x_{i} y_{j} )log_{2} \\frac{p(x_{i} y_{j} )}{p(y_{j}) }  ="f" {H_X_given_Y}")
    print("H(Y|X) = -\\sum_{X}^{} \\sum_{Y}^{} p (x_{i} y_{j} )log_{2} \\frac{p(x_{i} y_{j} )}{p(x_{i}) }  ="f" {H_Y_given_X}")
    print(f"I(X; Y)=H(X)+H(Y)-H(XY) = {I_XY}")

# 示例用法
# main('路径到你的excel文件.xlsx')
main(r'C:\Users\hp\Desktop\code\data1.xls')

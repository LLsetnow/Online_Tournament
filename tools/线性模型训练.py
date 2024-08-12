import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def linear_regression_model(file_path):
    """
    读取CSV文件并进行线性回归分析，最后可视化结果。

    Parameters:
    file_path (str): CSV文件路径

    Returns:
    dict: 回归模型的系数和截距
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 提取输入和输出变量
    X = data[['a', 'b']]
    y = data['y']

    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(X, y)

    # 获取回归系数和截距
    coefficients = model.coef_
    intercept = model.intercept_

    print(f'Coefficients: {coefficients}')
    print(f'Intercept: {intercept}')

    # 使用模型进行预测
    y_pred = model.predict(X)

    # 可视化结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['a'], data['b'], data['y'], color='b', label='Actual data')
    ax.scatter(data['a'], data['b'], y_pred, color='r', label='Predicted data')

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('y')
    plt.legend()
    plt.show()

    return {'coefficients': coefficients, 'intercept': intercept}


def polynomial_regression_model(file_path, degree=2):
    """
    读取CSV文件并进行二次（或多次）回归分析，最后可视化结果。

    Parameters:
    file_path (str): CSV文件路径
    degree (int): 多项式回归的阶数，默认为2
    使用degree=2的二次多项式则为（1，a, a^2, ab, b ,b^2)。

    Returns:
    dict: 回归模型的系数和截距
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 提取输入和输出变量
    X = data[['a', 'b']]
    y = data['y']

    # 创建多项式特征
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(X_poly, y)

    # 获取回归系数和截距
    coefficients = model.coef_
    intercept = model.intercept_

    print(f'Coefficients: {coefficients}')
    print(f'Intercept: {intercept}')

    # 使用模型进行预测
    y_pred = model.predict(X_poly)

    # 可视化结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['a'], data['b'], data['y'], color='b', label='Actual data')
    ax.scatter(data['a'], data['b'], y_pred, color='r', label='Predicted data')

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('y')
    plt.legend()
    plt.show()

    return {'coefficients': coefficients, 'intercept': intercept}


# 示例使用
linear_regression_model('D:\github\Online_Tournament\csvFiles\\rescue\\rescue_x_y_size.csv')
# polynomial_regression_model('D:\\github\\Online_Tournament\\csvFiles\\bombcone\\leftCone_x_y_size.csv', degree=2)

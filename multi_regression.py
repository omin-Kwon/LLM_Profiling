import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# 데이터 정의
compute_bound_data = {
    'x1': [256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 2048, 2048, 2048, 4096, 4096, 4096],
    'x2': [4096, 8192, 16384, 8192, 16384, 32768, 16384, 32768, 65536, 32768, 65536, 131072, 65536, 131072, 196608],
    'y': [35.886, 36.099, 36.51, 63.774, 63.898, 64.371, 121.246, 121.448, 122.149, 229.209, 229.866, 231.411, 356.359, 357.363, 357.879]
}


compute_bound_data_kv_cache_less_than_16 = {
    'x1': [
        256, 256, 256, 256,
        512, 512, 512, 512,
        1024, 1024, 1024, 1024,
        2048, 2048, 2048, 2048,
        4096, 4096, 4096, 4096
    ],
    'x2': [
        256, 512, 1024, 2048,
        512, 1024, 2048, 4096,
        1024, 2048, 4096, 8192,
        2048, 4096, 8192, 16384,
        4096, 8192, 16384, 32768
    ],
    'y': [
        34.107, 34.139, 34.278, 34.668,
        60.014, 60.113, 60.265, 60.947,
        113.857, 113.742, 114.973, 115.392,
        214.864, 215.671, 216.308, 218.423,
        335.265, 335.227, 337.227, 339.963
    ]
    }

df = pd.DataFrame(compute_bound_data)
X = df[['x1', 'x2']]
y = df['y']

# 다양한 회귀 모델 시도 및 설명
models = {
    "Linear": LinearRegression(),  # 단순 선형 회귀: y = a*x1 + b*x2 + c
    "Polynomial (Degree 2)": PolynomialFeatures(degree=2) # 2차 다항 회귀: y = a*x1^2 + b*x2^2 + c*x1*x2 + ...
    #"Polynomial (Degree 3)": PolynomialFeatures(degree=3),  # 3차 다항 회귀: y = a*x1^3 + b*x2^3 + c*x1*x2^2 + ...
    #"Polynomial (Degree 4)": PolynomialFeatures(degree=4)   # 4차 다항 회귀: y = a*x1^4 + b*x2^4 + ...
}

results = {}

for name, model in models.items():
    if "Polynomial" in name:
        poly = model
        X_poly = poly.fit_transform(X)
        regressor = LinearRegression()
        regressor.fit(X_poly, y)
        y_pred = regressor.predict(X_poly)
    else:
        regressor = model
        regressor.fit(X, y)
        y_pred = regressor.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    results[name] = {
        "Model": regressor,
        "MSE": mse,
        "R^2": r2
    }
    print(f"{name} Model -> MSE: {mse:.4f}, R^2: {r2:.4f}")

# 가장 적합한 모델 찾기 (MSE 기준)
best_model = min(results, key=lambda k: results[k]["MSE"])
best_regressor = results[best_model]["Model"]



poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

y_pred = best_regressor.predict(X_poly)

# 회귀 계수 출력
coefficients = best_regressor.coef_
intercept = best_regressor.intercept_
feature_names = poly.get_feature_names_out(['x1', 'x2'])

print("Best Model Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"  {feature}: {coef}")

print(f"Intercept: {intercept}")

# 3D 시각화 진행
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 실제 데이터 산점도
ax.scatter(df['x1'], df['x2'], df['y'], color='blue', label='Actual Data')

# 메쉬 그리드 생성
x1_range = np.linspace(df['x1'].min(), df['x1'].max(), 30)
x2_range = np.linspace(df['x2'].min(), df['x2'].max(), 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
X_pred_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
X_pred_grid_poly = poly.transform(X_pred_grid)

# 예측값 계산
y_pred_grid = best_regressor.predict(X_pred_grid_poly).reshape(x1_grid.shape)

# 회귀 평면 시각화
ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5)

# 레이블 설정
ax.set_title("Best Regression Model (Polynomial Degree 2)")
ax.set_xlabel("Batch Size (x1)")
ax.set_ylabel("KV Cache Size (x2)")
ax.set_zlabel("Decoding Latency (y)")

# 범례 추가
plt.legend()
plt.grid(True)
filename="best_regression.png"
# Save plot to file
plt.savefig(filename)
print(f"Plot saved as {filename}")
plt.close()

# 결과 출력
print("Best Model:", best_model)
print("MSE:", results[best_model]["MSE"])
print("R^2:", results[best_model]["R^2"])

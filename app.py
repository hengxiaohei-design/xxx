import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import numpy as np

st.title("成都房价分析Demo")

# 读取数据
df = pd.read_csv("chengdu_housing_v3.csv")

# 模型
X = df[["size", "age", "metro_distance", "school_level"]]
y = df["price"]

model = XGBRegressor()
model.fit(X, y)

# 输入
size = st.slider("面积", 50, 150, 90)
age = st.slider("房龄", 0, 30, 5)
metro = st.slider("地铁距离", 0, 2000, 500)
school = st.selectbox("学区", ["普通", "中等", "优质"])

school_map = {"普通":1, "中等":2, "优质":3}

if st.button("预测"):
    pred = model.predict(np.array([[size, age, metro, school_map[school]]]))[0]
    st.write(f"预测价格：{int(pred)} 元/㎡")
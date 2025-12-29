"""
🎯 交互式生物数据预测系统
功能：上传Excel文件、数据探索、模型训练、交互式预测、可视化
作者：修凯sey
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tempfile
import joblib
import warnings

warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

# 设置页面配置
st.set_page_config(
    page_title="🧬 生物数据预测系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3498DB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D5F5E3;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #27AE60;
    }
    .warning-box {
        background-color: #FDEBD0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F39C12;
    }
    .info-box {
        background-color: #D6EAF8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3498DB;
    }
    .metric-card {
        background-color: #F8F9F9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #D5D8DC;
        text-align: center;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3498DB;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# 应用标题
st.markdown('<h1 class="main-header">🧬 交互式生物数据预测系统</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>系统功能：</b>上传Excel文件 → 数据探索 → 数据预处理 → 模型训练 → 交互式预测 → 可视化分析
</div>
""", unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("📋 导航")
page = st.sidebar.radio(
    "选择功能",
    ["🏠 首页", "📤 上传数据", "🔍 数据探索", "🧹 数据预处理",
     "🤖 模型训练", "📊 模型评估", "🔮 交互式预测", "💾 模型管理"]
)

# ==================== 首页 ====================
if page == "🏠 首页":
    st.markdown('<h2 class="sub-header">欢迎使用生物数据预测系统</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 数据上传</h3>
            <p>支持多个Excel文件同时上传</p>
            <p>自动检测数据格式</p>
            <p>支持数据预览</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 智能分析</h3>
            <p>自动数据探索</p>
            <p>可视化统计分析</p>
            <p>缺失值检测</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 机器学习</h3>
            <p>多种预测模型</p>
            <p>自动超参数优化</p>
            <p>交叉验证评估</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 使用指南
    st.markdown('<h3>📚 使用指南</h3>', unsafe_allow_html=True)

    steps = [
        ("第一步：上传数据", "在'上传数据'页面选择您的Excel文件，支持批量上传"),
        ("第二步：数据探索", "在'数据探索'页面查看数据统计信息、分布和相关性"),
        ("第三步：数据预处理", "处理缺失值、编码分类变量、特征缩放等"),
        ("第四步：模型训练", "选择算法、调整参数、训练预测模型"),
        ("第五步：模型评估", "查看模型性能指标和可视化结果"),
        ("第六步：交互式预测", "使用滑块和输入框进行实时预测"),
        ("第七步：模型管理", "保存和加载训练好的模型")
    ]

    for i, (title, desc) in enumerate(steps, 1):
        with st.expander(f"第{i}步：{title}"):
            st.write(desc)

    # 支持的数据格式
    st.markdown('<h3>📁 支持的数据格式</h3>', unsafe_allow_html=True)
    st.write("""
    - **文件格式**: .xlsx, .xls, .csv
    - **数据类型**: 数值型、分类型、时间序列
    - **数据规模**: 支持大规模数据集（自动分块处理）
    - **特殊处理**: 自动处理合并单元格、缺失值、异常值
    """)

    # 快速开始按钮
    st.markdown("---")
    if st.button("🚀 快速开始", use_container_width=True):
        st.session_state.page = "📤 上传数据"
        st.rerun()

# ==================== 上传数据 ====================
elif page == "📤 上传数据":
    st.markdown('<h2 class="sub-header">📤 上传Excel数据文件</h2>', unsafe_allow_html=True)

    # 上传文件
    uploaded_files = st.file_uploader(
        "选择Excel文件（支持多个文件）",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="可以上传多个Excel文件，系统会自动合并或分别处理"
    )

    if uploaded_files:
        st.markdown(f'<div class="success-box">✅ 已选择 {len(uploaded_files)} 个文件</div>', unsafe_allow_html=True)

        # 处理每个上传的文件
        for uploaded_file in uploaded_files:
            try:
                # 根据文件类型读取
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # 存储到session state
                st.session_state.datasets[uploaded_file.name] = {
                    'df': df,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict()
                }

                st.success(f"✅ 文件 '{uploaded_file.name}' 加载成功！")

                # 显示文件信息
                with st.expander(f"📋 {uploaded_file.name} - 数据预览"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**数据形状**: {df.shape[0]} 行 × {df.shape[1]} 列")
                        st.write(f"**文件大小**: {uploaded_file.size / 1024:.2f} KB")
                    with col2:
                        st.write(f"**缺失值总数**: {df.isnull().sum().sum()}")
                        st.write(f"**重复行数**: {df.duplicated().sum()}")

                    # 显示前几行数据
                    st.write("**数据预览**:")
                    st.dataframe(df.head(), use_container_width=True)

                    # 显示列信息
                    st.write("**列信息**:")
                    col_info = pd.DataFrame({
                        '列名': df.columns,
                        '数据类型': df.dtypes.values,
                        '非空值数量': df.notnull().sum().values,
                        '缺失值数量': df.isnull().sum().values,
                        '缺失率%': (df.isnull().sum().values / len(df) * 100).round(2)
                    })
                    st.dataframe(col_info, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 文件 '{uploaded_file.name}' 加载失败: {str(e)}")

        # 选择当前操作的数据集
        if st.session_state.datasets:
            dataset_names = list(st.session_state.datasets.keys())
            selected_dataset = st.selectbox(
                "选择要操作的数据集",
                dataset_names,
                help="选择要进行探索、预处理和建模的数据集"
            )
            st.session_state.current_dataset = selected_dataset

            # 显示选中的数据集信息
            if selected_dataset:
                dataset_info = st.session_state.datasets[selected_dataset]
                st.markdown(
                    f'<div class="info-box">当前选择: <b>{selected_dataset}</b> | 形状: {dataset_info["shape"]}</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown('<div class="warning-box">⚠️ 请上传Excel或CSV文件开始分析</div>', unsafe_allow_html=True)

        # 提供示例数据下载
        st.write("### 需要示例数据？")
        if st.button("下载示例数据"):
            # 创建示例数据
            sample_data = pd.DataFrame({
                '样本ID': [f'Sample_{i}' for i in range(1, 101)],
                '基因表达量_A': np.random.normal(10, 2, 100),
                '基因表达量_B': np.random.normal(15, 3, 100),
                '基因表达量_C': np.random.normal(8, 1.5, 100),
                '年龄': np.random.randint(20, 70, 100),
                '性别': np.random.choice(['男', '女'], 100),
                '治疗方案': np.random.choice(['A组', 'B组', '对照组'], 100),
                '疾病状态': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
                '生存时间(天)': np.random.exponential(365, 100)
            })

            # 添加一些缺失值
            for col in sample_data.columns[1:-2]:
                mask = np.random.random(100) < 0.05
                sample_data.loc[mask, col] = np.nan

            # 提供下载
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载示例数据 (CSV)",
                data=csv,
                file_name="生物数据示例.csv",
                mime="text/csv",
            )

# ==================== 数据探索 ====================
elif page == "🔍 数据探索":
    st.markdown('<h2 class="sub-header">🔍 数据探索与分析</h2>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.warning("⚠️ 请先上传数据文件")
        st.stop()

    # 选择数据集
    dataset_names = list(st.session_state.datasets.keys())
    selected_dataset = st.selectbox(
        "选择要探索的数据集",
        dataset_names,
        index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset else 0
    )

    if selected_dataset:
        df = st.session_state.datasets[selected_dataset]['df']
        st.session_state.current_dataset = selected_dataset

        # 探索选项
        explore_options = st.multiselect(
            "选择探索功能",
            ["📊 基本统计", "📈 数据分布", "🔗 相关性分析", "📉 缺失值分析", "🎯 目标变量分析", "🔄 数据变换"],
            default=["📊 基本统计", "📈 数据分布"]
        )

        # 1. 基本统计
        if "📊 基本统计" in explore_options:
            st.markdown('<h3>📊 基本统计信息</h3>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总行数", len(df))
            with col2:
                st.metric("总列数", len(df.columns))
            with col3:
                st.metric("总缺失值", df.isnull().sum().sum())

            # 数值型统计
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("**数值型变量统计**:")
                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

            # 分类型统计
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.write("**分类型变量统计**:")
                cat_stats = {}
                for col in categorical_cols:
                    cat_stats[col] = {
                        '类别数': df[col].nunique(),
                        '最常见值': df[col].mode()[0] if not df[col].mode().empty else None,
                        '最常见频数': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                    }
                st.dataframe(pd.DataFrame(cat_stats).T, use_container_width=True)

        # 2. 数据分布
        if "📈 数据分布" in explore_options:
            st.markdown('<h3>📈 数据分布可视化</h3>', unsafe_allow_html=True)

            # 选择要可视化的列
            all_cols = list(df.columns)
            viz_cols = st.multiselect("选择要可视化的列", all_cols, default=all_cols[:min(5, len(all_cols))])

            if viz_cols:
                # 创建子图
                fig = make_subplots(
                    rows=len(viz_cols),
                    cols=2,
                    subplot_titles=[f"{col} - 分布" for col in viz_cols] + [f"{col} - 箱线图" for col in viz_cols],
                    vertical_spacing=0.05
                )

                for i, col in enumerate(viz_cols, 1):
                    if df[col].dtype in ['int64', 'float64']:
                        # 直方图
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col, nbinsx=30),
                            row=i, col=1
                        )
                        # 箱线图
                        fig.add_trace(
                            go.Box(y=df[col], name=col),
                            row=i, col=2
                        )
                    else:
                        # 分类变量的条形图
                        value_counts = df[col].value_counts().head(20)
                        fig.add_trace(
                            go.Bar(x=value_counts.index, y=value_counts.values, name=col),
                            row=i, col=1
                        )
                        # 饼图（只显示前10个类别）
                        top_categories = df[col].value_counts().head(10)
                        fig.add_trace(
                            go.Pie(labels=top_categories.index, values=top_categories.values, name=col),
                            row=i, col=2
                        )

                fig.update_layout(height=300 * len(viz_cols), showlegend=False, title_text="数据分布分析")
                st.plotly_chart(fig, use_container_width=True)

        # 3. 相关性分析
        if "🔗 相关性分析" in explore_options:
            st.markdown('<h3>🔗 相关性分析</h3>', unsafe_allow_html=True)

            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                # 计算相关系数矩阵
                corr_matrix = numeric_df.corr()

                # 热力图
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="特征相关性热力图"
                )
                st.plotly_chart(fig, use_container_width=True)

                # 相关系数表
                st.write("**相关系数矩阵**:")
                st.dataframe(corr_matrix, use_container_width=True)

                # 强相关性特征对
                st.write("**强相关性特征对 (|r| > 0.7)**:")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr = abs(corr_matrix.iloc[i, j])
                        if corr > 0.7:
                            strong_corr.append({
                                '特征1': corr_matrix.columns[i],
                                '特征2': corr_matrix.columns[j],
                                '相关系数': corr_matrix.iloc[i, j]
                            })

                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                else:
                    st.info("未发现强相关性特征对 (|r| > 0.7)")
            else:
                st.warning("数值型特征不足，无法进行相关性分析")

        # 4. 缺失值分析
        if "📉 缺失值分析" in explore_options:
            st.markdown('<h3>📉 缺失值分析</h3>', unsafe_allow_html=True)

            # 缺失值统计
            missing_stats = pd.DataFrame({
                '列名': df.columns,
                '缺失值数量': df.isnull().sum().values,
                '缺失率%': (df.isnull().sum().values / len(df) * 100).round(2)
            }).sort_values('缺失率%', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**缺失值统计**:")
                st.dataframe(missing_stats[missing_stats['缺失值数量'] > 0], use_container_width=True)

            with col2:
                # 缺失值热图
                if df.isnull().sum().sum() > 0:
                    fig = px.imshow(
                        df.isnull(),
                        aspect="auto",
                        labels=dict(x="特征", y="样本", color="是否缺失"),
                        color_continuous_scale=["white", "red"],
                        title="缺失值分布热图"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 缺失值模式分析
            st.write("**缺失值模式**:")
            missing_pattern = df.isnull().sum(axis=1).value_counts().sort_index()
            fig = px.bar(
                x=missing_pattern.index,
                y=missing_pattern.values,
                labels={'x': '每行缺失值数量', 'y': '样本数'},
                title="每行缺失值数量分布"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 5. 目标变量分析
        if "🎯 目标变量分析" in explore_options:
            st.markdown('<h3>🎯 目标变量分析</h3>', unsafe_allow_html=True)

            target_col = st.selectbox("选择目标变量", df.columns, key="target_explore")

            if target_col:
                target_series = df[target_col]

                col1, col2 = st.columns(2)

                with col1:
                    # 目标变量分布
                    if target_series.dtype in ['int64', 'float64']:
                        # 数值型目标变量
                        fig1 = px.histogram(
                            target_series,
                            nbins=30,
                            title=f"{target_col} 分布",
                            labels={'value': target_col}
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                        # 统计信息
                        stats = target_series.describe()
                        st.write("**统计描述**:")
                        st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
                    else:
                        # 分类型目标变量
                        value_counts = target_series.value_counts()
                        fig1 = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"{target_col} 类别分布"
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                        st.write("**类别分布**:")
                        st.dataframe(value_counts, use_container_width=True)

                with col2:
                    # 目标变量与其他变量的关系
                    if target_series.dtype in ['int64', 'float64']:
                        # 回归问题：目标变量与数值特征的关系
                        numeric_features = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_features) > 1:
                            feature_to_plot = st.selectbox(
                                "选择特征查看与目标变量的关系",
                                [col for col in numeric_features if col != target_col]
                            )
                            if feature_to_plot:
                                fig2 = px.scatter(
                                    df,
                                    x=feature_to_plot,
                                    y=target_col,
                                    title=f"{target_col} vs {feature_to_plot}",
                                    trendline="ols"
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                    else:
                        # 分类问题：目标变量与数值特征的关系
                        numeric_features = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_features) > 0:
                            feature_to_plot = st.selectbox(
                                "选择特征查看与目标变量的关系",
                                numeric_features,
                                key="feature_vs_target"
                            )
                            if feature_to_plot:
                                fig2 = px.box(
                                    df,
                                    x=target_col,
                                    y=feature_to_plot,
                                    title=f"{feature_to_plot} 在不同 {target_col} 类别中的分布"
                                )
                                st.plotly_chart(fig2, use_container_width=True)

# ==================== 数据预处理 ====================
elif page == "🧹 数据预处理":
    st.markdown('<h2 class="sub-header">🧹 数据预处理</h2>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.warning("⚠️ 请先上传数据文件")
        st.stop()

    # 选择数据集
    dataset_names = list(st.session_state.datasets.keys())
    selected_dataset = st.selectbox(
        "选择要预处理的数据集",
        dataset_names,
        index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset else 0
    )

    if selected_dataset:
        df = st.session_state.datasets[selected_dataset]['df'].copy()
        st.session_state.current_dataset = selected_dataset

        # 创建预处理选项标签页
        preprocess_tabs = st.tabs(["缺失值处理", "特征编码", "特征缩放", "特征选择", "异常值处理", "数据分割"])

        # 1. 缺失值处理
        with preprocess_tabs[0]:
            st.markdown('<h4>缺失值处理</h4>', unsafe_allow_html=True)

            # 显示缺失值情况
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                st.write(f"**发现 {len(missing_cols)} 个包含缺失值的列**")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("包含缺失值的列:")
                    for col in missing_cols:
                        missing_count = df[col].isnull().sum()
                        missing_pct = missing_count / len(df) * 100
                        st.write(f"- {col}: {missing_count}个缺失 ({missing_pct:.1f}%)")

                with col2:
                    # 缺失值处理策略
                    st.write("**处理策略**:")
                    strategy = st.radio(
                        "选择缺失值处理方法",
                        ["删除缺失行", "数值型：均值填充", "数值型：中位数填充", "分类型：众数填充", "向前填充",
                         "向后填充"],
                        horizontal=True
                    )

                    # 应用处理
                    if st.button("应用缺失值处理", key="impute_missing"):
                        df_processed = df.copy()

                        if strategy == "删除缺失行":
                            df_processed = df_processed.dropna()
                            st.success(f"已删除包含缺失值的行，剩余 {len(df_processed)} 行")

                        elif "数值型" in strategy:
                            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                            numeric_cols_with_missing = [col for col in numeric_cols if col in missing_cols]

                            if strategy == "数值型：均值填充":
                                for col in numeric_cols_with_missing:
                                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                            elif strategy == "数值型：中位数填充":
                                for col in numeric_cols_with_missing:
                                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())

                            st.success(f"已对 {len(numeric_cols_with_missing)} 个数值型列进行填充")

                        elif "分类型" in strategy:
                            categorical_cols = df_processed.select_dtypes(include=['object']).columns
                            categorical_cols_with_missing = [col for col in categorical_cols if col in missing_cols]

                            for col in categorical_cols_with_missing:
                                df_processed[col] = df_processed[col].fillna(
                                    df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown")

                            st.success(f"已对 {len(categorical_cols_with_missing)} 个分类型列进行填充")

                        elif strategy == "向前填充":
                            df_processed = df_processed.fillna(method='ffill')
                            st.success("已使用向前填充方法")

                        elif strategy == "向后填充":
                            df_processed = df_processed.fillna(method='bfill')
                            st.success("已使用向后填充方法")

                        # 更新数据
                        st.session_state.datasets[selected_dataset]['df'] = df_processed
                        st.rerun()
            else:
                st.success("✅ 数据中没有缺失值")

        # 2. 特征编码
        with preprocess_tabs[1]:
            st.markdown('<h4>特征编码</h4>', unsafe_allow_html=True)

            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write(f"**发现 {len(categorical_cols)} 个分类型特征**")

                for col in categorical_cols:
                    with st.expander(f"列: {col}"):
                        unique_vals = df[col].unique()
                        st.write(f"类别数: {len(unique_vals)}")
                        st.write(
                            f"类别值: {', '.join(map(str, unique_vals[:10]))}{'...' if len(unique_vals) > 10 else ''}")

                        # 编码选项
                        encoding_method = st.radio(
                            f"选择编码方法",
                            ["标签编码", "独热编码", "保留原始"],
                            key=f"encode_{col}",
                            horizontal=True
                        )

                        if st.button(f"应用编码到 {col}", key=f"apply_encode_{col}"):
                            df_processed = st.session_state.datasets[selected_dataset]['df'].copy()

                            if encoding_method == "标签编码":
                                # 创建或获取标签编码器
                                if col not in st.session_state.label_encoders:
                                    le = LabelEncoder()
                                    st.session_state.label_encoders[col] = le
                                else:
                                    le = st.session_state.label_encoders[col]

                                # 转换数据
                                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                                st.success(f"已将 '{col}' 转换为标签编码")

                            elif encoding_method == "独热编码":
                                # 使用pandas的get_dummies进行独热编码
                                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                                df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
                                st.success(f"已将 '{col}' 转换为独热编码 ({len(dummies.columns)}个新列)")

                            # 更新数据
                            st.session_state.datasets[selected_dataset]['df'] = df_processed
                            st.rerun()
            else:
                st.success("✅ 数据中没有分类型特征需要编码")

        # 3. 特征缩放
        with preprocess_tabs[2]:
            st.markdown('<h4>特征缩放</h4>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write(f"**发现 {len(numeric_cols)} 个数值型特征**")

                # 选择缩放方法
                scaling_method = st.selectbox(
                    "选择特征缩放方法",
                    ["标准化 (StandardScaler)", "归一化 (MinMaxScaler)", "鲁棒缩放 (RobustScaler)", "无缩放"]
                )

                # 选择要缩放的列
                cols_to_scale = st.multiselect(
                    "选择要缩放的列（默认选择所有数值型列）",
                    numeric_cols,
                    default=numeric_cols
                )

                if st.button("应用特征缩放"):
                    if scaling_method != "无缩放" and cols_to_scale:
                        df_processed = st.session_state.datasets[selected_dataset]['df'].copy()

                        if scaling_method == "标准化 (StandardScaler)":
                            scaler = StandardScaler()
                        elif scaling_method == "归一化 (MinMaxScaler)":
                            scaler = MinMaxScaler()
                        elif scaling_method == "鲁棒缩放 (RobustScaler)":
                            from sklearn.preprocessing import RobustScaler

                            scaler = RobustScaler()

                        # 应用缩放
                        df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])

                        # 保存缩放器
                        st.session_state.scaler = scaler

                        st.success(f"已对 {len(cols_to_scale)} 个特征进行{scaling_method.split(' ')[0]}")

                        # 更新数据
                        st.session_state.datasets[selected_dataset]['df'] = df_processed
                        st.rerun()
            else:
                st.info("没有数值型特征需要缩放")

        # 4. 特征选择
        with preprocess_tabs[3]:
            st.markdown('<h4>特征选择</h4>', unsafe_allow_html=True)

            st.write("**选择要保留的特征列**")
            all_cols = list(df.columns)
            selected_features = st.multiselect(
                "选择特征（取消选择将从数据中移除）",
                all_cols,
                default=all_cols
            )

            # 目标变量选择
            target_col = st.selectbox(
                "选择目标变量（用于特征重要性分析）",
                [None] + list(df.columns)
            )

            if target_col and target_col in selected_features:
                # 特征重要性分析（基于随机森林）
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

                # 准备数据
                X = df[selected_features].drop(columns=[target_col])
                y = df[target_col]

                # 处理缺失值
                X = X.fillna(X.mean())

                # 判断问题类型
                if y.dtype == 'object' or y.nunique() < 10:
                    # 分类问题
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    # 回归问题
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                # 训练模型
                model.fit(X, y)

                # 特征重要性
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    '特征': X.columns,
                    '重要性': importances
                }).sort_values('重要性', ascending=False)

                st.write("**特征重要性排名**:")
                st.dataframe(feature_importance_df, use_container_width=True)

                # 可视化
                fig = px.bar(
                    feature_importance_df.head(20),
                    x='重要性',
                    y='特征',
                    orientation='h',
                    title='Top 20 特征重要性'
                )
                st.plotly_chart(fig, use_container_width=True)

            # 应用特征选择
            if st.button("应用特征选择"):
                if selected_features:
                    df_processed = df[selected_features].copy()
                    st.session_state.datasets[selected_dataset]['df'] = df_processed
                    st.success(f"已选择 {len(selected_features)} 个特征")
                    st.rerun()

        # 5. 异常值处理
        with preprocess_tabs[4]:
            st.markdown('<h4>异常值处理</h4>', unsafe_allow_html=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("**选择要检测异常值的列**")
                outlier_cols = st.multiselect("选择列", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])

                if outlier_cols:
                    # 异常值检测方法
                    method = st.radio(
                        "异常值检测方法",
                        ["Z-score法", "IQR法", "百分位法"],
                        horizontal=True
                    )

                    threshold = st.slider("异常值阈值", 1.0, 5.0, 3.0, 0.5)

                    if st.button("检测异常值"):
                        df_processed = df.copy()
                        outlier_info = {}

                        for col in outlier_cols:
                            data = df_processed[col].dropna()

                            if method == "Z-score法":
                                z_scores = np.abs((data - data.mean()) / data.std())
                                outliers = z_scores > threshold
                                outlier_count = outliers.sum()

                            elif method == "IQR法":
                                Q1 = data.quantile(0.25)
                                Q3 = data.quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - threshold * IQR
                                upper_bound = Q3 + threshold * IQR
                                outliers = (data < lower_bound) | (data > upper_bound)
                                outlier_count = outliers.sum()

                            elif method == "百分位法":
                                lower_bound = data.quantile(0.01 * threshold)
                                upper_bound = data.quantile(1 - 0.01 * threshold)
                                outliers = (data < lower_bound) | (data > upper_bound)
                                outlier_count = outliers.sum()

                            outlier_info[col] = {
                                '异常值数量': outlier_count,
                                '异常值比例': outlier_count / len(data) * 100
                            }

                        # 显示异常值信息
                        outlier_df = pd.DataFrame(outlier_info).T
                        st.write("**异常值统计**:")
                        st.dataframe(outlier_df, use_container_width=True)

                        # 处理选项
                        treatment = st.radio(
                            "异常值处理方法",
                            ["不处理", "删除异常值", "用中位数替换", "用边界值替换"],
                            horizontal=True
                        )

                        if treatment != "不处理" and st.button("应用异常值处理"):
                            for col in outlier_cols:
                                data = df_processed[col].copy()

                                # 重新计算异常值
                                if method == "Z-score法":
                                    z_scores = np.abs((data - data.mean()) / data.std())
                                    outliers = z_scores > threshold
                                elif method == "IQR法":
                                    Q1 = data.quantile(0.25)
                                    Q3 = data.quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - threshold * IQR
                                    upper_bound = Q3 + threshold * IQR
                                    outliers = (data < lower_bound) | (data > upper_bound)
                                elif method == "百分位法":
                                    lower_bound = data.quantile(0.01 * threshold)
                                    upper_bound = data.quantile(1 - 0.01 * threshold)
                                    outliers = (data < lower_bound) | (data > upper_bound)

                                if treatment == "删除异常值":
                                    df_processed = df_processed[~outliers]
                                elif treatment == "用中位数替换":
                                    median_val = data.median()
                                    df_processed.loc[outliers, col] = median_val
                                elif treatment == "用边界值替换":
                                    if method == "IQR法":
                                        Q1 = data.quantile(0.25)
                                        Q3 = data.quantile(0.75)
                                        IQR = Q3 - Q1
                                        lower_bound = Q1 - threshold * IQR
                                        upper_bound = Q3 + threshold * IQR
                                    elif method == "百分位法":
                                        lower_bound = data.quantile(0.01 * threshold)
                                        upper_bound = data.quantile(1 - 0.01 * threshold)
                                    else:  # Z-score法
                                        mean_val = data.mean()
                                        std_val = data.std()
                                        lower_bound = mean_val - threshold * std_val
                                        upper_bound = mean_val + threshold * std_val

                                    df_processed.loc[outliers & (data < lower_bound), col] = lower_bound
                                    df_processed.loc[outliers & (data > upper_bound), col] = upper_bound

                            st.session_state.datasets[selected_dataset]['df'] = df_processed
                            st.success(f"异常值处理完成，剩余 {len(df_processed)} 行数据")
                            st.rerun()
            else:
                st.info("没有数值型特征进行异常值检测")

        # 6. 数据分割
        with preprocess_tabs[5]:
            st.markdown('<h4>数据分割</h4>', unsafe_allow_html=True)

            # 目标变量选择
            target_options = [None] + list(df.columns)
            target_col = st.selectbox("选择目标变量", target_options, key="split_target")

            if target_col:
                # 分割参数
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
                with col2:
                    random_state = st.number_input("随机种子", 0, 100, 42)
                with col3:
                    shuffle = st.checkbox("打乱数据", True)

                # 保存分割参数到session state
                st.session_state.split_params = {
                    'target_col': target_col,
                    'test_size': test_size,
                    'random_state': random_state,
                    'shuffle': shuffle
                }

                st.success(f"✅ 已设置目标变量: {target_col}")
                st.info(f"将按照 {1 - test_size:.0%}/{test_size:.0%} 的比例分割训练集和测试集")

# ==================== 模型训练 ====================
elif page == "🤖 模型训练":
    st.markdown('<h2 class="sub-header">🤖 机器学习模型训练</h2>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.warning("⚠️ 请先上传并预处理数据")
        st.stop()

    # 选择数据集
    dataset_names = list(st.session_state.datasets.keys())
    selected_dataset = st.selectbox(
        "选择要建模的数据集",
        dataset_names,
        index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset else 0
    )

    if selected_dataset:
        df = st.session_state.datasets[selected_dataset]['df']

        # 检查是否有分割参数
        if 'split_params' not in st.session_state:
            st.warning("⚠️ 请先在'数据预处理'页面设置数据分割参数")
            st.stop()

        split_params = st.session_state.split_params
        target_col = split_params['target_col']

        if target_col not in df.columns:
            st.error(f"❌ 目标变量 '{target_col}' 不在数据集中")
            st.stop()

        # 准备数据
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 判断问题类型
        if y.dtype == 'object' or y.nunique() < 10:
            problem_type = 'classification'
            st.info(f"🔍 检测到分类问题，目标变量有 {y.nunique()} 个类别")
        else:
            problem_type = 'regression'
            st.info(f"🔍 检测到回归问题，目标变量为连续数值")

        # 模型选择
        st.markdown('<h4>选择机器学习模型</h4>', unsafe_allow_html=True)

        # 根据问题类型显示不同的模型选项
        if problem_type == 'classification':
            models = {
                "随机森林": RandomForestClassifier,
                "梯度提升": GradientBoostingClassifier,
                "逻辑回归": LogisticRegression,
                "支持向量机": SVC,
                "K近邻": KNeighborsClassifier,
                "决策树": DecisionTreeClassifier,
                "神经网络": MLPClassifier,
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "AdaBoost": AdaBoostClassifier,
                "朴素贝叶斯": GaussianNB
            }
        else:
            models = {
                "随机森林": RandomForestRegressor,
                "梯度提升": GradientBoostingRegressor,
                "线性回归": LinearRegression,
                "支持向量回归": SVR,
                "K近邻回归": KNeighborsRegressor,
                "决策树回归": DecisionTreeRegressor,
                "神经网络回归": MLPRegressor,
                "XGBoost回归": xgb.XGBRegressor,
                "LightGBM回归": lgb.LGBMRegressor,
                "AdaBoost回归": AdaBoostRegressor,
                "岭回归": Ridge,
                "Lasso回归": Lasso
            }

        # 模型选择
        selected_model_name = st.selectbox("选择模型", list(models.keys()))

        # 显示模型描述
        model_descriptions = {
            "随机森林": "集成学习算法，通过多个决策树投票，抗过拟合能力强",
            "梯度提升": "逐步优化模型，通过梯度下降减少残差",
            "逻辑回归/线性回归": "线性模型，适合线性可分数据，解释性强",
            "支持向量机": "通过寻找最大间隔超平面进行分类，适合高维数据",
            "K近邻": "基于相似性度量，简单直观",
            "决策树": "树形结构，可解释性强",
            "神经网络": "深度学习模型，适合复杂非线性关系",
            "XGBoost": "优化的梯度提升算法，性能优异",
            "LightGBM": "基于直方图的梯度提升，训练速度快",
            "AdaBoost": "自适应提升算法，关注困难样本",
            "朴素贝叶斯": "基于贝叶斯定理，适合文本分类",
            "岭回归": "线性回归+L2正则化，防止过拟合",
            "Lasso回归": "线性回归+L1正则化，可进行特征选择"
        }

        st.info(f"**{selected_model_name}**: {model_descriptions.get(selected_model_name, '')}")

        # 超参数调节
        st.markdown('<h4>超参数调节</h4>', unsafe_allow_html=True)

        # 根据选择的模型显示相应的超参数
        params = {}

        if selected_model_name in ["随机森林", "梯度提升", "决策树"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                params['n_estimators'] = st.slider("树的数量", 10, 500, 100,
                                                   10) if selected_model_name != "决策树" else 1
            with col2:
                params['max_depth'] = st.slider("最大深度", 1, 20, 10)
            with col3:
                params['random_state'] = st.number_input("随机种子", 0, 100, 42)

        elif selected_model_name == "神经网络":
            col1, col2 = st.columns(2)
            with col1:
                hidden_layers = st.text_input("隐藏层结构", "100,50",
                                              help="例如: 100,50 表示两个隐藏层，分别有100和50个神经元")
                params['hidden_layer_sizes'] = tuple(map(int, hidden_layers.split(','))) if hidden_layers else (100,)
            with col2:
                params['max_iter'] = st.slider("最大迭代次数", 100, 5000, 1000, 100)
                params['random_state'] = st.number_input("随机种子", 0, 100, 42)

        elif selected_model_name in ["XGBoost", "LightGBM"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10)
            with col2:
                params['max_depth'] = st.slider("最大深度", 1, 20, 6)
            with col3:
                params['learning_rate'] = st.slider("学习率", 0.01, 0.5, 0.1, 0.01)

        elif selected_model_name == "支持向量机":
            col1, col2 = st.columns(2)
            with col1:
                params['C'] = st.slider("正则化参数C", 0.1, 10.0, 1.0, 0.1)
            with col2:
                params['kernel'] = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"])

        elif selected_model_name == "K近邻":
            params['n_neighbors'] = st.slider("邻居数量", 1, 20, 5)

        # 训练选项
        st.markdown('<h4>训练选项</h4>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            use_cross_val = st.checkbox("使用交叉验证", True)
        with col2:
            if use_cross_val:
                cv_folds = st.slider("交叉验证折数", 3, 10, 5)
        with col3:
            verbose = st.checkbox("显示训练详情", False)

        # 开始训练按钮
        if st.button("🚀 开始训练模型", use_container_width=True):
            with st.spinner("正在训练模型，请稍候..."):
                try:
                    # 创建模型实例
                    model_class = models[selected_model_name]

                    # 处理特殊参数
                    model_params = {}
                    for key, value in params.items():
                        if key == 'hidden_layer_sizes' and isinstance(value, str):
                            model_params[key] = tuple(map(int, value.split(',')))
                        else:
                            model_params[key] = value

                    # 创建模型
                    model = model_class(**model_params)

                    # 数据分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=split_params['test_size'],
                        random_state=split_params['random_state'],
                        shuffle=split_params['shuffle'],
                        stratify=y if problem_type == 'classification' else None
                    )

                    # 特征缩放
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # 保存缩放器
                    st.session_state.scaler = scaler

                    # 训练模型
                    if use_cross_val:
                        # 交叉验证
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds,
                                                    scoring='accuracy' if problem_type == 'classification' else 'r2')

                        # 显示交叉验证结果
                        st.success(f"✅ 交叉验证完成 ({cv_folds}折)")
                        st.write(f"交叉验证得分: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

                        # 可视化交叉验证结果
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f'Fold {i + 1}' for i in range(cv_folds)],
                                y=cv_scores,
                                text=[f'{score:.4f}' for score in cv_scores],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title=f"{selected_model_name} 交叉验证结果",
                            yaxis_title="得分",
                            xaxis_title="折叠",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 最终训练
                    model.fit(X_train_scaled, y_train)

                    # 保存模型到session state
                    st.session_state.model = model
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_test = y_test
                    st.session_state.X_train = X_train_scaled
                    st.session_state.y_train = y_train
                    st.session_state.feature_names = list(X.columns)
                    st.session_state.problem_type = problem_type
                    st.session_state.model_name = selected_model_name

                    st.success(f"✅ {selected_model_name} 模型训练完成！")

                    # 显示训练信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("训练集大小", len(X_train))
                    with col2:
                        st.metric("测试集大小", len(X_test))
                    with col3:
                        st.metric("特征数量", X_train.shape[1])

                except Exception as e:
                    st.error(f"❌ 模型训练失败: {str(e)}")

# ==================== 模型评估 ====================
elif page == "📊 模型评估":
    st.markdown('<h2 class="sub-header">📊 模型性能评估</h2>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        st.stop()

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    problem_type = st.session_state.problem_type

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    st.markdown('<h4>模型性能指标</h4>', unsafe_allow_html=True)

    if problem_type == 'classification':
        # 分类指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 显示指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("准确率", f"{accuracy:.4f}")
        with col2:
            st.metric("精确率", f"{precision:.4f}")
        with col3:
            st.metric("召回率", f"{recall:.4f}")
        with col4:
            st.metric("F1分数", f"{f1:.4f}")

        # 详细分类报告
        st.write("**详细分类报告**:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        # 混淆矩阵
        st.write("**混淆矩阵**:")
        cm = confusion_matrix(y_test, y_pred)

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="预测标签", y="真实标签", color="数量"),
            title="混淆矩阵"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ROC曲线（如果是二分类）
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC曲线 (AUC = {roc_auc:.2f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='随机分类器', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='ROC曲线',
                xaxis_title='假阳性率',
                yaxis_title='真阳性率',
                showlegend=True
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    else:
        # 回归指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 显示指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("均方误差 (MSE)", f"{mse:.4f}")
        with col2:
            st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
        with col3:
            st.metric("R²分数", f"{r2:.4f}")

        # 预测 vs 真实值图
        st.write("**预测结果 vs 真实值**:")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode='markers',
            marker=dict(size=10, opacity=0.6),
            name='预测点'
        ))
        fig.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='完美预测线'
        ))
        fig.update_layout(
            title='预测值 vs 真实值',
            xaxis_title='真实值',
            yaxis_title='预测值',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # 残差图
        st.write("**残差分析**:")
        residuals = y_test - y_pred

        fig_res = make_subplots(
            rows=1, cols=2,
            subplot_titles=('残差分布', '残差 vs 预测值')
        )

        fig_res.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='残差分布'),
            row=1, col=1
        )
        fig_res.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差'),
            row=1, col=2
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        fig_res.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

    # 特征重要性（如果可用）
    if hasattr(model, 'feature_importances_'):
        st.markdown('<h4>特征重要性</h4>', unsafe_allow_html=True)

        importances = model.feature_importances_
        feature_names = st.session_state.feature_names

        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        }).sort_values('重要性', ascending=False)

        # 显示表格
        st.dataframe(importance_df, use_container_width=True)

        # 可视化
        fig = px.bar(
            importance_df.head(20),
            x='重要性',
            y='特征',
            orientation='h',
            title='Top 20 特征重要性',
            color='重要性',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== 交互式预测 ====================
elif page == "🔮 交互式预测":
    st.markdown('<h2 class="sub-header">🔮 交互式预测</h2>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ 请先训练模型")
        st.stop()

    model = st.session_state.model
    scaler = st.session_state.scaler
    feature_names = st.session_state.feature_names
    problem_type = st.session_state.problem_type

    st.info(f"当前使用模型: **{st.session_state.model_name}** | 问题类型: **{problem_type}**")

    # 创建预测标签页
    predict_tabs = st.tabs(["手动输入", "批量预测", "参数探索"])

    # 1. 手动输入预测
    with predict_tabs[0]:
        st.write("### 手动输入特征值进行预测")

        # 动态创建输入框
        input_values = {}
        cols = st.columns(3)  # 每行3列

        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                # 获取该特征的统计信息用于指导输入
                if hasattr(st.session_state, 'X_train'):
                    # 从训练数据中获取统计信息
                    train_data = st.session_state.X_train[:, i]
                    mean_val = train_data.mean()
                    std_val = train_data.std()
                    min_val = train_data.min()
                    max_val = train_data.max()

                    # 创建输入框
                    input_values[feature] = st.slider(
                        f"{feature}",
                        float(min_val - 2 * std_val),
                        float(max_val + 2 * std_val),
                        float(mean_val),
                        help=f"范围: [{min_val:.2f}, {max_val:.2f}]"
                    )
                else:
                    # 如果没有训练数据信息，使用通用范围
                    input_values[feature] = st.number_input(f"{feature}", value=0.0)

        # 预测按钮
        if st.button("进行预测", key="manual_predict"):
            # 准备输入数据
            input_array = np.array([input_values[feature] for feature in feature_names]).reshape(1, -1)

            # 特征缩放
            if scaler is not None:
                input_scaled = scaler.transform(input_array)
            else:
                input_scaled = input_array

            # 进行预测
            prediction = model.predict(input_scaled)[0]

            # 显示结果
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write(f"### 📊 预测结果")

            if problem_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_scaled)[0]
                    st.write(f"**预测类别**: {prediction}")
                    st.write(f"**各类别概率**:")

                    # 显示概率条形图
                    prob_df = pd.DataFrame({
                        '类别': [str(i) for i in range(len(probabilities))],
                        '概率': probabilities
                    })

                    fig = px.bar(
                        prob_df,
                        x='类别',
                        y='概率',
                        title='类别概率分布',
                        color='概率',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"**预测结果**: {prediction}")
            else:
                st.write(f"**预测值**: {prediction:.4f}")

            st.markdown('</div>', unsafe_allow_html=True)

    # 2. 批量预测
    with predict_tabs[1]:
        st.write("### 批量预测（上传新数据文件）")

        uploaded_pred_file = st.file_uploader(
            "上传包含特征数据的Excel或CSV文件",
            type=['xlsx', 'xls', 'csv'],
            key="prediction_file"
        )

        if uploaded_pred_file:
            try:
                # 读取文件
                if uploaded_pred_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_pred_file)
                else:
                    new_data = pd.read_excel(uploaded_pred_file)

                st.success(f"✅ 文件加载成功，数据形状: {new_data.shape}")

                # 检查特征是否匹配
                missing_features = [f for f in feature_names if f not in new_data.columns]
                extra_features = [f for f in new_data.columns if f not in feature_names]

                if missing_features:
                    st.warning(f"⚠️ 数据中缺少以下特征: {missing_features}")

                if extra_features:
                    st.info(f"ℹ️ 数据中包含额外特征: {extra_features}")

                # 选择要使用的特征
                if missing_features:
                    st.warning("无法进行预测，特征不匹配")
                else:
                    # 提取特征数据
                    X_new = new_data[feature_names]

                    # 处理缺失值
                    if X_new.isnull().any().any():
                        st.warning("⚠️ 数据中存在缺失值，将使用中位数填充")
                        X_new = X_new.fillna(X_new.median())

                    # 特征缩放
                    if scaler is not None:
                        X_new_scaled = scaler.transform(X_new)
                    else:
                        X_new_scaled = X_new.values

                    # 批量预测
                    if st.button("开始批量预测", key="batch_predict"):
                        with st.spinner("正在批量预测..."):
                            predictions = model.predict(X_new_scaled)

                            # 如果是分类且有概率预测
                            if problem_type == 'classification' and hasattr(model, 'predict_proba'):
                                probabilities = model.predict_proba(X_new_scaled)

                                # 创建结果DataFrame
                                result_df = new_data.copy()
                                result_df['预测类别'] = predictions

                                # 添加每个类别的概率
                                for i in range(probabilities.shape[1]):
                                    result_df[f'类别_{i}_概率'] = probabilities[:, i]

                            else:
                                result_df = new_data.copy()
                                result_df['预测值'] = predictions

                            # 显示结果
                            st.success(f"✅ 预测完成，共 {len(predictions)} 条记录")

                            # 显示结果预览
                            st.write("**预测结果预览**:")
                            st.dataframe(result_df.head(), use_container_width=True)

                            # 统计信息
                            if problem_type == 'classification':
                                prediction_counts = pd.Series(predictions).value_counts()
                                st.write("**预测类别分布**:")
                                st.dataframe(prediction_counts, use_container_width=True)

                                # 可视化分布
                                fig = px.pie(
                                    values=prediction_counts.values,
                                    names=prediction_counts.index,
                                    title='预测类别分布'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("**预测值统计**:")
                                st.dataframe(pd.Series(predictions).describe(), use_container_width=True)

                            # 下载结果
                            csv = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 下载预测结果 (CSV)",
                                data=csv,
                                file_name="预测结果.csv",
                                mime="text/csv",
                            )

            except Exception as e:
                st.error(f"❌ 文件处理失败: {str(e)}")

    # 3. 参数探索
    with predict_tabs[2]:
        st.write("### 参数探索与影响分析")

        if len(feature_names) >= 2:
            # 选择要探索的特征
            col1, col2 = st.columns(2)
            with col1:
                feature_x = st.selectbox("选择X轴特征", feature_names, index=0)
            with col2:
                feature_y = st.selectbox("选择Y轴特征", feature_names, index=1 if len(feature_names) > 1 else 0)

            # 创建网格
            x_range = st.slider(f"{feature_x} 范围", -3.0, 3.0, (-2.0, 2.0), 0.1)
            y_range = st.slider(f"{feature_y} 范围", -3.0, 3.0, (-2.0, 2.0), 0.1)
            grid_size = st.slider("网格大小", 10, 100, 50)

            if st.button("生成预测热图", key="heatmap_predict"):
                # 创建网格
                x_values = np.linspace(x_range[0], x_range[1], grid_size)
                y_values = np.linspace(y_range[0], y_range[1], grid_size)
                xx, yy = np.meshgrid(x_values, y_values)

                # 创建基础特征矩阵（所有特征取平均值）
                base_values = np.zeros((grid_size * grid_size, len(feature_names)))

                # 获取其他特征的平均值
                if hasattr(st.session_state, 'X_train'):
                    other_feature_means = st.session_state.X_train.mean(axis=0)
                else:
                    other_feature_means = np.zeros(len(feature_names))

                # 填充基础值
                for i, feature in enumerate(feature_names):
                    base_values[:, i] = other_feature_means[i]

                # 设置选定特征的值
                x_idx = feature_names.index(feature_x)
                y_idx = feature_names.index(feature_y)

                base_values[:, x_idx] = xx.ravel()
                base_values[:, y_idx] = yy.ravel()

                # 进行预测
                predictions = model.predict(base_values)

                # 重塑预测结果
                if problem_type == 'classification':
                    zz = predictions.reshape(xx.shape)
                else:
                    zz = predictions.reshape(xx.shape)

                # 创建热图
                fig = go.Figure(data=[
                    go.Contour(
                        x=x_values,
                        y=y_values,
                        z=zz,
                        colorscale='Viridis',
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=12, color='white')
                        )
                    )
                ])

                fig.update_layout(
                    title=f"预测热图: {feature_x} vs {feature_y}",
                    xaxis_title=feature_x,
                    yaxis_title=feature_y,
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # 添加散点图显示训练数据点
                if hasattr(st.session_state, 'X_train'):
                    train_data = st.session_state.X_train

                    # 获取选定特征的训练数据
                    x_train = train_data[:, x_idx]
                    y_train = train_data[:, y_idx]

                    fig.add_trace(go.Scatter(
                        x=x_train,
                        y=y_train,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color='red',
                            symbol='circle',
                            opacity=0.6
                        ),
                        name='训练数据点'
                    ))

                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("需要至少2个特征才能进行参数探索")

# ==================== 模型管理 ====================
elif page == "💾 模型管理":
    st.markdown('<h2 class="sub-header">💾 模型管理</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h4>保存当前模型</h4>', unsafe_allow_html=True)

        if st.session_state.model is not None:
            model_name = st.text_input("模型名称", "bio_model")
            include_scaler = st.checkbox("包含特征缩放器", True)
            include_encoders = st.checkbox("包含标签编码器", True)

            if st.button("💾 保存模型"):
                try:
                    # 准备保存的数据
                    save_data = {
                        'model': st.session_state.model,
                        'model_name': st.session_state.model_name,
                        'problem_type': st.session_state.problem_type,
                        'feature_names': st.session_state.feature_names,
                        'timestamp': pd.Timestamp.now()
                    }

                    if include_scaler and st.session_state.scaler is not None:
                        save_data['scaler'] = st.session_state.scaler

                    if include_encoders and st.session_state.label_encoders:
                        save_data['label_encoders'] = st.session_state.label_encoders

                    # 保存模型
                    filename = f"{model_name}.pkl"
                    joblib.dump(save_data, filename)

                    st.success(f"✅ 模型已保存为: {filename}")

                    # 提供下载
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="📥 下载模型文件",
                            data=f,
                            file_name=filename,
                            mime="application/octet-stream"
                        )

                except Exception as e:
                    st.error(f"❌ 保存失败: {str(e)}")
        else:
            st.info("没有可保存的模型")

    with col2:
        st.markdown('<h4>加载已有模型</h4>', unsafe_allow_html=True)

        uploaded_model = st.file_uploader(
            "上传模型文件 (.pkl)",
            type=['pkl'],
            key="model_upload"
        )

        if uploaded_model:
            try:
                # 保存上传的文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(uploaded_model.getvalue())
                    tmp_path = tmp_file.name

                # 加载模型
                loaded_data = joblib.load(tmp_path)

                # 更新session state
                if 'model' in loaded_data:
                    st.session_state.model = loaded_data['model']
                    st.success("✅ 模型加载成功！")

                    # 显示模型信息
                    st.write("**模型信息**:")
                    info_cols = st.columns(2)

                    with info_cols[0]:
                        st.metric("模型名称", loaded_data.get('model_name', '未知'))
                        st.metric("问题类型", loaded_data.get('problem_type', '未知'))

                    with info_cols[1]:
                        st.metric("特征数量", len(loaded_data.get('feature_names', [])))
                        if 'timestamp' in loaded_data:
                            st.metric("创建时间", loaded_data['timestamp'].strftime('%Y-%m-%d %H:%M'))

                    # 加载其他组件
                    if 'scaler' in loaded_data:
                        st.session_state.scaler = loaded_data['scaler']
                        st.info("✅ 特征缩放器已加载")

                    if 'label_encoders' in loaded_data:
                        st.session_state.label_encoders = loaded_data['label_encoders']
                        st.info("✅ 标签编码器已加载")

                    # 更新feature_names
                    if 'feature_names' in loaded_data:
                        st.session_state.feature_names = loaded_data['feature_names']

                else:
                    st.error("❌ 模型文件格式不正确")

                # 删除临时文件
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"❌ 加载失败: {str(e)}")

    # 模型性能对比
    st.markdown("---")
    st.markdown('<h4>模型性能对比</h4>', unsafe_allow_html=True)

    if st.session_state.model is not None and hasattr(st.session_state, 'X_test') and hasattr(st.session_state,
                                                                                              'y_test'):
        # 快速测试多个模型
        if st.button("🔄 快速模型对比"):
            with st.spinner("正在对比多个模型..."):
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                problem_type = st.session_state.problem_type

                # 选择模型
                if problem_type == 'classification':
                    test_models = {
                        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
                        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
                        '支持向量机': SVC(random_state=42),
                        'K近邻': KNeighborsClassifier(),
                        '决策树': DecisionTreeClassifier(random_state=42)
                    }
                    scoring_func = accuracy_score
                    scoring_name = "准确率"
                else:
                    test_models = {
                        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
                        '线性回归': LinearRegression(),
                        '支持向量回归': SVR(),
                        'K近邻回归': KNeighborsRegressor(),
                        '决策树回归': DecisionTreeRegressor(random_state=42)
                    }
                    scoring_func = lambda y_true, y_pred: r2_score(y_true, y_pred)
                    scoring_name = "R²分数"

                # 训练和评估每个模型
                results = []
                for name, model in test_models.items():
                    try:
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        y_pred = model.predict(X_test)
                        score = scoring_func(y_test, y_pred)
                        results.append({'模型': name, scoring_name: score})
                    except Exception as e:
                        st.warning(f"{name} 训练失败: {str(e)}")

                if results:
                    results_df = pd.DataFrame(results).sort_values(scoring_name, ascending=False)

                    st.write(f"**模型性能对比 ({scoring_name})**:")
                    st.dataframe(results_df, use_container_width=True)

                    # 可视化对比
                    fig = px.bar(
                        results_df,
                        x='模型',
                        y=scoring_name,
                        title=f'模型性能对比',
                        color=scoring_name,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("需要先训练模型才能进行对比")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7F8C8D;'>
        <p>🧬 生物数据预测系统 | 基于Streamlit构建 | 支持多种机器学习算法</p>
    </div>
    """,
    unsafe_allow_html=True
)
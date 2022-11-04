import streamlit as st
import math
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# CONFIG

GRAPH_COLS = 5
SAMPLE_SHOW_NUM = 3

# ページタイトルと概要

st.title("Basic Statistics App")

st.write("""
### データ分析をもっと身近に。

初歩的な探索的データ分析を、クリック操作だけで簡単に実施します。
現在対応しているのは、
- 欠損値の確認
- 要約統計量の確認
    - 各データの分布/割合をグラフ化
- 簡単な重回帰分析
までとなります。

""")

st.write("#")

# 1. データを読み込む
st.subheader("1. データを読み込む")
uploaded_file=st.file_uploader("csvファイルをアップロードしてください。", type='csv')

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
    # 1-1. サンプルの表示
        st.write("##### データサンプル")
        st.dataframe(df.head(SAMPLE_SHOW_NUM))

    # 2. 要約統計量の確認
    st.write("#")
    st.subheader("2. 要約統計量の確認")
    if uploaded_file is not None:

        st.write(f"##### サンプルサイズ:  {df.shape[0]}")
        st.write(f"##### カラム数      :  {df.shape[1]}")

        st.write("###")
        st.write("##### 欠損値")
        st.write("各カラムの欠損値の個数を表示します。")

        null_df = pd.DataFrame(df.isnull().sum(), columns=["null"])
        st.dataframe(null_df)
        
        # TODO: 欠損値をどう埋めるか？を処理できるようにしたい。

        st.write("###")
        st.write("##### 要約統計量 (数値データのみ)")
        
        # TODO: カテゴリ変数に対応したいが、時系列データはdescribeでエラーをはくので、要改善
        # st.write(df.describe())
        st.dataframe(df.describe())

    # 3. 各データの分布/割合を確認
    st.write("#")
    st.subheader("3. 各データの分布を確認")
    if uploaded_file is not None:


        st.write("##### 分布の確認")
        cols_list = list(df.columns)
        options_hist = st.multiselect(
        'ヒストグラムで表示するカラムを選択 (初期値は最初の10カラム)',
        cols_list,
        cols_list[:10])

        if len(options_hist) > 0:

            rows = int(math.ceil(len(options_hist) / GRAPH_COLS))

            # グラフ描画エリアの設定
            fig = make_subplots(
                rows=rows, cols=GRAPH_COLS,
                subplot_titles=options_hist
            )

            # n行5列でグラフを描画
            for n, option in enumerate(options_hist):

                row = int(n // GRAPH_COLS) + 1
                col = int(n % GRAPH_COLS)  + 1

                fig.add_trace(
                    go.Histogram(x=df[option], name=option),
                    row=row, col=col
                )

                if col == 1:
                    fig.update_yaxes(title_text="counts", row=row, col=col)

            # グラフエリアの縦横長とgapの設定
            fig.update_layout(height=750, width=1000, bargap=0.2)
            st.plotly_chart(fig, use_container_width=False)


        st.write("###")
        st.write("##### 割合の確認 (カテゴリ変数)")
        # カテゴリ変数の円グラフ化
        df_obj_cols = list(df.select_dtypes(include='object').columns)

        options_pie = st.multiselect(
        '円グラフで表示するカラムを選択 (初期値は最初の10カラム)',
        df_obj_cols,
        df_obj_cols[:10])

        if len(options_pie) > 0:

            rows = int(math.ceil(len(options_pie) / GRAPH_COLS))

            specs = [[{'type':'domain'} for n in range(GRAPH_COLS)] for n in range(rows)]

            # グラフ描画エリアの設定
            fig = make_subplots(
                rows=rows, cols=GRAPH_COLS,
                specs=specs,
                subplot_titles=options_pie
            )

            # n行5列でグラフを描画
            for n, option_pie in enumerate(options_pie):

                row = int(n // GRAPH_COLS) + 1
                col = int(n % GRAPH_COLS)  + 1

                x = df[option_pie].value_counts()

                fig.add_trace(
                    go.Pie(labels=x.index, values=x),
                    row=row, col=col
                )
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig, use_container_width=False)


    # 4. 相関係数の表示
    st.write("#")
    st.subheader("4. 相関係数")
    if uploaded_file is not None:
        fig = px.imshow(df.corr(), text_auto=True)
        st.plotly_chart(fig)

    # 5. 重回帰分析
    st.write("#")
    st.subheader("5. 重回帰分析")
    if uploaded_file is not None:
        scaler = StandardScaler()
        
        cols_x = list(df.columns)
        options_multi_reg = st.multiselect(
            '説明変数を選択してください。',
            cols_x,
            cols_x
        )
        option_target = st.selectbox(
            '目標値を選択してください。',
            cols_x
        )

        result_flag = 0

        if len(options_multi_reg) > 0 and (st.button("重回帰分析 開始")):
            st.write("分析を開始しました。")
            x = df[options_multi_reg]
            y = df[option_target]

            x_scaled = scaler.fit_transform(x)

            multi_OLS = sm.OLS(y, sm.add_constant(x_scaled)) 
            result = multi_OLS.fit()

            st.write("分析が終了しました。結果を表示します。")
            st.text(result.summary())

            st.write(f"自由度調整済決定係数は{result.rsquared_adj:.2f}です。")
            
            # result_flag = 1
        
        # if (st.button("予測を開始")):
            
            pred = result.predict(sm.add_constant(x_scaled))
            num = list(range(0, len(x))) 

            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(x=num, y=y,
                                mode='markers',
                                name='target'))
            fig.add_trace(go.Scatter(x=num, y=pred,
                                mode='markers',
                                name='prediction'))
            fig.update_xaxes(title_text="Sample No.")
            fig.update_yaxes(title_text="Target / Prediction Value")
            st.plotly_chart(fig)

except:
    st.error(
      """
      Oops! Error has happened.
      """
    )

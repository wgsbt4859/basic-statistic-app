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

# ページレイアウト
st.set_page_config(page_title="basic statistics app", layout="wide")


# ページタイトルと概要

st.title("Basic Statistics App")

st.write("""
    ### データ分析をもっと身近に。

    初歩的な探索的データ分析を、クリック操作だけで簡単に実施します。

    Kaggle等で公開されているデータ(csv)の中身をパッと確認したいときに役立ちます。

    今後、主成分分析等にも対応していく予定。
""")


with st.expander("使い方を見る"):

    st.write("""
    ##### 1. データをアップロードします。
    現在対応しているファイルは、csvのみとなります。

    アップロードが完了すると、各種統計量が自動的に計算され、表示されます。

    ##### 2. データの確認方法
    ###### サンプルサイズ / 欠損値
    サンプルサイズ、欠損値は、画面左側のサイドバーにまとめて表示されます。

    ###### 各カラムのデータの分布
    「3. 各データの分布を確認」から、ヒストグラムを確認できます。

    ヒストグラム化されるカラムは、最初は10カラムまでで設定されています。

    10以上の項目の表示も可能です。

    ###### 各カラムのデータの割合
    カテゴリ変数を含むデータセットの場合、その出現回数割合を円グラフで確認できます。

    こちらも、最初は10カラムまでが表示されます。

    10以上の項目の表示も可能です。

    ##### 3. 簡単な重回帰分析
    説明変数と目的変数を選択して、statsmodels.apiを用いた重回帰分析を行うことができます。

    また、重回帰分析結果から、各サンプルの予測値と目標値を見比べることもできます。
    
    ##### 今後の対応予定
    今後、主成分分析等にも対応予定です。

    """)




st.write("#")

# 1. データを読み込む
st.subheader("1. データをアップロードする")
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

        st.sidebar.write("# 基本情報")
        st.sidebar.write(f"### サンプルサイズ:  {df.shape[0]}")
        st.sidebar.write(f"### カラム数      :  {df.shape[1]}")

        st.sidebar.write("##")
        st.sidebar.write("### 欠損値")
        st.sidebar.write("各カラムの欠損値")

        null_df = pd.DataFrame(df.isnull().sum(), columns=["null"])
        st.sidebar.dataframe(null_df)
        
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

        left_column, right_column = st.columns(2)

        left_column.write("##### 分布の確認")
        cols_list = list(df.columns)
        options_hist = left_column.multiselect(
        'ヒストグラムで表示するカラムを選択 (初期値は最初の10カラム)',
        cols_list,
        cols_list[:10])

        ## 検証開始

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
            # fig.update_layout(height=750, width=750, bargap=0.2)
            fig.update_layout(bargap=0.2)
            left_column.plotly_chart(fig, use_container_width=True)

        ## 検証終了

        right_column.write("##### 割合の確認 (カテゴリ変数がある場合のみ)")
        # カテゴリ変数の円グラフ化
        df_obj_cols = list(df.select_dtypes(include='object').columns)

        options_pie = right_column.multiselect(
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
            # fig.update_layout(width=750, height=750)
            right_column.plotly_chart(fig, use_container_width=True)


    # 4. 相関係数の表示
    st.write("#")
    st.subheader("4. 相関係数")
    if uploaded_file is not None:
        fig = px.imshow(df.corr(), text_auto=True)
        # fig.update_layout(width=1200, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # 5. 重回帰分析
    st.write("#")
    st.subheader("5. 重回帰分析")
    if uploaded_file is not None:
        scaler = StandardScaler()
        
        cols_x = list(df.columns)

        left_column, right_column = st.columns(2)
        
        options_multi_reg = left_column.multiselect(
            '説明変数を選択してください。',
            cols_x,
            cols_x
        )
        option_target = left_column.selectbox(
            '目的変数を選択してください。',
            cols_x
        )

        if len(options_multi_reg) > 0 and (left_column.button("重回帰分析 開始")):
            # left_column.write("分析を開始しました。")
            x = df[options_multi_reg]
            y = df[option_target]

            x_scaled = scaler.fit_transform(x)

            multi_OLS = sm.OLS(y, sm.add_constant(x_scaled)) 
            result = multi_OLS.fit()

            left_column.write("分析が終了しました。結果を表示します。")
            right_column.text(result.summary())

            left_column.write(f"自由度調整済決定係数は{result.rsquared_adj:.2f}でした。")
            
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
            left_column.plotly_chart(fig, use_container_width=True)

except:
    st.error(
      """
      Oops! Error has happened.
      """
    )

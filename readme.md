# Basic Statistics App

データ分析をもっと身近に簡単に。
  
## Features
 
初歩的な探索的データ分析を、クリック操作だけで簡単に実施します。
現在対応しているのは、
- 欠損値の確認
- 要約統計量の確認
    - 各データの分布/割合をグラフ化
- 簡単な重回帰分析
までとなります。
 
## Requirements

このアプリはstreamlit上にデプロイしています。

URLは[こちら](https://wgsbt4859-basic-statistic-app-main-y3vp06.streamlit.app/)

使用しているライブラリは以下のとおりです。
streamlit==1.10.0
pandas==1.4.2
numpy==1.21.5
plotly==5.6.0
scikit-learn==1.1.0
statsmodels==0.13.2


## Note

簡単に、欠損値や要約統計量などの確認ができることを目的としています。

将来的に、ダッシュボードのようなものが作れればいいなと思っています。

また、欠損値の補完、主成分分析等も実装していく予定です。

## Author
 
* 作成者: Jun SAKAMOTO
* E-mail: kikoriatashi@gmail.com
# 動かして学ぶディープラーニング（deep-learning-textbook）

三重大学 教育学部 技術・ものづくり教育コース 卒業研究。

動画「Give Me 1 Hour, I'll Make Deep Learning Click Forever」（Zachary Huang 氏）で紹介される
深層学習の主要アーキテクチャを、ブラウザ内で動かせるコードで体験的・視覚的に学べる Web 教材です。
対象は中学生・高校生・大学生など幅広い学生で、プログラミングや高度な数学の知識を前提としません。

- 公開 URL: https://224096-cmd.github.io/deep-learning-textbook/
- 技術: HTML / CSS / JavaScript のみ（外部ライブラリ・サーバ・API 不要。スマートフォンのブラウザでオフライン動作）
- 公開: GitHub Pages（GitHub Actions による自動デプロイ）

## 章構成（13 章）

| 章 | 内容 | 主な体験 |
|---|---|---|
| 1 | パーセプトロン | 学習則で直線が動く様子、XOR が解けないこと |
| 2 | MLP | 隠れ層・活性化関数・誤差逆伝播で曲がった境界を学習 |
| 3 | CNN | 画像を描いてフィルタをかける、プーリング、重み数の比較 |
| 4 | ResNet | 深いネットの層ごとの勾配の大きさ、残差接続の有無で学習比較 |
| 5 | 単語埋め込み | skip-gram を学習し、散布図・類似度・単語の足し算 |
| 6 | RNN | 隠れ状態の推移、時間をさかのぼる勾配、記憶課題の学習 |
| 7 | LSTM | ゲートを手で操作、勾配の流れ比較、RNN との学習レース、学習済みゲートの可視化 |
| 8 | GRU | 更新ゲートの操作、パラメータ数、3 モデルの学習レース |
| 9 | Seq2Seq | エンコーダ・デコーダのアニメ、固定長ベクトルの限界実験 |
| 10 | Attention | Q・K・V をドラッグして重みを体感、翻訳アライメント |
| 11 | Transformer | マルチヘッド自己注意の計算、位置エンコーディング、n² の計算量 |
| 12 | BERT | 穴埋め予測（左だけ vs 両側）、全単語テスト |
| 13 | GPT | 次の文字予測による文章生成、温度・方式、ミニ・スケーリング実験 |

各章は「ねらい（要点・身近なたとえ・使われている場所）→ やってみる（アニメーション・実験）→ コードで動かす（Python 風コードの疑似実行）→ ふりかえり → 小テスト（5 問）」の構成。各章の先頭に元動画の該当時間へのリンクあり。

## フォルダ構成

```
deep-learning-textbook/
├── index.html                 トップページ（章一覧）
├── results.html               学習記録の集計・書き出し
├── chapters/ch01.html … ch13.html   各章の教材ページ
├── assets/
│   ├── css/style.css          共通スタイル
│   └── js/
│       ├── common.js          章ナビ・記録(localStorage)・小テスト・簡易グラフ・ネットワークアニメ・コードパネル
│       ├── content.js         各章の補足（たとえ話・用途・ふりかえり・追加問題・動画時間）
│       ├── seq.js             RNN / LSTM / GRU の順伝播・BPTT・Adam/SGD（第6〜8章）
│       ├── seqwidget.js       記憶課題の学習レース・勾配の流れグラフ
│       └── main.js            トップページの動作チェック
├── .github/workflows/pages.yml   GitHub Pages への自動デプロイ
├── .nojekyll
└── README.md
```

## 結果データの保存（卒論用）

- 操作ログと小テストの結果はブラウザの localStorage に保存され、`results.html` からまとめて CSV / JSON / Markdown で書き出せます（外部送信はしません）。
- 学習者 ID を入力してから書き出すと、複数の学習者のデータを区別できます。

## 更新のしかた

1. ファイルを編集（または zip を展開して上書き）
2. `git add -A` → `git commit -m "メッセージ"` → `git push`
3. 数十秒〜1 分で GitHub Pages に反映（Actions タブで進行状況を確認）

## 参考

- 動画: https://youtu.be/wbc-oxFFV9c （Zachary Huang, "Give Me 1 Hour, I'll Make Deep Learning Click Forever"）
- 本教材のコードはすべて独自実装で、外部データセット・学習済みモデルは使用していません。

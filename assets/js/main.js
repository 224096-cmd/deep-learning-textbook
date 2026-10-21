// 動作チェック用の最小スクリプト
// 「ニューロン1個」の計算（重み付き和 → 活性化関数）を試して、JS が動いていることを確認する

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

document.getElementById("check-btn").addEventListener("click", function () {
  var inputs = [0.5, -1.2, 2.0];
  var weights = [0.8, 0.3, -0.5];
  var bias = 0.1;

  var sum = bias;
  for (var i = 0; i < inputs.length; i++) {
    sum += inputs[i] * weights[i];
  }
  var out = sigmoid(sum);

  var text =
    "入力 x = [" + inputs.join(", ") + "]\n" +
    "重み w = [" + weights.join(", ") + "]\n" +
    "バイアス b = " + bias + "\n" +
    "重み付き和 z = " + sum.toFixed(3) + "\n" +
    "シグモイド σ(z) = " + out.toFixed(4) + "\n\n" +
    "✅ JavaScript が正しく動いています（GitHub Pages 公開 OK）";

  document.getElementById("check-result").textContent = text;
});

/* 記憶課題の「学習レース」ウィジェット（第6〜8章で共用）
   SEQ.race(containerId, chapter, {types:["rnn","lstm"], T:30, H:8, lr:0.01, opt:"adam", steps:1500})
*/
(function () {
  "use strict";
  var NAMES = { rnn: "RNN", lstm: "LSTM", gru: "GRU" };
  var COLORS = { rnn: "#dc2626", lstm: "#2563eb", gru: "#16a34a" };

  SEQ.race = function (containerId, chapter, opts) {
    var $ = DL.$, el = DL.el, c = $(containerId);
    var types = opts.types, records = [], timer = null;
    var id = function (s) { return containerId + "_" + s; };

    c.innerHTML = "";
    var ctl = el("div", { class: "controls" });
    ctl.appendChild(el("label", { html: "覚えておく長さ T <input type='range' id='" + id("T") + "' min='5' max='60' step='5' value='" + (opts.T || 30) + "'> <span class='val' id='" + id("Tv") + "'>" + (opts.T || 30) + "</span>" }));
    ctl.appendChild(el("label", { html: "隠れ状態の大きさ <input type='range' id='" + id("H") + "' min='2' max='16' step='1' value='" + (opts.H || 8) + "'> <span class='val' id='" + id("Hv") + "'>" + (opts.H || 8) + "</span>" }));
    ctl.appendChild(el("label", { html: "最適化：<select id='" + id("opt") + "'><option value='adam'>Adam</option><option value='sgd'>SGD（素朴）</option></select>" }));
    ctl.appendChild(el("label", { html: "学習率 <input type='number' id='" + id("lr") + "' value='" + (opts.lr || 0.01) + "' step='0.005' min='0.0005' style='width:80px'>" }));
    var ctl2 = el("div", { class: "controls" });
    var btn = el("button", { class: "btn btn-small", text: "学習スタート（最大 " + (opts.steps || 1500) + " ステップ）" });
    var stop = el("button", { class: "btn btn-small btn-outline", text: "停止" });
    var st = el("span", { class: "status" });
    ctl2.appendChild(btn); ctl2.appendChild(stop); ctl2.appendChild(st);
    c.appendChild(ctl); c.appendChild(ctl2);
    var row = el("div", { class: "canvas-row" });
    var lossCv = el("canvas", { width: 440, height: 220 }), accCv = el("canvas", { width: 440, height: 220 });
    row.appendChild(el("div", {}, [lossCv, el("p", { class: "canvas-caption", text: "損失（低いほど良い）" })]));
    row.appendChild(el("div", {}, [accCv, el("p", { class: "canvas-caption", text: "正答率（0.5 = 当てずっぽう、1.0 = 完全に記憶できた）" })]));
    c.appendChild(row);
    var tbl = el("table", { class: "data" });
    tbl.innerHTML = "<tr><th>モデル</th><th>T</th><th>隠れ</th><th>パラメータ数</th><th>最適化</th><th>学習率</th><th>解けたステップ</th><th>最終正答率</th><th>最終損失</th></tr>";
    c.appendChild(el("div", { class: "table-wrap" }, [tbl]));
    var exp = el("div", { id: id("exp") }); c.appendChild(exp);
    DL.exportButtons(id("exp"), function () { return records; }, ["model", "T", "hidden", "params", "optimizer", "lr", "solved_step", "final_acc", "final_loss"], "ch" + (chapter < 10 ? "0" : "") + chapter + "_memory_race", chapter);

    $(id("T")).addEventListener("input", function () { $(id("Tv")).textContent = this.value; });
    $(id("H")).addEventListener("input", function () { $(id("Hv")).textContent = this.value; });
    $(id("opt")).addEventListener("change", function () { $(id("lr")).value = this.value === "adam" ? 0.01 : 0.3; });

    function run() {
      if (timer) clearInterval(timer);
      var T = parseInt($(id("T")).value, 10), H = parseInt($(id("H")).value, 10), opt = $(id("opt")).value, lr = parseFloat($(id("lr")).value), maxSteps = opts.steps || 1500;
      var rng = new DL.RNG(12345);
      var models = types.map(function (t) { return { type: t, m: new SEQ.Model(t, H, new DL.RNG(777)), loss: [], acc: [], solved: -1, done: false }; });
      var step = 0, evalSet = SEQ.makeBatch(T, 60, rng);
      btn.disabled = true;
      timer = setInterval(function () {
        var chunk = 5;
        for (var k = 0; k < chunk; k++) {
          var batch = SEQ.makeBatch(T, 8, rng);
          models.forEach(function (mm) {
            if (mm.done) return;
            var r = mm.m.trainStep(batch, lr, opt);
            if (step % 5 === 0) { mm.loss.push(r.loss); }
          });
          step++;
          if (step % 25 === 0) models.forEach(function (mm) {
            if (mm.done) return;
            var a = mm.m.evaluate(evalSet); mm.acc.push(a);
            if (a >= 0.98 && mm.solved < 0) { mm.solved = step; mm.done = true; }
          });
        }
        DL.lineChart(lossCv, models.map(function (mm) { return { name: NAMES[mm.type], values: mm.loss, color: COLORS[mm.type] }; }), { xLabel: "ステップ（×5）", yLabel: "loss", yMin: 0 });
        DL.lineChart(accCv, models.map(function (mm) { return { name: NAMES[mm.type], values: mm.acc, color: COLORS[mm.type] }; }), { xLabel: "ステップ（×25）", yLabel: "正答率", yMin: 0.3, yMax: 1 });
        st.textContent = "ステップ " + step + " ／ " + models.map(function (mm) { return NAMES[mm.type] + (mm.solved > 0 ? "：解けた(" + mm.solved + ")" : "：学習中"); }).join("、");
        if (step >= maxSteps || models.every(function (mm) { return mm.done; })) finish();
      }, 10);
      function finish() {
        clearInterval(timer); timer = null; btn.disabled = false;
        models.forEach(function (mm) {
          var rec = { model: NAMES[mm.type], T: T, hidden: H, params: mm.m.P.count(), optimizer: opt, lr: lr, solved_step: mm.solved > 0 ? mm.solved : "未解決(" + step + ")", final_acc: +mm.m.evaluate(evalSet).toFixed(3), final_loss: +(mm.loss[mm.loss.length - 1] || 0).toFixed(4) };
          records.push(rec); DL.log(chapter, "race", rec);
          var tr = document.createElement("tr");
          tr.innerHTML = ["model", "T", "hidden", "params", "optimizer", "lr", "solved_step", "final_acc", "final_loss"].map(function (k) { return "<td>" + rec[k] + "</td>"; }).join("");
          tbl.appendChild(tr);
        });
        st.textContent += " ／ 終了";
      }
      stop.onclick = finish;
    }
    btn.addEventListener("click", run);
  };

  // 時間方向の勾配の流れ比較グラフ
  SEQ.gradFlowChart = function (cv, types, T, H, seed) {
    var rng = new DL.RNG(seed || 5), xs = SEQ.makeBatch(T, 1, rng)[0].xs;
    var series = types.map(function (t) {
      var m = new SEQ.Model(t, H, new DL.RNG(seed || 5));
      return { name: NAMES[t], values: m.gradThroughTime(xs, 1), color: COLORS[t] };
    });
    DL.lineChart(cv, series, { logY: true, xLabel: "時刻 t（左＝最初の入力）", yLabel: "|∂L/∂h_t|" });
    return series;
  };
  SEQ.NAMES = NAMES; SEQ.COLORS = COLORS;
})();

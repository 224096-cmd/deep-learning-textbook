/* =========================================================
   動かして学ぶディープラーニング 共通ライブラリ
   - 章ナビゲーション
   - 学習記録（localStorage）と CSV / JSON / Markdown 書き出し
   - 小テスト
   - 簡易グラフ（折れ線・ヒートマップ）
   ライブラリ不要、オフラインで動作
   ========================================================= */
var DL = (function () {
  "use strict";

  var CHAPTERS = [
    { no: 1,  file: "ch01.html", title: "パーセプトロン",            en: "Perceptron" },
    { no: 2,  file: "ch02.html", title: "MLP（多層パーセプトロン）",   en: "MLP" },
    { no: 3,  file: "ch03.html", title: "CNN（畳み込み）",            en: "CNN" },
    { no: 4,  file: "ch04.html", title: "ResNet（残差接続）",         en: "ResNet" },
    { no: 5,  file: "ch05.html", title: "単語埋め込み",               en: "Word Embeddings" },
    { no: 6,  file: "ch06.html", title: "RNN（再帰型）",              en: "RNN" },
    { no: 7,  file: "ch07.html", title: "LSTM",                      en: "LSTM" },
    { no: 8,  file: "ch08.html", title: "GRU",                       en: "GRU" },
    { no: 9,  file: "ch09.html", title: "Seq2Seq",                   en: "Seq2Seq" },
    { no: 10, file: "ch10.html", title: "Attention（注意機構）",      en: "Attention" },
    { no: 11, file: "ch11.html", title: "Transformer",               en: "Transformer" },
    { no: 12, file: "ch12.html", title: "BERT",                      en: "BERT" },
    { no: 13, file: "ch13.html", title: "GPT",                       en: "GPT" }
  ];

  var STORAGE_KEY = "dl-textbook-log-v1";

  /* ---------- ユーティリティ ---------- */
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") e.className = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function nowISO() { return new Date().toISOString(); }
  function pad2(n) { return (n < 10 ? "0" : "") + n; }
  function stamp() {
    var d = new Date();
    return d.getFullYear() + pad2(d.getMonth() + 1) + pad2(d.getDate()) + "_" + pad2(d.getHours()) + pad2(d.getMinutes()) + pad2(d.getSeconds());
  }
  function fmt(x, d) { return (typeof x === "number") ? x.toFixed(d === undefined ? 3 : d) : String(x); }

  // 再現性のための疑似乱数（シード付き）
  function RNG(seed) {
    var s = (seed >>> 0) || 123456789;
    this.next = function () {
      s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
    this.gauss = function () {
      var u = 1 - this.next(), v = this.next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    this.int = function (n) { return Math.floor(this.next() * n); };
  }

  function softmax(arr, temp) {
    temp = temp || 1;
    var m = -Infinity; arr.forEach(function (v) { if (v > m) m = v; });
    var ex = arr.map(function (v) { return Math.exp((v - m) / temp); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / s; });
  }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function tanh(x) { return Math.tanh(x); }
  function relu(x) { return x > 0 ? x : 0; }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  /* ---------- 章ナビゲーション ---------- */
  function nav(no) {
    var ch = CHAPTERS[no - 1];
    document.title = "第" + no + "章 " + ch.title + " | 動かして学ぶディープラーニング";
    var header = el("header", { class: "site-header compact" }, [
      el("div", { class: "nav-row" }, [
        el("a", { href: "../index.html", class: "nav-home", text: "← 目次へ" }),
        el("span", { class: "nav-title", text: "第" + no + "章 " + ch.title + " (" + ch.en + ")" }),
        el("a", { href: "../results.html", class: "nav-home", text: "記録を見る" })
      ])
    ]);
    document.body.insertBefore(header, document.body.firstChild);

    var footer = el("footer", { class: "site-footer" }, [
      el("div", { class: "nav-row" }, [
        no > 1 ? el("a", { href: CHAPTERS[no - 2].file, class: "btn btn-outline", text: "← 第" + (no - 1) + "章 " + CHAPTERS[no - 2].title }) : el("span"),
        no < CHAPTERS.length ? el("a", { href: CHAPTERS[no].file, class: "btn", text: "第" + (no + 1) + "章 " + CHAPTERS[no].title + " →" }) : el("a", { href: "../results.html", class: "btn", text: "記録を書き出す →" })
      ]),
      el("p", { class: "muted", text: "動かして学ぶディープラーニング ｜ © 2026 三重大学 教育学部 技術・ものづくり教育コース 卒業研究" })
    ]);
    document.body.appendChild(footer);
    log(no, "open", {});
  }

  /* ---------- 記録 ---------- */
  function loadLog() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch (e) { return []; }
  }
  function saveLog(arr) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(arr)); } catch (e) { /* 容量超過などは無視 */ }
  }
  function log(chapter, event, data) {
    var arr = loadLog();
    arr.push({ time: nowISO(), chapter: chapter, event: event, data: data || {} });
    if (arr.length > 5000) arr = arr.slice(arr.length - 5000);
    saveLog(arr);
  }
  function clearLog() { saveLog([]); }

  /* ---------- 書き出し ---------- */
  function download(text, filename, mime) {
    var blob = new Blob(["﻿" + text], { type: (mime || "text/plain") + ";charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  function csvEscape(v) {
    var s = (v === null || v === undefined) ? "" : String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  // rows: 配列の配列 or オブジェクト配列
  function toCSV(rows, headers) {
    if (!rows.length) return (headers || []).join(",") + "\n";
    if (!Array.isArray(rows[0])) {
      headers = headers || Object.keys(rows[0]);
      rows = rows.map(function (r) { return headers.map(function (h) { return r[h]; }); });
    }
    var out = headers ? [headers.map(csvEscape).join(",")] : [];
    rows.forEach(function (r) { out.push(r.map(csvEscape).join(",")); });
    return out.join("\n") + "\n";
  }
  function toMarkdownTable(rows, headers) {
    if (!rows.length) return "(データなし)\n";
    if (!Array.isArray(rows[0])) {
      headers = headers || Object.keys(rows[0]);
      rows = rows.map(function (r) { return headers.map(function (h) { return r[h]; }); });
    }
    var out = ["| " + headers.join(" | ") + " |", "|" + headers.map(function () { return " --- "; }).join("|") + "|"];
    rows.forEach(function (r) { out.push("| " + r.map(function (v) { return String(v === undefined ? "" : v).replace(/\|/g, "\\|"); }).join(" | ") + " |"); });
    return out.join("\n") + "\n";
  }
  // 章ページ用：結果テーブルに「CSV / JSON / MD」ボタンを付ける
  function exportButtons(containerId, getRows, headers, baseName, chapter) {
    var c = $(containerId);
    if (!c) return;
    c.innerHTML = "";
    c.className = "export-row";
    var mk = function (label, fn) {
      var b = el("button", { class: "btn btn-small", text: label });
      b.addEventListener("click", function () {
        var rows = getRows();
        if (!rows || !rows.length) { alert("まだ書き出せるデータがありません。先に実験を実行してください。"); return; }
        fn(rows);
        log(chapter, "export", { format: label, rows: rows.length });
      });
      return b;
    };
    c.appendChild(el("span", { class: "muted", text: "結果を保存：" }));
    c.appendChild(mk("CSV", function (rows) { download(toCSV(rows, headers), baseName + "_" + stamp() + ".csv", "text/csv"); }));
    c.appendChild(mk("JSON", function (rows) { download(JSON.stringify({ exported: nowISO(), chapter: chapter, headers: headers, rows: rows }, null, 2), baseName + "_" + stamp() + ".json", "application/json"); }));
    c.appendChild(mk("Markdown", function (rows) { download("# " + baseName + "\n\n書き出し: " + nowISO() + "\n\n" + toMarkdownTable(rows, headers), baseName + "_" + stamp() + ".md", "text/markdown"); }));
  }

  /* ---------- 小テスト ---------- */
  // questions: [{q, choices:[...], answer: index, explain}]
  function quiz(containerId, chapter, questions) {
    var c = $(containerId);
    if (!c) return;
    c.innerHTML = "";
    var form = el("div", { class: "quiz" });
    questions.forEach(function (qq, qi) {
      var box = el("div", { class: "quiz-q" }, [el("p", { class: "quiz-text", text: "Q" + (qi + 1) + ". " + qq.q })]);
      qq.choices.forEach(function (ch, ci) {
        var id = containerId + "_q" + qi + "_c" + ci;
        var lab = el("label", { class: "quiz-choice", for: id });
        var inp = el("input", { type: "radio", name: containerId + "_q" + qi, id: id, value: ci });
        lab.appendChild(inp); lab.appendChild(document.createTextNode(" " + ch));
        box.appendChild(lab);
      });
      box.appendChild(el("p", { class: "quiz-explain", id: containerId + "_ex" + qi }));
      form.appendChild(box);
    });
    var btn = el("button", { class: "btn", text: "採点する" });
    var res = el("p", { class: "quiz-result" });
    btn.addEventListener("click", function () {
      var correct = 0, answered = 0, detail = [];
      questions.forEach(function (qq, qi) {
        var sel = form.querySelector("input[name='" + containerId + "_q" + qi + "']:checked");
        var ex = $(containerId + "_ex" + qi);
        var ok = false;
        if (sel) {
          answered++;
          ok = parseInt(sel.value, 10) === qq.answer;
          if (ok) correct++;
        }
        ex.textContent = (sel ? (ok ? "○ 正解。" : "× 不正解。") : "未回答。") + "正解は「" + qq.choices[qq.answer] + "」。" + (qq.explain || "");
        ex.className = "quiz-explain " + (sel ? (ok ? "ok" : "ng") : "");
        detail.push({ q: qi + 1, selected: sel ? parseInt(sel.value, 10) : null, correct: ok });
      });
      res.textContent = "結果：" + correct + " / " + questions.length + " 問 正解（回答 " + answered + " 問）";
      log(chapter, "quiz", { correct: correct, total: questions.length, detail: detail });
    });
    form.appendChild(btn); form.appendChild(res);
    c.appendChild(form);
  }

  /* ---------- 簡易グラフ ---------- */
  function clearCanvas(cv) {
    var ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    return ctx;
  }
  // series: [{name, values:[...], color}] / opts: {yMin,yMax,xLabel,yLabel,logY}
  function lineChart(cv, series, opts) {
    opts = opts || {};
    var ctx = clearCanvas(cv), W = cv.width, H = cv.height;
    var L = 48, R = 12, T = 14, B = 30;
    var n = 0, yMin = Infinity, yMax = -Infinity;
    series.forEach(function (s) {
      n = Math.max(n, s.values.length);
      s.values.forEach(function (v) { if (isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); } });
    });
    if (opts.yMin !== undefined) yMin = opts.yMin;
    if (opts.yMax !== undefined) yMax = opts.yMax;
    if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
    if (yMax === yMin) yMax = yMin + 1;
    var ty = function (v) { return opts.logY ? Math.log10(Math.max(v, 1e-12)) : v; };
    var lo = ty(yMin), hi = ty(yMax);
    var px = function (i) { return L + (W - L - R) * (n <= 1 ? 0 : i / (n - 1)); };
    var py = function (v) { return T + (H - T - B) * (1 - (ty(v) - lo) / (hi - lo)); };
    // 軸
    ctx.strokeStyle = "#9ca3af"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, H - B); ctx.lineTo(W - R, H - B); ctx.stroke();
    ctx.fillStyle = "#6b7280"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    for (var k = 0; k <= 4; k++) {
      var vv = lo + (hi - lo) * k / 4;
      var yy = T + (H - T - B) * (1 - k / 4);
      ctx.fillText(opts.logY ? ("1e" + vv.toFixed(1)) : vv.toFixed(2), L - 4, yy + 4);
      ctx.strokeStyle = "#e5e7eb"; ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(W - R, yy); ctx.stroke();
    }
    ctx.textAlign = "center";
    ctx.fillText(opts.xLabel || "", (L + W - R) / 2, H - 6);
    ctx.save(); ctx.translate(12, (T + H - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(opts.yLabel || "", 0, 0); ctx.restore();
    // 系列
    series.forEach(function (s, si) {
      ctx.strokeStyle = s.color || ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"][si % 5];
      ctx.lineWidth = 2; ctx.beginPath();
      var started = false;
      s.values.forEach(function (v, i) {
        if (!isFinite(v)) return;
        var x = px(i), y = py(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle; ctx.textAlign = "left";
      ctx.fillText(s.name || "", L + 6 + si * 110, T + 10);
    });
  }
  // matrix: 2次元配列（値 0〜1 推奨）, rowLabels/colLabels
  function heatmap(cv, matrix, rowLabels, colLabels, opts) {
    opts = opts || {};
    var ctx = clearCanvas(cv), W = cv.width, H = cv.height;
    var L = opts.left || 60, T = opts.top || 40;
    var rows = matrix.length, cols = matrix[0].length;
    var cw = (W - L - 8) / cols, chh = (H - T - 8) / rows;
    var vmax = opts.vmax, vmin = opts.vmin;
    if (vmax === undefined || vmin === undefined) {
      vmax = -Infinity; vmin = Infinity;
      matrix.forEach(function (r) { r.forEach(function (v) { vmax = Math.max(vmax, v); vmin = Math.min(vmin, v); }); });
      if (vmax === vmin) vmax = vmin + 1;
    }
    ctx.font = "11px sans-serif";
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var t = (matrix[i][j] - vmin) / (vmax - vmin);
        t = Math.max(0, Math.min(1, t));
        ctx.fillStyle = opts.diverging ? divColor(t) : seqColor(t);
        ctx.fillRect(L + j * cw, T + i * chh, cw - 1, chh - 1);
        if (opts.showValues && cw > 28) {
          ctx.fillStyle = t > 0.6 ? "#fff" : "#1f2430"; ctx.textAlign = "center";
          ctx.fillText(matrix[i][j].toFixed(opts.digits === undefined ? 2 : opts.digits), L + j * cw + cw / 2, T + i * chh + chh / 2 + 4);
        }
      }
    }
    ctx.fillStyle = "#374151"; ctx.textAlign = "right";
    (rowLabels || []).forEach(function (lab, i) { ctx.fillText(lab, L - 4, T + i * chh + chh / 2 + 4); });
    ctx.textAlign = "center";
    (colLabels || []).forEach(function (lab, j) {
      ctx.save(); ctx.translate(L + j * cw + cw / 2, T - 6);
      if (cw < 40) ctx.rotate(-Math.PI / 4);
      ctx.fillText(lab, 0, 0); ctx.restore();
    });
  }
  function seqColor(t) { // 白 → 青
    var r = Math.round(255 - 218 * t), g = Math.round(255 - 156 * t), b = Math.round(255 - 20 * t);
    return "rgb(" + r + "," + g + "," + b + ")";
  }
  function divColor(t) { // 赤 ← 白 → 青（t=0.5 が白）
    if (t < 0.5) { var u = 1 - t * 2; return "rgb(255," + Math.round(255 - 180 * u) + "," + Math.round(255 - 200 * u) + ")"; }
    var w = (t - 0.5) * 2; return "rgb(" + Math.round(255 - 218 * w) + "," + Math.round(255 - 156 * w) + ",255)";
  }

  // 棒グラフ（横）: items [{label, value, color}]
  function barChart(cv, items, opts) {
    opts = opts || {};
    var ctx = clearCanvas(cv), W = cv.width, H = cv.height;
    var L = opts.left || 90, R = 50;
    var vmax = opts.vmax || Math.max.apply(null, items.map(function (i) { return i.value; })) || 1;
    var bh = (H - 10) / items.length;
    ctx.font = "12px sans-serif";
    items.forEach(function (it, i) {
      var w = (W - L - R) * Math.max(0, it.value) / vmax;
      ctx.fillStyle = it.color || "#2563eb";
      ctx.fillRect(L, 5 + i * bh + 3, w, bh - 6);
      ctx.fillStyle = "#374151"; ctx.textAlign = "right";
      ctx.fillText(it.label, L - 6, 5 + i * bh + bh / 2 + 4);
      ctx.textAlign = "left";
      ctx.fillText(opts.format ? opts.format(it.value) : it.value.toFixed(3), L + w + 4, 5 + i * bh + bh / 2 + 4);
    });
  }

  return {
    CHAPTERS: CHAPTERS, $: $, el: el, RNG: RNG, softmax: softmax, sigmoid: sigmoid, tanh: tanh, relu: relu, dot: dot, fmt: fmt,
    nav: nav, log: log, loadLog: loadLog, clearLog: clearLog,
    download: download, toCSV: toCSV, toMarkdownTable: toMarkdownTable, exportButtons: exportButtons, stamp: stamp,
    quiz: quiz, lineChart: lineChart, heatmap: heatmap, barChart: barChart
  };
})();

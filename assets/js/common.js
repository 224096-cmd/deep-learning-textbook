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
    if (tag === "canvas" && typeof hidpi === "function") hidpi(e);
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

  /* ---------- 高解像度描画（ぼやけ防止） ----------
     canvas の内部解像度を devicePixelRatio 倍にし、描画座標はこれまでどおり（論理サイズ）で扱えるようにする */
  var DPR = Math.min(3, Math.max(1, (window.devicePixelRatio || 1)));
  function hidpi(cv) {
    if (cv.__hidpi || !(cv instanceof HTMLCanvasElement)) return cv;
    var W = cv.width, H = cv.height; cv.__hidpi = true;
    var apply = function () {
      HTMLCanvasElement.prototype.__lookupSetter__("width").call(cv, Math.round(W * DPR));
      HTMLCanvasElement.prototype.__lookupSetter__("height").call(cv, Math.round(H * DPR));
      cv.style.aspectRatio = W + " / " + H;
      if (!cv.style.width) cv.style.width = "100%"; cv.style.maxWidth = "min(100%, " + Math.round(W * 1.25) + "px)";
    };
    apply();
    Object.defineProperty(cv, "width", { get: function () { return W; }, set: function (v) { W = v; apply(); }, configurable: true });
    Object.defineProperty(cv, "height", { get: function () { return H; }, set: function (v) { H = v; apply(); }, configurable: true });
    var orig = cv.getContext.bind(cv);
    cv.getContext = function (type, opts) {
      var c = orig(type, opts);
      if (c && type === "2d" && !c.__scaled) {
        c.__scaled = true; c.setTransform(DPR, 0, 0, DPR, 0, 0);
        // putImageData は変換を無視するので、一度別キャンバスに置いてから drawImage で拡大する
        c.putImageData = function (img, dx, dy) {
          var tmp = document.createElement("canvas"); tmp.width = img.width; tmp.height = img.height;
          tmp.getContext("2d").putImageData(img, 0, 0); c.drawImage(tmp, dx, dy, img.width, img.height);
        };
      }
      return c;
    };
    return cv;
  }
  Array.prototype.forEach.call(document.querySelectorAll("canvas"), hidpi);

  /* ---------- 章ナビゲーション ---------- */
  function nav(no) {
    var ch = CHAPTERS[no - 1];
    document.title = "第" + no + "章 " + ch.title + " | 動かして学ぶディープラーニング";
    var dots = el("div", { class: "progress-dots", title: "第" + no + "章 / 全" + CHAPTERS.length + "章" });
    CHAPTERS.forEach(function (c) {
      var d = el("a", { href: c.file, class: "dot" + (c.no === no ? " current" : c.no < no ? " done" : ""), title: "第" + c.no + "章 " + c.title });
      dots.appendChild(d);
    });
    var header = el("header", { class: "site-header compact sticky" }, [
      el("div", { class: "nav-row" }, [
        el("a", { href: "../index.html", class: "nav-home", text: "← 目次" }),
        el("span", { class: "nav-title", text: "第" + no + "章 " + ch.title }),
        el("span", { class: "nav-right" }, [dots, el("a", { href: "../glossary.html", class: "nav-home nav-small", text: "用語集" })])
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

    // ---- 補足コンテンツの差し込み（content.js） ----
    var C = (typeof DL_CONTENT !== "undefined") ? DL_CONTENT[no] : null;
    var main = document.querySelector("main");
    if (C && main) {
      var hero = el("section", { class: "chapter-hero" }, [
        el("div", { class: "hero-num", text: String(no) }),
        el("div", { class: "hero-text" }, [
          el("div", { class: "hero-title", text: ch.title + " " }, [el("span", { class: "hero-en", text: ch.en })]),
          el("p", { class: "hero-tagline", text: C.tagline })
        ]),
        el("div", { class: "hero-meta" }, [
          el("span", { class: "chip", text: "約 " + C.minutes + " 分" })
        ])
      ]);
      main.insertBefore(hero, main.firstChild);
      var goal = main.querySelector(".goal");
      if (goal) {
        var extra = el("div", { class: "extra-grid" }, [
          el("div", { class: "box box-analogy" }, [el("h4", { text: "身近なたとえ" }), el("p", { text: C.analogy })]),
          el("div", { class: "box box-uses" }, [el("h4", { text: "どこで使われている？" }), el("ul", {}, C.uses.map(function (u) { return el("li", { text: u }); }))])
        ]);
        goal.parentNode.insertBefore(extra, goal.nextSibling);
      }
      if (typeof DL_DIAGRAMS !== "undefined" && DL_DIAGRAMS[no]) {
        var firstCard = main.querySelector("section.card");
        var dcard = el("section", { class: "card" }, [el("h2", { text: "しくみの図（部品をクリックすると説明が出ます）" }), el("div", { id: "arch" })]);
        if (firstCard) firstCard.parentNode.insertBefore(dcard, firstCard.nextSibling); else main.appendChild(dcard);
        DL.archApi = arch("arch", DL_DIAGRAMS[no]);
      }
      var quizCard = main.querySelector("[id^=quiz]"); if (quizCard) quizCard = quizCard.closest("section");
      var wrap = el("section", { class: "card card-review" }, [
        el("h2", { text: "ふりかえり：この章の 3 つのポイント" }),
        el("ol", { class: "takeaways" }, C.takeaways.map(function (t) { return el("li", { text: t }); })),
        C.deeper ? el("details", { class: "deeper" }, [el("summary", { text: "発展：もっと知りたい人へ" }), el("ul", {}, C.deeper.map(function (t) { return el("li", { text: t }); }))]) : el("span"),
        C.refs ? el("div", { class: "refs" }, [el("div", { class: "refs-title", text: "参考文献・さらに読む" }), el("ul", {}, C.refs.map(function (r) { return el("li", {}, [el("a", { href: r.u, target: "_blank", rel: "noopener", text: r.t }), r.n ? el("span", { class: "muted", text: "　" + r.n }) : el("span")]); }))]) : el("span"),
        el("p", { class: "muted", text: "次の小テストで確かめましょう。" })
      ]);
      if (quizCard) main.insertBefore(wrap, quizCard); else main.appendChild(wrap);
      // セクション見出しにタグを付ける
      Array.prototype.forEach.call(main.querySelectorAll(".card > h2"), function (h) {
        var t = h.textContent, tag = /^体験|^まず/.test(t) ? "やってみる" : /^コード/.test(t) ? "コード" : /^小テスト/.test(t) ? "テスト" : /^ふりかえり/.test(t) ? "まとめ" : /^この章/.test(t) ? "ねらい" : /^記録/.test(t) ? "記録" : null;
        if (tag) h.insertBefore(el("span", { class: "sec-tag tag-" + tag, text: tag }), h.firstChild);
      });
    }
    polishUI(main || document);
    log(no, "open", {});
  }

  /* ---------- 体験部分の UI を整える ----------
     ・.controls → ツールバー（ラベル付きの入力フィールド、主ボタン／副ボタン）
     ・canvas + キャプション → 枠付きの図
     ・.status → バッジ */
  function polishUI(root) {
    Array.prototype.forEach.call(root.querySelectorAll(".controls"), function (ctl) {
      if (ctl.classList.contains("toolbar")) return;
      ctl.classList.add("toolbar");
      Array.prototype.forEach.call(ctl.querySelectorAll("label"), function (lab) {
        var input = lab.querySelector("input, select"); if (!input) return;
        var isCheck = input.type === "checkbox";
        if (isCheck) { lab.classList.add("field-check"); return; }
        // 先頭のテキストノードをラベル名に
        var texts = []; Array.prototype.forEach.call(lab.childNodes, function (n) { if (n.nodeType === 3 && n.textContent.trim()) texts.push(n); });
        var name = texts.length ? texts[0].textContent.replace(/[：:]\s*$/, "").trim() : "";
        texts.forEach(function (n) { lab.removeChild(n); });
        var val = lab.querySelector(".val");
        var row = el("div", { class: "field-row" });
        while (lab.firstChild) row.appendChild(lab.firstChild);
        lab.appendChild(el("span", { class: "field-name", text: name })); lab.appendChild(row);
        lab.classList.add("field"); if (input.type === "range") lab.classList.add("field-range");
        if (val) { row.removeChild(val); lab.querySelector(".field-name").appendChild(val); }
      });
      Array.prototype.forEach.call(ctl.querySelectorAll("button.btn"), function (b) {
        var t = b.textContent;
        if (/リセット|初期化|戻す|消去|再計算|配置し直す|入れ替える|乱数/.test(t) && !/開始|学習|実行|再生/.test(t)) b.classList.add("btn-ghost");
        else b.classList.add("btn-primary");
      });
    });
    // 図を枠で囲む
    Array.prototype.forEach.call(root.querySelectorAll("canvas"), function (cv) {
      if (cv.closest(".figure") || cv.closest(".codepanel")) return;
      var cap = cv.nextElementSibling && cv.nextElementSibling.classList.contains("canvas-caption") ? cv.nextElementSibling : null;
      var titleEl = cv.previousElementSibling && cv.previousElementSibling.tagName === "P" && cv.previousElementSibling.querySelector("b") && cv.parentNode.classList.contains("canvas-row") === false && cv.parentNode.parentNode.classList.contains("canvas-row") ? cv.previousElementSibling : null;
      var fig = el("div", { class: "figure" });
      cv.parentNode.insertBefore(fig, cv);
      if (titleEl) { titleEl.classList.add("figure-title"); fig.appendChild(titleEl); }
      fig.appendChild(cv);
      if (cap) { cap.classList.add("figure-caption"); fig.appendChild(cap); }
    });
    Array.prototype.forEach.call(root.querySelectorAll(".status"), function (st) { st.classList.add("badge"); });
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
    if (typeof DL_CONTENT !== "undefined" && DL_CONTENT[chapter] && DL_CONTENT[chapter].quiz) questions = questions.concat(DL_CONTENT[chapter].quiz);
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
      ctx.fillStyle = ctx.strokeStyle;
      if (s.values.length <= 40) s.values.forEach(function (v, i) { if (isFinite(v)) { ctx.beginPath(); ctx.arc(px(i), py(v), 3, 0, 7); ctx.fill(); } });
      ctx.textAlign = "left";
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

  /* ---------- 信号が流れるネットワーク図（アニメーション） ----------
     DL.netAnim(canvas, { layers: [2, 4, 1], labels: ["入力", "隠れ層", "出力"], skip: false })
       .start()  : 順伝播（青い光）→ 逆伝播（赤い光）を繰り返し再生
       .pulse(dir): 1 回だけ再生（"fwd" / "bwd" / "both"）
       .setActs(arrays): ノードの色を活性値で塗る
  */
  function netAnim(cv, opts) {
    var ctx = cv.getContext("2d"), layers = opts.layers, labels = opts.labels || [], skip = !!opts.skip, recurrent = !!opts.recurrent;
    var W = cv.width, H = cv.height, padX = 60, padY = 36;
    var pos = layers.map(function (n, l) {
      var x = padX + (W - 2 * padX) * (layers.length === 1 ? 0.5 : l / (layers.length - 1));
      return Array.from({ length: n }, function (_, i) { return [x, padY + (H - 2 * padY) * (n === 1 ? 0.5 : i / (n - 1))]; });
    });
    var acts = null, t = 0, dir = "fwd", playing = false, loop = false, raf = null, speed = opts.speed || 1, queue = [];
    function drawStatic() {
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1;
      for (var l = 0; l < layers.length - 1; l++) {
        pos[l].forEach(function (a) { pos[l + 1].forEach(function (b) { ctx.strokeStyle = "#d1d5db"; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }); });
        if (skip && l > 0 && l < layers.length - 1 && layers[l] === layers[l + 1]) {
          ctx.strokeStyle = "#fbbf24"; ctx.setLineDash([4, 4]); ctx.beginPath();
          ctx.moveTo(pos[l][0][0], pos[l][0][1] - 18); ctx.quadraticCurveTo((pos[l][0][0] + pos[l + 1][0][0]) / 2, pos[l][0][1] - 36, pos[l + 1][0][0], pos[l + 1][0][1] - 18); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      if (recurrent) pos.forEach(function (layer) { layer.forEach(function (a) { ctx.strokeStyle = "#a78bfa"; ctx.beginPath(); ctx.arc(a[0], a[1] - 18, 10, 0.3 * Math.PI, 0.7 * Math.PI, true); ctx.stroke(); }); });
      pos.forEach(function (layer, l) {
        layer.forEach(function (a, i) {
          var v = acts && acts[l] ? acts[l][i] : null;
          ctx.fillStyle = v === null || v === undefined ? "#fff" : (v >= 0 ? "rgba(37,99,235," + Math.min(1, Math.abs(v)) * 0.85 + ")" : "rgba(220,38,38," + Math.min(1, Math.abs(v)) * 0.85 + ")");
          ctx.beginPath(); ctx.arc(a[0], a[1], 12, 0, 7); ctx.fill(); ctx.strokeStyle = "#374151"; ctx.lineWidth = 1.5; ctx.stroke();
          if (v !== null && v !== undefined) { ctx.fillStyle = Math.abs(v) > 0.6 ? "#fff" : "#1f2430"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(v.toFixed(1), a[0], a[1] + 3.5); }
        });
        if (labels[l]) { ctx.fillStyle = "#6b7280"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.fillText(labels[l], layer[0][0], H - 8); }
      });
    }
    function drawPulse() {
      // t ∈ [0, layers.length-1)：どの層間を光が進んでいるか
      var fwd = dir === "fwd", seg = Math.floor(t), frac = t - seg;
      var l = fwd ? seg : layers.length - 2 - seg; if (l < 0 || l >= layers.length - 1) return;
      var color = fwd ? "#2563eb" : "#dc2626";
      pos[l].forEach(function (a) { pos[l + 1].forEach(function (b) {
        var f = fwd ? frac : 1 - frac, x = a[0] + (b[0] - a[0]) * f, y = a[1] + (b[1] - a[1]) * f;
        ctx.strokeStyle = fwd ? "rgba(37,99,235,0.35)" : "rgba(220,38,38,0.35)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(x, y); ctx.stroke();
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fill();
      }); });
      if (skip && l > 0 && l < layers.length - 1 && layers[l] === layers[l + 1]) {
        var f2 = fwd ? frac : 1 - frac, a0 = pos[l][0], b0 = pos[l + 1][0], mx = (a0[0] + b0[0]) / 2;
        var qx = (1 - f2) * (1 - f2) * a0[0] + 2 * (1 - f2) * f2 * mx + f2 * f2 * b0[0], qy = (1 - f2) * (1 - f2) * (a0[1] - 18) + 2 * (1 - f2) * f2 * (a0[1] - 36) + f2 * f2 * (b0[1] - 18);
        ctx.fillStyle = "#f59e0b"; ctx.beginPath(); ctx.arc(qx, qy, 5, 0, 7); ctx.fill();
      }
      ctx.fillStyle = color; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(fwd ? "→ 順伝播：入力から予測へ" : "← 逆伝播：誤差から各重みへ", 8, 16);
    }
    function frame() {
      drawStatic(); if (playing) drawPulse();
      if (playing) {
        t += 0.02 * speed;
        if (t >= layers.length - 1) {
          t = 0;
          if (queue.length) dir = queue.shift();
          else if (loop) dir = dir === "fwd" ? "bwd" : "fwd";
          else { playing = false; }
        }
      }
      raf = requestAnimationFrame(frame);
    }
    var api = {
      start: function () { loop = true; playing = true; dir = "fwd"; t = 0; return api; },
      stop: function () { loop = false; playing = false; return api; },
      pulse: function (d) { loop = false; t = 0; if (d === "both") { dir = "fwd"; queue = ["bwd"]; } else { dir = d || "fwd"; queue = []; } playing = true; return api; },
      setActs: function (a) { acts = a; return api; },
      isPlaying: function () { return playing; }
    };
    frame();
    return api;
  }

  /* ---------- コードパネル（Python 風コードを「疑似実行」する） ----------
     DL.codePanel("id", {
       title: "...", 
       code: ["x = {{n}}", "print(x * 2)"],         // {{名前}} が編集できる入力欄になる
       params: { n: { value: 3, min: 1, max: 10, step: 1 } },   // type: number(既定) / text / select(options)
       run: function (p, out) { out.print("..."); }  // out.print で printed 欄に出力。Promise を返してもよい
     })
  */
  function codePanel(containerId, spec) {
    var c = $(containerId); if (!c) return null;
    c.innerHTML = ""; c.className = "codepanel";
    var head = el("div", { class: "code-head" }, [
      el("span", { class: "code-title", text: spec.title || "コードで動かす" }),
      el("span", { class: "code-hint", text: "枠の中の値を書き換えて ▶ 実行" })
    ]);
    var body = el("div", { class: "code-body" });
    var inputs = {};
    var KW = /\b(def|for|in|if|else|return|import|from|as|print|range|while|and|or|not|True|False|None|lambda)\b/g;
    function renderLine(line, idx) {
      var row = el("div", { class: "code-line" }, [el("span", { class: "code-ln", text: String(idx + 1) })]);
      var content = el("span", { class: "code-content" });
      var parts = line.split(/(\{\{[a-zA-Z0-9_]+\}\})/);
      parts.forEach(function (part) {
        var m = part.match(/^\{\{([a-zA-Z0-9_]+)\}\}$/);
        if (m) {
          var name = m[1], ps = spec.params[name] || {};
          var inp;
          if (ps.type === "select") {
            inp = el("select", { class: "code-input" });
            (ps.options || []).forEach(function (o) { inp.appendChild(el("option", { value: o, text: o })); });
            inp.value = ps.value;
          } else {
            inp = el("input", { class: "code-input", type: ps.type === "text" ? "text" : "number", value: ps.value });
            if (ps.type !== "text") { if (ps.min !== undefined) inp.min = ps.min; if (ps.max !== undefined) inp.max = ps.max; inp.step = ps.step || "any"; }
            inp.style.width = ps.type === "text" ? ((String(ps.value).length + 4) * 0.62 + "em") : (Math.max(3, String(ps.value).length + 2) * 0.72 + "em");
            if (ps.type === "text") inp.addEventListener("input", function () { inp.style.width = (inp.value.length + 4) * 0.62 + "em"; });
          }
          inp.title = ps.label || name;
          inputs[name] = inp;
          content.appendChild(inp);
        } else {
          // コメント・文字列・キーワードの簡易色付け
          var ci = part.indexOf("#"), codeText = ci >= 0 ? part.slice(0, ci) : part, comment = ci >= 0 ? part.slice(ci) : "";
          var html = codeText.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/("[^"]*")/g, '<span class="c-str">$1</span>').replace(KW, '<span class="c-kw">$1</span>').replace(/\b(\d+\.?\d*)\b/g, '<span class="c-num">$1</span>');
          var sp = el("span", { html: html }); content.appendChild(sp);
          if (comment) content.appendChild(el("span", { class: "c-com", text: comment }));
        }
      });
      row.appendChild(content);
      return row;
    }
    var lineEls = spec.code.map(function (ln, i) { var r = renderLine(ln, i); body.appendChild(r); return r; });
    var runBtn = el("button", { class: "btn btn-small code-run", text: "▶ 実行" });
    var resetBtn = el("button", { class: "btn btn-small btn-outline", text: "値を戻す" });
    var figLink = el("a", { href: "#", class: "code-fig", text: "↑ 図を見る" });
    figLink.addEventListener("click", function (e) { e.preventDefault(); var cv = document.querySelector("main canvas"); if (cv) cv.scrollIntoView({ behavior: "smooth", block: "center" }); });
    var bar = el("div", { class: "code-bar" }, [runBtn, resetBtn, el("span", { class: "code-status" }), figLink]);
    var out = el("div", { class: "code-out" }, [el("span", { class: "code-out-label", text: "printed" }), el("pre", { class: "code-out-text", text: "（▶ 実行 を押すと、ここに結果が表示されます）" })]);
    c.appendChild(head); c.appendChild(body); c.appendChild(bar); c.appendChild(out);
    var pre = out.querySelector("pre"), status = bar.querySelector(".code-status");
    function values() {
      var v = {};
      Object.keys(spec.params).forEach(function (k) { var ps = spec.params[k], raw = inputs[k] ? inputs[k].value : ps.value; v[k] = (ps.type === "text" || ps.type === "select") ? raw : parseFloat(raw); if (ps.type !== "text" && ps.type !== "select" && !isFinite(v[k])) v[k] = ps.value; });
      return v;
    }
    var running = false;
    function execute() {
      if (running) return; running = true; runBtn.disabled = true; status.textContent = "実行中…";
      pre.textContent = ""; var lines = [];
      var printer = { print: function (t) { lines.push(String(t)); pre.textContent = lines.join("\n"); }, clear: function () { lines = []; pre.textContent = ""; } };
      var i = 0;
      var hl = setInterval(function () {
        lineEls.forEach(function (r, j) { r.classList.toggle("active", j === i); });
        i++;
        if (i >= lineEls.length) {
          clearInterval(hl);
          var done = function () { lineEls.forEach(function (r) { r.classList.remove("active"); }); running = false; runBtn.disabled = false; status.textContent = "完了"; log(spec.chapter || 0, "code_run", values()); };
          try { var ret = spec.run(values(), printer); if (ret && typeof ret.then === "function") ret.then(done, function (e) { printer.print("エラー: " + e); done(); }); else done(); }
          catch (e) { printer.print("エラー: " + e.message); done(); }
        }
      }, Math.max(40, Math.min(140, 900 / lineEls.length)));
    }
    runBtn.addEventListener("click", execute);
    resetBtn.addEventListener("click", function () { Object.keys(inputs).forEach(function (k) { inputs[k].value = spec.params[k].value; }); });
    Object.keys(inputs).forEach(function (k) { inputs[k].addEventListener("keydown", function (e) { if (e.key === "Enter") execute(); }); });
    return { run: execute, values: values, set: function (k, v) { if (inputs[k]) inputs[k].value = v; } };
  }

  /* ---------- しくみの図（SVG・クリックで説明） ----------
     DL.arch("id", { w, h, intro, nodes:[{id,x,y,w,h,label,sub,color,shape,desc}], edges:[{from,to,label,color,dashed,bend,path}] })
     color: blue | green | amber | rose | violet | gray | dark      shape: box(既定) | circle | op（小さな丸：⊕ × σ など）
     戻り値 api.set(id, {fill: 0〜1, sub: "..."}) でノードの色の濃さ・サブ文字を後から変えられる */
  var ARCH_COLORS = { blue: ["#dbeafe", "#2563eb"], green: ["#d1fae5", "#059669"], amber: ["#fef3c7", "#d97706"], rose: ["#ffe4e6", "#e11d48"], violet: ["#ede9fe", "#7c3aed"], gray: ["#f1f5f9", "#64748b"], dark: ["#1e293b", "#0f172a"] };
  function arch(containerId, spec) {
    var c = $(containerId); if (!c) return null;
    var NS = "http://www.w3.org/2000/svg";
    function sv(tag, attrs, text) { var e = document.createElementNS(NS, tag); Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); }); if (text !== undefined) e.textContent = text; return e; }
    c.innerHTML = ""; c.classList.add("arch");
    var svg = sv("svg", { viewBox: "0 0 " + spec.w + " " + spec.h, class: "arch-svg", role: "img" });
    var defs = sv("defs");
    Object.keys(ARCH_COLORS).forEach(function (k) {
      var m = sv("marker", { id: containerId + "-arrow-" + k, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "11", markerHeight: "11", markerUnits: "userSpaceOnUse", orient: "auto-start-reverse" });
      m.appendChild(sv("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: ARCH_COLORS[k][1] })); defs.appendChild(m);
    });
    svg.appendChild(defs);
    var byId = {}; spec.nodes.forEach(function (n) { byId[n.id] = n; n.w = n.w || (n.shape === "op" ? 28 : n.shape === "circle" ? 44 : 120); n.h = n.h || (n.shape === "op" ? 28 : n.shape === "circle" ? 44 : 50); });
    function center(n) { return [n.x + n.w / 2, n.y + n.h / 2]; }
    function port(n, toward) {
      var cx = center(n), dx = toward[0] - cx[0], dy = toward[1] - cx[1];
      if (n.shape === "circle" || n.shape === "op") { var r = n.w / 2, L = Math.hypot(dx, dy) || 1; return [cx[0] + dx / L * r, cx[1] + dy / L * r]; }
      if (Math.abs(dx) * n.h > Math.abs(dy) * n.w) return [dx > 0 ? n.x + n.w : n.x, cx[1]];
      return [cx[0], dy > 0 ? n.y + n.h : n.y];
    }
    var gE = sv("g"), gN = sv("g"); svg.appendChild(gE); svg.appendChild(gN);
    (spec.edges || []).forEach(function (e) {
      var col = e.color || "gray", stroke = ARCH_COLORS[col][1], d;
      if (e.path) d = e.path;
      else {
        var a = byId[e.from], b = byId[e.to], ca = center(a), cb = center(b), p1 = port(a, cb), p2 = port(b, ca);
        if (e.bend) { var mx = (p1[0] + p2[0]) / 2, my = (p1[1] + p2[1]) / 2 + e.bend; p1 = port(a, [mx, my]); p2 = port(b, [mx, my]); d = "M" + p1[0] + " " + p1[1] + " Q" + mx + " " + my + " " + p2[0] + " " + p2[1]; }
        else d = "M" + p1[0] + " " + p1[1] + " L" + p2[0] + " " + p2[1];
      }
      var path = sv("path", { d: d, fill: "none", stroke: stroke, "stroke-width": e.width || 2, "marker-end": e.noArrow ? "" : "url(#" + containerId + "-arrow-" + col + ")" });
      if (e.dashed) path.setAttribute("stroke-dasharray", "6 4");
      gE.appendChild(path);
      if (e.label) {
        var lab = sv("text", { class: "arch-edge-label", fill: stroke }, e.label), lx, ly;
        if (e.labelAt) { lx = e.labelAt[0]; ly = e.labelAt[1]; }
        else if (e.bend) { lx = ((p1[0] + p2[0]) / 2 + mx) / 2; ly = ((p1[1] + p2[1]) / 2 + my) / 2 + (e.bend < 0 ? -6 : 14); }
        else { var dx0 = p2[0] - p1[0], dy0 = p2[1] - p1[1], L0 = Math.hypot(dx0, dy0) || 1; lx = (p1[0] + p2[0]) / 2 - dy0 / L0 * 10; ly = (p1[1] + p2[1]) / 2 + dx0 / L0 * 10 - 2; }
        lab.setAttribute("x", lx); lab.setAttribute("y", ly); lab.setAttribute("text-anchor", "middle"); gE.appendChild(lab);
      }
    });
    var desc = el("div", { class: "arch-desc" }, [el("p", { class: "arch-desc-text", text: spec.intro || "図の部品をクリックすると説明が出ます。" })]);
    var nodeEls = {};
    spec.nodes.forEach(function (n) {
      var col = ARCH_COLORS[n.color || "blue"], g = sv("g", { class: "arch-node" + (n.desc ? " clickable" : "") });
      var shape;
      if (n.shape === "circle" || n.shape === "op") shape = sv("circle", { cx: n.x + n.w / 2, cy: n.y + n.h / 2, r: n.w / 2, fill: col[0], stroke: col[1], "stroke-width": 2 });
      else shape = sv("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 10, fill: col[0], stroke: col[1], "stroke-width": 2 });
      g.appendChild(shape);
      var cx = n.x + n.w / 2, cy = n.y + n.h / 2, textColor = (n.color === "dark") ? "#fff" : "#1e293b";
      var lines = String(n.label).split("\n"), subEl = null;
      var base = cy - (lines.length - 1) * 8 + (n.sub ? -7 : 0) + 5;
      lines.forEach(function (ln, i) { g.appendChild(sv("text", { x: cx, y: base + i * 16, "text-anchor": "middle", class: "arch-label" + (n.shape === "op" ? " op" : ""), fill: textColor }, ln)); });
      if (n.sub) { subEl = sv("text", { x: cx, y: base + (lines.length - 1) * 16 + 16, "text-anchor": "middle", class: "arch-sub", fill: n.color === "dark" ? "#cbd5e1" : "#475569" }, n.sub); g.appendChild(subEl); }
      if (n.note) g.appendChild(sv("text", { x: cx, y: n.y + n.h + 16, "text-anchor": "middle", class: "arch-sub", fill: "#64748b" }, n.note));
      if (n.desc) {
        g.addEventListener("click", function () {
          Array.prototype.forEach.call(gN.querySelectorAll(".arch-node"), function (x) { x.classList.remove("selected"); });
          g.classList.add("selected");
          desc.innerHTML = ""; desc.appendChild(el("p", { class: "arch-desc-title", text: n.label.replace(/\n/g, " ") })); desc.appendChild(el("p", { class: "arch-desc-text", text: n.desc }));
          log(0, "arch_click", { diagram: containerId, node: n.id });
        });
      }
      gN.appendChild(g); nodeEls[n.id] = { g: g, shape: shape, sub: subEl, col: col };
    });
    c.appendChild(svg); c.appendChild(desc);
    return {
      set: function (id, o) {
        var ne = nodeEls[id]; if (!ne) return;
        if (o.fill !== undefined) {
          var t = Math.max(0, Math.min(1, o.fill)); ne.shape.setAttribute("fill", mixColor("#ffffff", ne.col[1], 0.12 + 0.75 * t));
          Array.prototype.forEach.call(ne.g.querySelectorAll("text"), function (tx) { tx.setAttribute("fill", t > 0.55 ? "#fff" : (tx.classList.contains("arch-sub") ? "#475569" : "#1e293b")); });
        }
        if (o.sub !== undefined && ne.sub) ne.sub.textContent = o.sub;
      }
    };
  }
  function mixColor(h1, h2, t) {
    var a = [1, 3, 5].map(function (i) { return parseInt(h1.slice(i, i + 2), 16); }), b = [1, 3, 5].map(function (i) { return parseInt(h2.slice(i, i + 2), 16); });
    return "rgb(" + a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); }).join(",") + ")";
  }

  // 凡例を描く（右上）
  function legend(cv, items) {
    var ctx = cv.getContext("2d"), w = 0; ctx.font = "11px sans-serif";
    items.forEach(function (it) { w = Math.max(w, ctx.measureText(it.label).width); });
    var bw = w + 30, bh = items.length * 16 + 8, x = cv.width - bw - 8, y = 8;
    ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillRect(x, y, bw, bh); ctx.strokeStyle = "#d1d5db"; ctx.strokeRect(x, y, bw, bh);
    items.forEach(function (it, i) { ctx.fillStyle = it.color; ctx.beginPath(); ctx.arc(x + 12, y + 12 + i * 16, 5, 0, 7); ctx.fill(); ctx.fillStyle = "#1f2430"; ctx.textAlign = "left"; ctx.fillText(it.label, x + 22, y + 16 + i * 16); });
  }

  return {
    netAnim: netAnim, codePanel: codePanel, legend: legend, arch: arch,
    CHAPTERS: CHAPTERS, $: $, el: el, RNG: RNG, softmax: softmax, sigmoid: sigmoid, tanh: tanh, relu: relu, dot: dot, fmt: fmt,
    nav: nav, log: log, loadLog: loadLog, clearLog: clearLog,
    download: download, toCSV: toCSV, toMarkdownTable: toMarkdownTable, exportButtons: exportButtons, stamp: stamp,
    quiz: quiz, lineChart: lineChart, heatmap: heatmap, barChart: barChart
  };
})();

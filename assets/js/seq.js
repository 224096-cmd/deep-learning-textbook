/* =========================================================
   小さな系列モデル（RNN / LSTM / GRU）を JavaScript だけで学習する
   課題：「最初の 1 ビットを、T ステップの雑音の後まで覚えておく」
   - 入力 x_t = [ビット, 先頭マーカー]（2 次元）
   - 最後のステップで先頭ビット（0/1）を答える
   学習：BPTT（時間方向の誤差逆伝播）＋ Adam ＋ 勾配クリッピング
   ========================================================= */
var SEQ = (function () {
  "use strict";

  /* ---------- 行列ユーティリティ ---------- */
  function zeros(r, c) { var m = []; for (var i = 0; i < r; i++) { var row = []; for (var j = 0; j < c; j++) row.push(0); m.push(row); } return m; }
  function randn(r, c, s, rng) { var m = zeros(r, c); for (var i = 0; i < r; i++) for (var j = 0; j < c; j++) m[i][j] = rng.gauss() * s; return m; }
  function matvec(W, v) { var out = new Array(W.length); for (var i = 0; i < W.length; i++) { var s = 0, row = W[i]; for (var j = 0; j < v.length; j++) s += row[j] * v[j]; out[i] = s; } return out; }
  function matTvec(W, v) { var out = new Array(W[0].length).fill(0); for (var i = 0; i < W.length; i++) for (var j = 0; j < W[0].length; j++) out[j] += W[i][j] * v[i]; return out; }
  function outerAdd(G, a, b) { for (var i = 0; i < a.length; i++) for (var j = 0; j < b.length; j++) G[i][j] += a[i] * b[j]; }
  function vadd(a, b) { return a.map(function (x, i) { return x + b[i]; }); }
  function sig(x) { return 1 / (1 + Math.exp(-x)); }
  function tanh(x) { return Math.tanh(x); }

  /* ---------- 課題データ ---------- */
  function makeBatch(T, n, rng) {
    var batch = [];
    for (var k = 0; k < n; k++) {
      var bit = rng.int(2), xs = [[bit, 1]];
      for (var t = 0; t < T; t++) xs.push([rng.int(2), 0]);
      batch.push({ xs: xs, target: bit });
    }
    return batch;
  }

  /* ---------- パラメータ管理（Adam） ---------- */
  function Params() { this.list = []; }
  Params.prototype.add = function (name, m) { this[name] = m; this.list.push(name); return m; };
  Params.prototype.zeroGrads = function () { var g = {}; var self = this; this.list.forEach(function (n) { var m = self[n]; g[n] = Array.isArray(m[0]) ? zeros(m.length, m[0].length) : new Array(m.length).fill(0); }); return g; };
  Params.prototype.count = function () { var c = 0; var self = this; this.list.forEach(function (n) { var m = self[n]; c += Array.isArray(m[0]) ? m.length * m[0].length : m.length; }); return c; };
  Params.prototype.gradNorm = function (grads) {
    var norm = 0;
    this.list.forEach(function (n) { var g = grads[n]; if (Array.isArray(g[0])) g.forEach(function (r) { r.forEach(function (x) { norm += x * x; }); }); else g.forEach(function (x) { norm += x * x; }); });
    return Math.sqrt(norm);
  };
  // 素朴な SGD（勾配の大きさがそのまま更新量になる＝勾配消失の影響が見える）
  Params.prototype.sgd = function (grads, lr, clip) {
    var self = this, norm = this.gradNorm(grads), sc = clip && norm > clip ? clip / norm : 1;
    this.list.forEach(function (n) {
      var p = self[n], g = grads[n];
      if (Array.isArray(p[0])) { for (var i = 0; i < p.length; i++) for (var j = 0; j < p[0].length; j++) p[i][j] -= lr * sc * g[i][j]; }
      else for (var k = 0; k < p.length; k++) p[k] -= lr * sc * g[k];
    });
    return norm;
  };
  Params.prototype.adam = function (grads, lr, clip) {
    var self = this; if (!this._m) { this._m = this.zeroGrads(); this._v = this.zeroGrads(); this._t = 0; }
    this._t++;
    var norm = this.gradNorm(grads); var sc = clip && norm > clip ? clip / norm : 1;
    var b1 = 0.9, b2 = 0.999, t = this._t;
    this.list.forEach(function (n) {
      var p = self[n], g = grads[n], m = self._m[n], v = self._v[n];
      var upd = function (pi, gi, mi, vi, set) {
        var gg = gi * sc; var nm = b1 * mi + (1 - b1) * gg, nv = b2 * vi + (1 - b2) * gg * gg;
        var mh = nm / (1 - Math.pow(b1, t)), vh = nv / (1 - Math.pow(b2, t));
        set(pi - lr * mh / (Math.sqrt(vh) + 1e-8), nm, nv);
      };
      if (Array.isArray(p[0])) { for (var i = 0; i < p.length; i++) for (var j = 0; j < p[0].length; j++) (function (i, j) { upd(p[i][j], g[i][j], m[i][j], v[i][j], function (np, nm, nv) { p[i][j] = np; m[i][j] = nm; v[i][j] = nv; }); })(i, j); }
      else { for (var k = 0; k < p.length; k++) (function (k) { upd(p[k], g[k], m[k], v[k], function (np, nm, nv) { p[k] = np; m[k] = nm; v[k] = nv; }); })(k); }
    });
    return norm;
  };

  /* ---------- モデル本体 ---------- */
  // type: "rnn" | "lstm" | "gru"
  function Model(type, H, rng) {
    this.type = type; this.H = H; this.X = 2;
    var P = this.P = new Params(), s = 1 / Math.sqrt(H);
    if (type === "rnn") {
      P.add("Wx", randn(H, 2, 0.5, rng)); P.add("Wh", randn(H, H, s, rng)); P.add("b", new Array(H).fill(0));
    } else if (type === "lstm") {
      // ゲート：f=忘却, i=入力, o=出力, g=候補。忘却ゲートのバイアスは 1 で初期化（最初は「覚えておく」側に）
      ["f", "i", "o", "g"].forEach(function (gn) {
        P.add("Wx" + gn, randn(H, 2, 0.5, rng)); P.add("Wh" + gn, randn(H, H, s, rng));
        P.add(gn === "o" ? "bog" : "b" + gn, new Array(H).fill(gn === "f" ? 1 : 0));
      });
    } else {
      // ゲート：z=更新, r=リセット, n=候補。更新ゲートのバイアスは -1 で初期化（最初は「前の状態を保つ」側に）
      ["z", "r", "n"].forEach(function (gn) { P.add("Wx" + gn, randn(H, 2, 0.5, rng)); P.add("Wh" + gn, randn(H, H, s, rng)); P.add("b" + gn, new Array(H).fill(gn === "z" ? -1 : 0)); });
    }
    P.add("Wo", [randn(1, H, s, rng)[0]]); P.add("bo", [0]);
  }

  // 1 系列の順伝播。各ステップの中間値を記録して返す
  Model.prototype.forward = function (xs) {
    var H = this.H, P = this.P, h = new Array(H).fill(0), c = new Array(H).fill(0), steps = [];
    for (var t = 0; t < xs.length; t++) {
      var x = xs[t], st = { x: x, hprev: h, cprev: c };
      if (this.type === "rnn") {
        var a = vadd(vadd(matvec(P.Wx, x), matvec(P.Wh, h)), P.b);
        h = a.map(tanh); st.h = h;
      } else if (this.type === "lstm") {
        var f = vadd(vadd(matvec(P.Wxf, x), matvec(P.Whf, h)), P.bf).map(sig);
        var i = vadd(vadd(matvec(P.Wxi, x), matvec(P.Whi, h)), P.bi).map(sig);
        var o = vadd(vadd(matvec(P.Wxo, x), matvec(P.Who, h)), P.bog).map(sig);
        var g = vadd(vadd(matvec(P.Wxg, x), matvec(P.Whg, h)), P.bg).map(tanh);
        c = c.map(function (cp, k) { return f[k] * cp + i[k] * g[k]; });
        var tc = c.map(tanh);
        h = o.map(function (ok, k) { return ok * tc[k]; });
        st.f = f; st.i = i; st.o = o; st.g = g; st.c = c; st.tc = tc; st.h = h;
      } else {
        var z = vadd(vadd(matvec(P.Wxz, x), matvec(P.Whz, h)), P.bz).map(sig);
        var r = vadd(vadd(matvec(P.Wxr, x), matvec(P.Whr, h)), P.br).map(sig);
        var rh = r.map(function (rk, k) { return rk * h[k]; });
        var n = vadd(vadd(matvec(P.Wxn, x), matvec(P.Whn, rh)), P.bn).map(tanh);
        var hn = h.map(function (hp, k) { return (1 - z[k]) * hp + z[k] * n[k]; });
        st.z = z; st.r = r; st.rh = rh; st.n = n; st.h = hn; h = hn;
      }
      steps.push(st);
    }
    var yl = matvec(P.Wo, h)[0] + P.bo[0], y = sig(yl);
    return { steps: steps, y: y, h: h };
  };

  // BPTT。grads に勾配を加算して返す
  Model.prototype.backward = function (fw, target, grads) {
    var H = this.H, P = this.P, steps = fw.steps, T = steps.length;
    var dy = fw.y - target;                        // BCE + sigmoid の微分
    outerAdd(grads.Wo, [dy], fw.h); grads.bo[0] += dy;
    var dh = matTvec(P.Wo, [dy]), dc = new Array(H).fill(0);
    for (var t = T - 1; t >= 0; t--) {
      var st = steps[t], x = st.x, hp = st.hprev, k;
      if (this.type === "rnn") {
        var da = st.h.map(function (hk, k) { return dh[k] * (1 - hk * hk); });
        outerAdd(grads.Wx, da, x); outerAdd(grads.Wh, da, hp); for (k = 0; k < H; k++) grads.b[k] += da[k];
        dh = matTvec(P.Wh, da);
      } else if (this.type === "lstm") {
        var dO = [], dCt = [], dF = [], dI = [], dG = [], dcp = [];
        for (k = 0; k < H; k++) {
          dO[k] = dh[k] * st.tc[k] * st.o[k] * (1 - st.o[k]);
          dCt[k] = dc[k] + dh[k] * st.o[k] * (1 - st.tc[k] * st.tc[k]);
          dF[k] = dCt[k] * st.cprev[k] * st.f[k] * (1 - st.f[k]);
          dI[k] = dCt[k] * st.g[k] * st.i[k] * (1 - st.i[k]);
          dG[k] = dCt[k] * st.i[k] * (1 - st.g[k] * st.g[k]);
          dcp[k] = dCt[k] * st.f[k];
        }
        outerAdd(grads.Wxf, dF, x); outerAdd(grads.Whf, dF, hp); outerAdd(grads.Wxi, dI, x); outerAdd(grads.Whi, dI, hp);
        outerAdd(grads.Wxo, dO, x); outerAdd(grads.Who, dO, hp); outerAdd(grads.Wxg, dG, x); outerAdd(grads.Whg, dG, hp);
        for (k = 0; k < H; k++) { grads.bf[k] += dF[k]; grads.bi[k] += dI[k]; grads.bog[k] += dO[k]; grads.bg[k] += dG[k]; }
        dh = vadd(vadd(matTvec(P.Whf, dF), matTvec(P.Whi, dI)), vadd(matTvec(P.Who, dO), matTvec(P.Whg, dG)));
        dc = dcp;
      } else {
        var dZ = [], dN = [], dhp = [], dRH;
        for (k = 0; k < H; k++) {
          dZ[k] = dh[k] * (st.n[k] - hp[k]) * st.z[k] * (1 - st.z[k]);
          dN[k] = dh[k] * st.z[k] * (1 - st.n[k] * st.n[k]);
          dhp[k] = dh[k] * (1 - st.z[k]);
        }
        outerAdd(grads.Wxn, dN, x); outerAdd(grads.Whn, dN, st.rh); for (k = 0; k < H; k++) grads.bn[k] += dN[k];
        dRH = matTvec(P.Whn, dN);
        var dR = [];
        for (k = 0; k < H; k++) { dR[k] = dRH[k] * hp[k] * st.r[k] * (1 - st.r[k]); dhp[k] += dRH[k] * st.r[k]; }
        outerAdd(grads.Wxz, dZ, x); outerAdd(grads.Whz, dZ, hp); outerAdd(grads.Wxr, dR, x); outerAdd(grads.Whr, dR, hp);
        for (k = 0; k < H; k++) { grads.bz[k] += dZ[k]; grads.br[k] += dR[k]; }
        dh = vadd(vadd(dhp, matTvec(P.Whz, dZ)), matTvec(P.Whr, dR));
      }
    }
    return -(target * Math.log(fw.y + 1e-9) + (1 - target) * Math.log(1 - fw.y + 1e-9));
  };

  // 1 ステップの学習（ミニバッチ）。平均損失・正答率・勾配ノルムを返す
  Model.prototype.trainStep = function (batch, lr, opt) {
    var grads = this.P.zeroGrads(), loss = 0, correct = 0, self = this;
    batch.forEach(function (ex) {
      var fw = self.forward(ex.xs);
      loss += self.backward(fw, ex.target, grads);
      if ((fw.y > 0.5 ? 1 : 0) === ex.target) correct++;
    });
    // 平均化
    this.P.list.forEach(function (n) { var g = grads[n]; if (Array.isArray(g[0])) g.forEach(function (r) { for (var j = 0; j < r.length; j++) r[j] /= batch.length; }); else for (var k = 0; k < g.length; k++) g[k] /= batch.length; });
    var gnorm = opt === "sgd" ? this.P.sgd(grads, lr, 5) : this.P.adam(grads, lr, 5);
    return { loss: loss / batch.length, acc: correct / batch.length, gnorm: gnorm };
  };

  // 時間方向の勾配の流れ：最終出力の損失が、各ステップの隠れ状態にどれだけ届くか（|∂L/∂h_t|）
  Model.prototype.gradThroughTime = function (xs, target) {
    var fw = this.forward(xs), P = this.P, H = this.H, steps = fw.steps, T = steps.length;
    var dy = fw.y - target, dh = matTvec(P.Wo, [dy]), dc = new Array(H).fill(0), norms = new Array(T), k;
    for (var t = T - 1; t >= 0; t--) {
      var st = steps[t], hp = st.hprev, n2 = 0;
      for (k = 0; k < H; k++) n2 += dh[k] * dh[k] + (this.type === "lstm" ? dc[k] * dc[k] : 0);
      norms[t] = Math.sqrt(n2);
      if (this.type === "rnn") {
        var da = st.h.map(function (hk, k) { return dh[k] * (1 - hk * hk); }); dh = matTvec(P.Wh, da);
      } else if (this.type === "lstm") {
        var dO = [], dF = [], dI = [], dG = [], dcp = [];
        for (k = 0; k < H; k++) {
          dO[k] = dh[k] * st.tc[k] * st.o[k] * (1 - st.o[k]);
          var dCt = dc[k] + dh[k] * st.o[k] * (1 - st.tc[k] * st.tc[k]);
          dF[k] = dCt * st.cprev[k] * st.f[k] * (1 - st.f[k]); dI[k] = dCt * st.g[k] * st.i[k] * (1 - st.i[k]); dG[k] = dCt * st.i[k] * (1 - st.g[k] * st.g[k]); dcp[k] = dCt * st.f[k];
        }
        dh = vadd(vadd(matTvec(P.Whf, dF), matTvec(P.Whi, dI)), vadd(matTvec(P.Who, dO), matTvec(P.Whg, dG))); dc = dcp;
      } else {
        var dZ = [], dN = [], dhp = [];
        for (k = 0; k < H; k++) { dZ[k] = dh[k] * (st.n[k] - hp[k]) * st.z[k] * (1 - st.z[k]); dN[k] = dh[k] * st.z[k] * (1 - st.n[k] * st.n[k]); dhp[k] = dh[k] * (1 - st.z[k]); }
        var dRH = matTvec(P.Whn, dN), dR = [];
        for (k = 0; k < H; k++) { dR[k] = dRH[k] * hp[k] * st.r[k] * (1 - st.r[k]); dhp[k] += dRH[k] * st.r[k]; }
        dh = vadd(vadd(dhp, matTvec(P.Whz, dZ)), matTvec(P.Whr, dR));
      }
    }
    return norms;
  };

  Model.prototype.evaluate = function (batch) {
    var c = 0, self = this; batch.forEach(function (ex) { if ((self.forward(ex.xs).y > 0.5 ? 1 : 0) === ex.target) c++; }); return c / batch.length;
  };

  return { Model: Model, makeBatch: makeBatch };
})();

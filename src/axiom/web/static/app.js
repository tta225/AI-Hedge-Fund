/* AXIOM operator console.
 *
 * No framework and no build step. That is a deliberate constraint: the console
 * ships inside a Python package, and a user who has just run `pip install` must
 * not also need Node, npm and a bundler before they can look at a chart.
 *
 * The chart is hand-drawn on a canvas rather than pulled from a charting
 * library, for the same reason — and because ICT overlays are price *bands*
 * anchored to bar ranges (gaps, order blocks, OTE zones), which most candlestick
 * libraries only support through plugins anyway.
 */
'use strict';

// ───────────────────────────── utilities ─────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmt = {
  /** Price. Decimals follow magnitude: 0.00003412 and 21,450.25 both need to read. */
  price(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const abs = Math.abs(v);
    const dp = abs >= 1000 ? 2 : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 8;
    return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  money(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return '$' + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  },
  num(v, dp = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
  },
  pct(v, dp = 1) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return v.toFixed(dp) + '%';
  },
  signed(v, dp = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return (v >= 0 ? '+' : '') + v.toFixed(dp);
  },
  /** Append a unit only when there is a value. Otherwise "—×" reads as a number. */
  unit(v, suffix, dp = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return fmt.num(v, dp) + suffix;
  },
  time(ms) {
    if (!ms) return '—';
    return new Date(ms).toISOString().replace('T', ' ').slice(0, 16) + 'Z';
  },
  date(ms) {
    if (!ms) return '—';
    return new Date(ms).toISOString().slice(0, 10);
  },
};

/** Read a themed CSS custom property. Keeps canvas colours in step with the theme. */
function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Build a table from rows of cells. Text is set via textContent, never innerHTML. */
function table(headers, rows, emptyMessage) {
  if (!rows.length) {
    const p = el('p', 'muted', emptyMessage || 'None detected in this window.');
    return p;
  }
  const t = el('table');
  const thead = el('thead');
  const hr = el('tr');
  headers.forEach((h) => hr.appendChild(el('th', null, h)));
  thead.appendChild(hr);
  t.appendChild(thead);

  const tbody = el('tbody');
  rows.forEach((cells) => {
    const tr = el('tr');
    cells.forEach((cell) => {
      const td = el('td');
      if (cell && typeof cell === 'object' && 'text' in cell) {
        if (cell.tag) {
          const span = el('span', 'tag ' + (cell.cls || ''), cell.text);
          td.appendChild(span);
        } else {
          td.textContent = cell.text;
          if (cell.cls) td.className = cell.cls;
        }
      } else {
        td.textContent = cell === null || cell === undefined ? '—' : String(cell);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  t.appendChild(tbody);
  return t;
}

function statTile(label, value, opts = {}) {
  const tile = el('div', 'stat');
  tile.appendChild(el('div', 'stat-label', label));
  tile.appendChild(el('div', 'stat-value ' + (opts.cls || ''), value));
  if (opts.sub) tile.appendChild(el('div', 'stat-sub', opts.sub));
  return tile;
}

// ───────────────────────────── network ─────────────────────────────

let busyDepth = 0;

function setBusy(on, label) {
  busyDepth = Math.max(0, busyDepth + (on ? 1 : -1));
  const node = $('#busy');
  if (label) $('#busy-label').textContent = label;
  node.hidden = busyDepth === 0;
}

let toastTimer = null;
function toast(message) {
  const node = $('#toast');
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 9000);
}

/**
 * Call the API.
 *
 * Errors are surfaced with the server's own message. The API's 503 for "no data
 * provider could serve this" carries a hint the user can act on, and replacing
 * it with a generic "request failed" would throw away the only useful part.
 */
async function api(path, body) {
  setBusy(true);
  try {
    const options = body
      ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
      : { method: 'GET' };
    const response = await fetch(path, options);
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; } catch { payload = { raw: text }; }

    if (!response.ok) {
      const detail = payload && payload.detail;
      const message = detail && typeof detail === 'object'
        ? [detail.message, detail.hint].filter(Boolean).join(' — ')
        : (typeof detail === 'string' ? detail : `HTTP ${response.status}`);
      throw new Error(message);
    }
    $('#conn').textContent = 'connected';
    $('#conn').className = 'pill pill-ok';
    return payload;
  } catch (err) {
    $('#conn').textContent = 'error';
    $('#conn').className = 'pill pill-bad';
    throw err;
  } finally {
    setBusy(false);
  }
}

// ───────────────────────────── chart ─────────────────────────────

/**
 * Candlestick chart with ICT overlays.
 *
 * Coordinates: `x(i)` maps a bar index to a pixel centre, `y(p)` maps a price.
 * Both are recomputed on every draw because the canvas is resized by CSS and the
 * price domain changes with the data — caching them is how a chart ends up
 * drawing yesterday's scale over today's bars.
 */
class Chart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.ict = null;
    this.layers = { fvg: true, ob: true, liq: true, struct: true, range: true };
    this.hover = null;
    this.pad = { top: 12, right: 74, bottom: 26, left: 10 };

    canvas.addEventListener('mousemove', (e) => this.onMove(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.draw(); this.hideReadout(); });
    window.addEventListener('resize', () => this.draw());
  }

  setData(series, ict) {
    this.data = series;
    this.ict = ict;
    this.draw();
  }

  /** Device-pixel-ratio aware sizing. Without this every line is blurry on a Mac. */
  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    if (this.w <= 0 || this.h <= 0) return false;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }

  domain() {
    const d = this.data;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < d.low.length; i++) {
      if (d.low[i] !== null && d.low[i] < lo) lo = d.low[i];
      if (d.high[i] !== null && d.high[i] > hi) hi = d.high[i];
    }
    // Widen for the dealing range only. Liquidity pools are deliberately not
    // considered: a single distant pool would compress every candle into the
    // middle third of the plot to make room for one line. Pools outside the
    // price range are simply clipped, and the Draw on Liquidity table reports
    // them with their exact distance.
    if (this.ict && this.layers.range && this.ict.dealing_range) {
      const r = this.ict.dealing_range;
      [r.high, r.low].forEach((v) => {
        if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      });
    }
    const span = hi - lo || 1;
    return [lo - span * 0.06, hi + span * 0.06];
  }

  draw() {
    if (!this.data || !this.resize()) return;
    const ctx = this.ctx;
    const d = this.data;
    const n = d.close.length;
    if (!n) return;

    const [lo, hi] = this.domain();
    const plotW = this.w - this.pad.left - this.pad.right;
    const plotH = this.h - this.pad.top - this.pad.bottom;

    this.x = (i) => this.pad.left + ((i + 0.5) / n) * plotW;
    this.y = (p) => this.pad.top + (1 - (p - lo) / (hi - lo)) * plotH;
    this.barW = Math.max(1, Math.min(14, (plotW / n) * 0.68));

    ctx.clearRect(0, 0, this.w, this.h);
    this.drawGrid(lo, hi, plotW, plotH);

    // Bands first, then levels, then candles: candles must stay readable on top
    // of every overlay, because they are the only thing that is not an inference.
    if (this.ict) {
      if (this.layers.range) this.drawRange();
      if (this.layers.fvg) this.drawGaps();
      if (this.layers.ob) this.drawOrderBlocks();
      if (this.layers.liq) this.drawLiquidity();
    }
    this.drawCandles();
    if (this.ict && this.layers.struct) this.drawStructure();
    this.drawAxis(lo, hi, plotW, plotH);
    if (this.hover !== null) this.drawCrosshair();
  }

  drawGrid(lo, hi, plotW, plotH) {
    const ctx = this.ctx;
    ctx.strokeStyle = css('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= 5; k++) {
      const yy = Math.round(this.pad.top + (k / 5) * plotH) + 0.5;
      ctx.moveTo(this.pad.left, yy);
      ctx.lineTo(this.pad.left + plotW, yy);
    }
    ctx.stroke();
  }

  drawCandles() {
    const ctx = this.ctx;
    const d = this.data;
    const bull = css('--bull');
    const bear = css('--bear');
    const wick = css('--wick');
    const half = this.barW / 2;

    for (let i = 0; i < d.close.length; i++) {
      const o = d.open[i], h = d.high[i], l = d.low[i], c = d.close[i];
      if (o === null || c === null) continue;
      const up = c >= o;
      const xc = this.x(i);

      ctx.strokeStyle = this.barW < 2.5 ? (up ? bull : bear) : wick;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(xc) + 0.5, this.y(h));
      ctx.lineTo(Math.round(xc) + 0.5, this.y(l));
      ctx.stroke();

      if (this.barW >= 2.5) {
        const yo = this.y(o), yc = this.y(c);
        const top = Math.min(yo, yc);
        const height = Math.max(1, Math.abs(yc - yo));
        ctx.fillStyle = up ? bull : bear;
        ctx.fillRect(xc - half, top, this.barW, height);
      }
    }
  }

  /** A price band spanning bar `from` to the right edge. */
  band(from, top, bottom, fill, stroke) {
    const ctx = this.ctx;
    const x0 = this.x(Math.max(0, from));
    const x1 = this.w - this.pad.right;
    if (x1 <= x0) return;
    const yTop = this.y(top);
    const yBot = this.y(bottom);
    ctx.fillStyle = fill;
    ctx.fillRect(x0, yTop, x1 - x0, Math.max(1, yBot - yTop));
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.strokeRect(x0 + 0.5, Math.round(yTop) + 0.5, x1 - x0 - 1, Math.max(1, yBot - yTop));
    }
  }

  /*
   * Overlay density is capped, and that is a correctness decision, not a
   * cosmetic one. A 30-day 15m window yields ~440 fair value gaps and ~410
   * liquidity pools. Drawing all of them fills the plot with overlapping bands
   * until the candles are invisible — the chart stops showing the market and
   * starts showing the detector's output. So each layer draws only what is
   * still live, ranked by relevance, capped, and everything else stays in the
   * tables below, which are complete.
   */
  static CAPS = { fvg: 30, ob: 24, liq: 18 };

  /** Live features nearest current price, most recent first, capped. */
  nearest(items, limit, priceOf) {
    const last = this.data.close[this.data.close.length - 1];
    return items
      .filter((f) => f.confirmed_index >= 0)
      .map((f) => ({ f, d: Math.abs(priceOf(f) - last) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, limit)
      .map((x) => x.f);
  }

  drawGaps() {
    // Only unmitigated gaps. A filled gap has already done what it was going to
    // do; it stays in the table, where it can be read without costing the chart.
    const live = this.ict.fair_value_gaps.filter((g) => g.is_active);
    this.nearest(live, Chart.CAPS.fvg, (g) => g.midpoint).forEach((g) => {
      const fill = g.direction === 'bullish' ? css('--fvg-bull') : css('--fvg-bear');
      this.band(g.confirmed_index, g.top, g.bottom, fill, null);
    });
  }

  drawOrderBlocks() {
    const live = this.ict.order_blocks.filter((b) => b.is_active);
    this.nearest(live, Chart.CAPS.ob, (b) => b.mean_threshold).forEach((b) => {
      const fill = b.effective_direction === 'bullish' ? css('--ob-bull') : css('--ob-bear');
      this.band(b.confirmed_index, b.top, b.bottom, fill, null);
    });
  }

  drawLiquidity() {
    const ctx = this.ctx;
    const colour = css('--liq');
    // Unswept pools only — a swept pool is spent liquidity, not a draw. Ranked
    // by strength rather than distance: an equal-highs cluster further away is a
    // more meaningful target than a lone swing just overhead.
    const live = this.ict.liquidity_pools
      .filter((p) => !p.is_swept && p.confirmed_index >= 0)
      .sort((a, b) => b.strength - a.strength)
      .slice(0, Chart.CAPS.liq);

    live.forEach((p) => {
      const yy = Math.round(this.y(p.price)) + 0.5;
      ctx.strokeStyle = colour;
      ctx.globalAlpha = p.is_equal_cluster ? 0.8 : 0.4;
      ctx.lineWidth = p.is_equal_cluster ? 1.5 : 1;
      ctx.setLineDash(p.is_equal_cluster ? [] : [3, 3]);
      ctx.beginPath();
      ctx.moveTo(this.x(p.confirmed_index), yy);
      ctx.lineTo(this.w - this.pad.right, yy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    });
  }

  drawRange() {
    const r = this.ict.dealing_range;
    if (!r) return;
    const ctx = this.ctx;
    const colour = css('--range');

    ctx.strokeStyle = colour;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    [r.high, r.low].forEach((level) => {
      const yy = Math.round(this.y(level)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(this.pad.left, yy);
      ctx.lineTo(this.w - this.pad.right, yy);
      ctx.stroke();
    });

    ctx.setLineDash([5, 4]);
    const yEq = Math.round(this.y(r.equilibrium)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(this.pad.left, yEq);
    ctx.lineTo(this.w - this.pad.right, yEq);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // OTE band for the side the current bias favours.
    const ote = this.ict.bias === 'bearish' ? r.short_ote : r.long_ote;
    if (ote && ote[0] !== null && ote[1] !== null) {
      ctx.fillStyle = colour;
      ctx.globalAlpha = 0.1;
      const yTop = this.y(ote[1]);
      const yBot = this.y(ote[0]);
      ctx.fillRect(this.pad.left, yTop, this.w - this.pad.right - this.pad.left, Math.max(1, yBot - yTop));
      ctx.globalAlpha = 1;
    }
  }

  drawStructure() {
    const ctx = this.ctx;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    // Most recent only — the same density argument as the bands. The Structure
    // Events table carries the full history.
    this.ict.structure_events.slice(-45).forEach((e) => {
      if (e.confirmed_index < 0) return;
      const xx = this.x(e.confirmed_index);
      const yy = this.y(e.level);
      const bull = e.direction === 'bullish';
      ctx.fillStyle = e.is_reversal ? css('--info') : (bull ? css('--bull') : css('--bear'));
      ctx.globalAlpha = 0.9;
      const size = 3.5;
      ctx.beginPath();
      if (bull) {
        ctx.moveTo(xx, yy - size); ctx.lineTo(xx - size, yy + size); ctx.lineTo(xx + size, yy + size);
      } else {
        ctx.moveTo(xx, yy + size); ctx.lineTo(xx - size, yy - size); ctx.lineTo(xx + size, yy - size);
      }
      ctx.closePath();
      ctx.fill();
      if (this.barW > 4) {
        ctx.fillText(e.event_type.toUpperCase(), xx, bull ? yy + 16 : yy - 10);
      }
      ctx.globalAlpha = 1;
    });
    ctx.textAlign = 'start';
  }

  drawAxis(lo, hi, plotW, plotH) {
    const ctx = this.ctx;
    ctx.fillStyle = css('--text-mute');
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let k = 0; k <= 5; k++) {
      const price = hi - (k / 5) * (hi - lo);
      const yy = this.pad.top + (k / 5) * plotH;
      ctx.fillText(fmt.price(price), this.pad.left + plotW + 7, yy);
    }

    // Last close, highlighted — the number an operator looks for first.
    const last = this.data.close[this.data.close.length - 1];
    if (last !== null) {
      const yy = this.y(last);
      const label = fmt.price(last);
      const width = ctx.measureText(label).width + 10;
      ctx.fillStyle = css('--accent');
      ctx.fillRect(this.pad.left + plotW + 3, yy - 8, width, 16);
      ctx.fillStyle = css('--bg');
      ctx.fillText(label, this.pad.left + plotW + 8, yy);
    }

    ctx.fillStyle = css('--text-mute');
    ctx.textBaseline = 'top';
    const ticks = Math.min(6, this.data.time.length);
    for (let k = 0; k < ticks; k++) {
      const i = Math.floor((k / (ticks - 1 || 1)) * (this.data.time.length - 1));
      const label = fmt.date(this.data.time[i]);
      ctx.textAlign = k === 0 ? 'left' : k === ticks - 1 ? 'right' : 'center';
      ctx.fillText(label, this.x(i), this.pad.top + plotH + 7);
    }
    ctx.textAlign = 'start';
  }

  drawCrosshair() {
    const ctx = this.ctx;
    const i = this.hover;
    const xx = Math.round(this.x(i)) + 0.5;
    ctx.strokeStyle = css('--text-mute');
    ctx.globalAlpha = 0.45;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xx, this.pad.top);
    ctx.lineTo(xx, this.h - this.pad.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  onMove(event) {
    if (!this.data) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const n = this.data.close.length;
    const plotW = this.w - this.pad.left - this.pad.right;
    const i = Math.round(((px - this.pad.left) / plotW) * n - 0.5);
    if (i < 0 || i >= n) { this.hover = null; this.hideReadout(); this.draw(); return; }
    this.hover = i;
    this.draw();
    this.showReadout(i, px, rect);
  }

  showReadout(i, px, rect) {
    const d = this.data;
    const node = $('#crosshair-readout');
    if (!node) return;
    const rows = [
      ['time', fmt.time(d.time[i])],
      ['open', fmt.price(d.open[i])],
      ['high', fmt.price(d.high[i])],
      ['low', fmt.price(d.low[i])],
      ['close', fmt.price(d.close[i])],
      ['volume', fmt.num(d.volume[i], 0)],
    ];
    node.replaceChildren(
      ...rows.map(([k, v]) => {
        const line = el('div');
        line.appendChild(el('span', 'k', k));
        line.appendChild(el('span', null, v));
        return line;
      })
    );
    node.hidden = false;
    const flip = px > rect.width * 0.6;
    node.style.left = flip ? '' : `${px + 18}px`;
    node.style.right = flip ? `${rect.width - px + 18}px` : '';
    node.style.top = '16px';
  }

  hideReadout() {
    const node = $('#crosshair-readout');
    if (node) node.hidden = true;
  }
}

/** Equity curve — a simple filled line, same coordinate discipline as Chart. */
class LineChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.points = null;
    window.addEventListener('resize', () => this.draw());
  }

  setData(times, values, baseline) {
    this.times = times;
    this.values = values;
    this.baseline = baseline;
    this.draw();
  }

  draw() {
    if (!this.values || !this.values.length) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const pad = { top: 12, right: 74, bottom: 24, left: 10 };
    const plotW = rect.width - pad.left - pad.right;
    const plotH = rect.height - pad.top - pad.bottom;
    const clean = this.values.filter((v) => v !== null && isFinite(v));
    let lo = Math.min(...clean, this.baseline);
    let hi = Math.max(...clean, this.baseline);
    const span = hi - lo || 1;
    lo -= span * 0.08; hi += span * 0.08;

    const x = (i) => pad.left + (i / Math.max(1, this.values.length - 1)) * plotW;
    const y = (v) => pad.top + (1 - (v - lo) / (hi - lo)) * plotH;

    ctx.strokeStyle = css('--grid');
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= 4; k++) {
      const yy = Math.round(pad.top + (k / 4) * plotH) + 0.5;
      ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + plotW, yy);
    }
    ctx.stroke();

    // Starting equity: the line that decides whether this run made money.
    const yBase = Math.round(y(this.baseline)) + 0.5;
    ctx.strokeStyle = css('--text-mute');
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, yBase); ctx.lineTo(pad.left + plotW, yBase);
    ctx.stroke();
    ctx.setLineDash([]);

    const final = clean[clean.length - 1];
    const colour = final >= this.baseline ? css('--bull') : css('--bear');

    ctx.beginPath();
    this.values.forEach((v, i) => {
      if (v === null || !isFinite(v)) return;
      const px = x(i), py = y(v);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.75;
    ctx.stroke();

    ctx.lineTo(x(this.values.length - 1), y(lo));
    ctx.lineTo(x(0), y(lo));
    ctx.closePath();
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.globalAlpha = 1;

    ctx.fillStyle = css('--text-mute');
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const value = hi - (k / 4) * (hi - lo);
      ctx.fillText('$' + Math.round(value).toLocaleString(), pad.left + plotW + 7, pad.top + (k / 4) * plotH);
    }
  }
}

// ───────────────────────────── views ─────────────────────────────

const chart = new Chart($('#chart'));
const equityChart = new LineChart($('#equity-chart'));

function query() {
  return {
    symbol: $('#symbol').value.trim().toUpperCase() || 'ES',
    timeframe: $('#timeframe').value,
    days: Math.max(1, parseInt($('#days').value, 10) || 30),
    synthetic: $('#synthetic').checked,
    seed: 7,
  };
}

function renderProvenance(provenance) {
  const node = $('#provenance');
  node.hidden = false;
  if (provenance.is_evidential) {
    node.className = 'provenance is-real';
    node.textContent = `REAL MARKET DATA — ${provenance.describe}`;
  } else {
    node.className = 'provenance is-synthetic';
    node.textContent = `⚠ GENERATED DATA — NOT MARKET HISTORY. A correctness check, never evidence. (${provenance.describe})`;
  }
}

function renderDashboard(payload) {
  const { series, ict } = payload;
  renderProvenance(series.provenance);
  chart.setData(series, ict);

  const last = ict.last_price;
  const prev = series.close[series.close.length - 2];
  const change = prev ? last - prev : 0;
  const changePct = prev ? (change / prev) * 100 : 0;

  const biasCls = ict.bias === 'bullish' ? 'up' : ict.bias === 'bearish' ? 'down' : 'neutral';
  const range = ict.dealing_range;

  $('#statrow').replaceChildren(
    statTile(series.symbol + ' · ' + series.timeframe, fmt.price(last), {
      cls: change >= 0 ? 'up' : 'down',
      sub: `${fmt.signed(change)} (${fmt.signed(changePct)}%)`,
    }),
    statTile('Bias', ict.bias.toUpperCase(), { cls: biasCls, sub: ict.session || 'off session' }),
    statTile('Range position',
      range ? fmt.pct(range.position * 100) : '—',
      { cls: range ? (range.is_premium ? 'down' : 'up') : 'neutral',
        sub: range ? (range.is_premium ? 'PREMIUM' : 'DISCOUNT') : 'no dealing range' }),
    statTile('Active FVGs', String(ict.fair_value_gaps.filter((g) => g.is_active).length),
      { sub: `${ict.counts.fair_value_gaps} total` }),
    statTile('Active OBs', String(ict.order_blocks.filter((b) => b.is_active).length),
      { sub: `${ict.counts.order_blocks} total` }),
    statTile('Sweeps', String(ict.counts.sweeps), { sub: `${ict.counts.liquidity_pools} pools` }),
    statTile('Bars', series.bars.toLocaleString(), { sub: `of ${series.total_bars.toLocaleString()}` }),
  );

  $('#chart-foot').textContent = series.describe;

  // ── structure panel
  const dl = el('dl', 'kv');
  const addRow = (k, v, cls) => {
    dl.appendChild(el('dt', null, k));
    dl.appendChild(el('dd', cls, v));
  };
  const lastEvent = ict.structure_events[ict.structure_events.length - 1];
  addRow('bias', ict.bias.toUpperCase(), biasCls);
  addRow('session', ict.session || 'off session');
  addRow('last event', lastEvent
    ? `${lastEvent.event_type.toUpperCase()} ${lastEvent.direction} @ ${fmt.price(lastEvent.level)}`
    : 'none');
  if (range) {
    addRow('range', `${fmt.price(range.low)} — ${fmt.price(range.high)}`);
    addRow('equilibrium', fmt.price(range.equilibrium));
    addRow('position', `${fmt.pct(range.position * 100)} ${range.is_premium ? 'PREMIUM' : 'DISCOUNT'}`);
    const ote = ict.bias === 'bearish' ? range.short_ote : range.long_ote;
    addRow(ict.bias === 'bearish' ? 'short OTE' : 'long OTE', `${fmt.price(ote[0])} — ${fmt.price(ote[1])}`);
  }
  addRow('swings', String(ict.counts.swings));
  $('#structure-panel').replaceChildren(dl);

  // ── liquidity
  const pools = ict.liquidity_pools
    .filter((p) => !p.is_swept)
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 30)
    .map((p) => [
      { text: p.kind.toUpperCase(), tag: true, cls: p.kind === 'buyside' ? 'tag-bull' : 'tag-bear' },
      fmt.price(p.price),
      { text: fmt.signed(p.price - last), cls: p.price >= last ? 'up' : 'down' },
      p.is_equal_cluster ? `${p.size}×` : '—',
      fmt.num(p.strength, 2),
    ]);
  $('#liquidity-panel').replaceChildren(
    table(['Side', 'Level', 'Distance', 'Equal', 'Strength'], pools, 'No unswept liquidity in this window.')
  );

  // ── zones
  const zones = [];
  ict.order_blocks.filter((b) => b.is_active).slice(-20).reverse().forEach((b) => {
    zones.push([
      { text: b.is_breaker ? 'BREAKER' : 'OB', tag: true, cls: b.effective_direction === 'bullish' ? 'tag-bull' : 'tag-bear' },
      `${fmt.price(b.bottom)}–${fmt.price(b.top)}`,
      fmt.price(b.mean_threshold),
      fmt.num(b.displacement_atr, 2),
      [b.anchored_at_pivot ? 'pivot' : '', b.has_imbalance ? 'FVG' : ''].filter(Boolean).join(' ') || '—',
    ]);
  });
  ict.fair_value_gaps.filter((g) => g.is_active).slice(-20).reverse().forEach((g) => {
    zones.push([
      { text: 'FVG', tag: true, cls: g.direction === 'bullish' ? 'tag-bull' : 'tag-bear' },
      `${fmt.price(g.bottom)}–${fmt.price(g.top)}`,
      fmt.price(g.midpoint),
      fmt.num(g.displacement_atr, 2),
      fmt.pct(g.fill_ratio * 100, 0) + ' filled',
    ]);
  });
  $('#zones-panel').replaceChildren(
    table(['Type', 'Zone', 'Mean / CE', 'Displ. ATR', 'Quality'], zones, 'No active zones in this window.')
  );

  // ── events
  const events = ict.structure_events.slice(-25).reverse().map((e) => [
    { text: e.event_type.toUpperCase(), tag: true,
      cls: e.is_reversal ? 'tag-info' : (e.direction === 'bullish' ? 'tag-bull' : 'tag-bear') },
    { text: e.direction, cls: e.direction === 'bullish' ? 'up' : 'down' },
    fmt.price(e.level),
    fmt.num(e.displacement_atr, 2),
    fmt.time(e.timestamp),
  ]);
  $('#events-panel').replaceChildren(
    table(['Event', 'Direction', 'Level', 'Displ. ATR', 'Confirmed'], events, 'No structure events in this window.')
  );
}

async function runAnalyse() {
  const btn = $('#run');
  btn.disabled = true;
  try {
    const payload = await api('/api/analyse', { ...query(), strict: $('#strict').checked, max_bars: 1500 });
    renderDashboard(payload);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

// ── backtest

function renderBacktest(payload) {
  const bt = payload.backtest;
  const m = bt.metrics;
  renderProvenance(bt.provenance);

  const warn = $('#bt-warning');
  if (bt.warning) { warn.hidden = false; warn.textContent = bt.warning; }
  else { warn.hidden = true; }

  const ret = m.total_return_pct;
  $('#bt-stats').replaceChildren(
    statTile('Ending equity', fmt.money(bt.ending_equity), {
      cls: bt.ending_equity >= bt.starting_equity ? 'up' : 'down',
      sub: `from ${fmt.money(bt.starting_equity)}`,
    }),
    statTile('Total return', fmt.unit(ret, '%'), { cls: ret >= 0 ? 'up' : 'down' }),
    statTile('Trades', fmt.num(m.trades, 0), { sub: fmt.pct(m.win_rate_pct) + ' win rate' }),
    statTile('Profit factor', fmt.num(m.profit_factor)),
    statTile('Expectancy', fmt.money(m.expectancy), { sub: fmt.unit(m.avg_r, 'R avg') }),
    statTile('Max drawdown', fmt.pct(m.max_drawdown_pct), { cls: 'down' }),
    statTile('Sharpe', fmt.num(m.sharpe), { sub: 'Sortino ' + fmt.num(m.sortino) }),
    statTile('Worst loss vs budget', fmt.unit(m.worst_loss_vs_budget, '×'), {
      cls: m.worst_loss_vs_budget > 1 ? 'down' : 'up',
      sub: 'per-trade risk budget',
    }),
  );

  const curve = bt.equity_curve;
  equityChart.setData(curve.time, curve.value, bt.starting_equity);
  $('#bt-span').textContent = curve.time.length
    ? `${fmt.date(curve.time[0])} → ${fmt.date(curve.time[curve.time.length - 1])}`
    : '';

  const f = bt.funnel;
  const dl = el('dl', 'kv');
  const add = (k, v) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', null, v)); };
  add('signals generated', String(f.signals));
  add('orders routed', String(f.orders));
  add('trades completed', String(f.trades));
  Object.entries(f.risk_rejections).forEach(([k, v]) => add('rejected · ' + k, String(v)));
  Object.entries(f.rejections).forEach(([k, v]) => add('skipped · ' + k, String(v)));
  $('#funnel-panel').replaceChildren(dl);

  const notes = el('ul');
  notes.style.margin = '0';
  notes.style.paddingLeft = '18px';
  notes.style.fontSize = '12.5px';
  notes.style.lineHeight = '1.7';
  bt.notes.forEach((n) => notes.appendChild(el('li', 'muted', n)));
  $('#notes-panel').replaceChildren(bt.notes.length ? notes : el('p', 'muted', 'No caveats recorded.'));

  const trades = bt.trades.map((t) => [
    { text: t.side.toUpperCase(), tag: true, cls: t.side === 'buy' ? 'tag-bull' : 'tag-bear' },
    fmt.time(t.entry_time),
    fmt.price(t.entry_price),
    fmt.price(t.exit_price),
    fmt.num(t.quantity, 2),
    { text: fmt.money(t.pnl), cls: t.is_win ? 'up' : 'down' },
    { text: fmt.signed(t.r_multiple) + 'R', cls: t.is_win ? 'up' : 'down' },
    t.exit_reason || '—',
  ]);
  $('#trades-panel').replaceChildren(
    table(['Side', 'Entry time', 'Entry', 'Exit', 'Qty', 'P&L', 'R', 'Reason'], trades,
      'No trades were taken. Check the signal funnel above.')
  );
}

async function runBacktest() {
  const btn = $('#run-backtest');
  btn.disabled = true;
  try {
    const payload = await api('/api/backtest', {
      ...query(),
      strategy: $('#strategy').value,
      equity: parseFloat($('#equity').value) || 250000,
      warmup: parseInt($('#warmup').value, 10) || 0,
      seed: 11,
    });
    renderBacktest(payload);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

// ── agents

function renderAgents(payload) {
  const pipeline = payload.pipeline;
  renderProvenance(payload.provenance);

  const llm = $('#agent-llm');
  llm.textContent = payload.llm_enabled ? 'LLM enabled' : 'deterministic (no ANTHROPIC_API_KEY)';
  llm.className = 'pill ' + (payload.llm_enabled ? 'pill-ok' : 'pill-muted');

  const gate = $('#agent-gate');
  gate.hidden = false;
  gate.replaceChildren(
    el('strong', null, 'HUMAN APPROVAL REQUIRED — this pipeline cannot place an order. '),
    el('span', null, pipeline.approval_summary || '')
  );

  const host = $('#agent-stages');
  host.replaceChildren(...pipeline.reports.map((report) => {
    const panel = el('div', 'panel');
    const head = el('div', 'panel-head');
    head.appendChild(el('h2', null, report.role));
    head.appendChild(
      el('span', 'pill ' + (report.used_llm ? 'pill-ok' : 'pill-muted'),
         report.used_llm ? `narrative · ${report.model || 'llm'}` : 'facts only')
    );
    panel.appendChild(head);

    const body = el('div', 'panel-body');

    // Facts and narrative are kept visually distinct, not just structurally.
    // Merging them is the one thing that would make the LLM unsafe here.
    const facts = Object.entries(report.facts || {});
    if (facts.length) {
      body.appendChild(el('div', 'stat-label', 'measured facts'));
      const dl = el('dl', 'kv');
      facts.forEach(([k, v]) => {
        dl.appendChild(el('dt', null, k));
        dl.appendChild(el('dd', null, formatFact(v)));
      });
      body.appendChild(dl);
    }

    if (report.narrative) {
      body.appendChild(el('div', 'stat-label', 'generated narrative'));
      const prose = el('p', 'agent-narrative', report.narrative);
      body.appendChild(prose);
    }

    if (report.warnings && report.warnings.length) {
      body.appendChild(el('div', 'stat-label', 'warnings'));
      const ul = el('ul', 'agent-warnings');
      report.warnings.forEach((w) => ul.appendChild(el('li', null, w)));
      body.appendChild(ul);
    }

    panel.appendChild(body);
    return panel;
  }));
}

function formatFact(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return fmt.num(value);
  if (Array.isArray(value)) return value.map(formatFact).join(', ') || '—';
  if (typeof value === 'object') {
    return Object.entries(value).map(([k, v]) => `${k}=${formatFact(v)}`).join('  ') || '—';
  }
  return String(value);
}

async function runAgents() {
  const btn = $('#run-agents');
  btn.disabled = true;
  try {
    const payload = await api('/api/pipeline', {
      ...query(),
      strategy: $('#agent-strategy').value,
      equity: parseFloat($('#agent-equity').value) || 250000,
    });
    renderAgents(payload);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

// ── broker

function renderBroker(payload) {
  const live = $('#broker-live');
  live.textContent = payload.is_live ? 'LIVE — REAL MONEY' : 'paper — no real money';
  live.className = 'pill ' + (payload.is_live ? 'pill-bad' : 'pill-ok');

  const account = payload.account || {};
  $('#broker-stats').replaceChildren(
    // 'unavailable' rather than 'no credentials' — NinjaTrader is unavailable
    // on a Mac for a reason that has nothing to do with keys, and naming the
    // wrong cause sends someone off checking credentials that are fine.
    statTile('Venue', payload.venue || '—', { sub: payload.available ? 'connected' : 'unavailable — see below' }),
    statTile('Equity', fmt.money(parseFloat(account.equity))),
    statTile('Buying power', fmt.money(parseFloat(account.buying_power))),
    statTile('Cash', fmt.money(parseFloat(account.cash))),
    statTile('Positions', String((payload.positions || []).length)),
    statTile('Status', account.status || '—'),
  );

  const recon = $('#broker-recon');
  if (payload.reconciliation) {
    const r = payload.reconciliation;
    const box = el('div', 'callout ' + (r.is_clean ? 'callout-info' : 'callout-warn'));
    box.appendChild(el('strong', null, r.is_clean ? 'Reconciled. ' : 'RECONCILIATION FAILED — do not trade until resolved. '));
    box.appendChild(el('span', null, r.render || ''));
    recon.replaceChildren(box);
  } else {
    recon.replaceChildren();
  }

  const rows = (payload.positions || []).map((p) => [
    p.symbol,
    { text: fmt.num(p.quantity, 2), cls: p.quantity >= 0 ? 'up' : 'down' },
    fmt.price(p.average_price),
    // Futures venues report size and price but not a marked value. An em dash
    // says "not reported"; a $0.00 would say "worthless", which is a different
    // and much worse claim.
    p.market_value == null ? '—' : fmt.money(p.market_value),
    p.unrealised_pnl == null
      ? '—'
      : { text: fmt.money(p.unrealised_pnl), cls: p.unrealised_pnl >= 0 ? 'up' : 'down' },
  ]);
  $('#broker-positions').replaceChildren(
    table(['Symbol', 'Qty', 'Avg price', 'Market value', 'Unrealised'], rows,
      payload.available ? 'No open positions at the broker.' : (payload.status || 'Venue unavailable.'))
  );

  if (payload.error) toast(payload.error);
}

async function loadBroker(reconcile = false) {
  const venue = $('#broker-venue')?.value || 'alpaca';
  const params = new URLSearchParams({ venue_name: venue });
  if (reconcile) params.set('reconcile', 'true');
  try {
    renderBroker(await api('/api/broker?' + params));
  } catch (err) {
    toast(err.message);
  }
}

// ── data sources

async function loadStatus() {
  try {
    const s = await api('/api/status');
    $('#sources-panel').replaceChildren(
      ...s.sources.map((src) => {
        const row = el('div', 'source');
        row.appendChild(el('span', 'pill ' + (src.ok ? 'pill-ok' : 'pill-bad'), src.ok ? 'OK' : 'FAIL'));
        row.appendChild(el('span', 'source-name', src.name));
        row.appendChild(el('span', 'source-detail', src.detail));
        return row;
      })
    );
    $('#creds-panel').textContent = s.credentials;

    const dl = el('dl', 'kv');
    const add = (k, v, cls) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', cls, v)); };
    add('order', s.registry.order.join(', ') || 'none');
    add('credentialed', s.registry.credentialed.join(', ') || 'none',
      s.registry.credentialed.length ? '' : 'down');
    add('reachable', s.registry.reachable.join(', ') || 'none',
      s.registry.reachable.length ? 'up' : 'down');
    add('trading mode', s.settings.trading_mode);
    add('kill switch', s.settings.kill_switch ? 'ENGAGED' : 'clear',
      s.settings.kill_switch ? 'down' : 'up');
    $('#registry-panel').replaceChildren(dl);
  } catch (err) {
    toast(err.message);
  }
}

// ───────────────────────────── wiring ─────────────────────────────

function switchView(name) {
  $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('is-active', v.id === 'view-' + name));
  // Canvases sized while hidden measure zero, so redraw once visible.
  if (name === 'dashboard') chart.draw();
  if (name === 'backtest') equityChart.draw();
  if (name === 'data') loadStatus();
  if (name === 'broker') loadBroker();
}

async function boot() {
  const saved = localStorage.getItem('axiom-theme');
  if (saved) document.documentElement.dataset.theme = saved;

  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('axiom-theme', next);
    chart.draw();
    equityChart.draw();
  });

  $$('.tab').forEach((tab) => tab.addEventListener('click', () => switchView(tab.dataset.view)));
  $('#run').addEventListener('click', runAnalyse);
  $('#run-backtest').addEventListener('click', runBacktest);
  $('#refresh-status').addEventListener('click', loadStatus);
  $('#run-agents').addEventListener('click', runAgents);
  $('#refresh-broker').addEventListener('click', () => loadBroker(false));
  $('#run-reconcile').addEventListener('click', () => loadBroker(true));
  // Switching venue reloads without reconciling: a reconciliation carried over
  // from the previous venue would be read as belonging to the new one.
  $('#broker-venue').addEventListener('change', () => loadBroker(false));
  $('#symbol').addEventListener('keydown', (e) => { if (e.key === 'Enter') runAnalyse(); });

  $$('#legend input').forEach((box) => {
    box.addEventListener('change', () => {
      chart.layers[box.dataset.layer] = box.checked;
      chart.draw();
    });
  });

  try {
    const meta = await api('/api/instruments');
    $('#symbols').replaceChildren(
      ...meta.instruments.map((i) => {
        const opt = el('option');
        opt.value = i.symbol;
        opt.label = `${i.description} (${i.asset_class})`;
        return opt;
      })
    );
    $('#timeframe').replaceChildren(
      ...meta.timeframes.map((t) => {
        const opt = el('option', null, t);
        opt.value = t;
        if (t === '15m') opt.selected = true;
        return opt;
      })
    );
    const strategyOptions = () => meta.strategies.map((s) => {
      const opt = el('option', null, s);
      opt.value = s;
      return opt;
    });
    $('#strategy').replaceChildren(...strategyOptions());
    $('#agent-strategy').replaceChildren(...strategyOptions());
  } catch (err) {
    toast('Could not reach the AXIOM API: ' + err.message);
    return;
  }

  // Probe data sources before the first analyse. With no reachable provider the
  // default (real data) can only fail, and starting on generated bars — clearly
  // labelled — beats opening on an error the user did not cause.
  try {
    const s = await api('/api/status');
    if (!s.registry.reachable.length) {
      $('#synthetic').checked = true;
      toast('No real data source is reachable, so this opened on generated bars. ' +
            'See the Data Sources tab.');
    }
  } catch { /* the analyse call will report it */ }

  runAnalyse();
}

boot();

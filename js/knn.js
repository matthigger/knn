// k nearest neighbors: distances, neighbor order, vote and average, the
// error at every k at once, and the vote over a grid for the shaded
// decision regions. X holds the training features, y their labels.

/** Distance between feature arrays a and b, metric "l2" or "l1". */
function dist(a, b, metric) {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const d = a[j] - b[j];
    s += metric === "l1" ? Math.abs(d) : d * d;
  }
  return metric === "l1" ? s : Math.sqrt(s);
}

/**
 * Sort the training samples by distance to q.
 *
 * Returns:
 *   {order, d}: order (n,) indices nearest first (ties by index),
 *     d (n,) distance of each training sample to q
 */
function neighborOrder(q, X, metric) {
  const d = X.map(x => dist(q, x, metric));
  const order = X.map((_, i) => i).sort((a, b) => d[a] - d[b] || a - b);
  return { order, d };
}

/**
 * Majority vote of the k nearest, labels 0..C-1.
 *
 * A tie drops the farthest neighbor and votes again (k-1 NN), until the
 * tie breaks; at worst this ends at the single nearest.
 *
 * Returns:
 *   {counts, winner, kUsed}: counts (C,) votes of the k nearest, winner
 *     the estimate, kUsed the number of nearest whose vote decided it
 */
function vote(order, y, k, C) {
  const counts = new Array(C).fill(0);
  for (let r = 0; r < k; r++) counts[y[order[r]]]++;
  const c = counts.slice();
  let kUsed = k;
  while (isTie(c)) c[y[order[--kUsed]]]--;
  return { counts, winner: argmax(c), kUsed };
}

/** Index of the largest entry, the first one on a tie. */
function argmax(a) {
  let w = 0;
  for (let c = 1; c < a.length; c++) if (a[c] > a[w]) w = c;
  return w;
}

/** Whether the largest entry of a is shared. */
function isTie(a) {
  const top = Math.max(...a);
  return a.indexOf(top) !== a.lastIndexOf(top);
}

/** Average label of the k nearest. */
function average(order, y, k) {
  let s = 0;
  for (let r = 0; r < k; r++) s += y[order[r]];
  return s / k;
}

/**
 * Error of the k-NN estimate on (evalX, evalY) for every k = 1..n.
 *
 * Each evaluation sample is sorted once; growing k adds one neighbor. A
 * tie at k keeps the estimate from k-1, which is the rule in vote().
 * Evaluating on the training set counts each sample among its own
 * neighbors (distance 0), as k-NN does when asked about a training sample.
 *
 * Args:
 *   C (number): class count, or 0 for regression
 *
 * Returns:
 *   err (Float64Array): (n+1,) err[k] = error rate (C > 0) or MSE (C = 0);
 *     err[0] unused
 */
function errorByK(evalX, evalY, X, y, C, metric) {
  const n = X.length, err = new Float64Array(n + 1);
  for (let m = 0; m < evalX.length; m++) {
    const { order } = neighborOrder(evalX[m], X, metric);
    if (C) {
      const counts = new Array(C).fill(0);
      let w = 0;
      for (let k = 1; k <= n; k++) {
        counts[y[order[k - 1]]]++;
        if (!isTie(counts)) w = argmax(counts);
        if (w !== evalY[m]) err[k]++;
      }
    } else {
      let s = 0;
      for (let k = 1; k <= n; k++) {
        s += y[order[k - 1]];
        err[k] += (evalY[m] - s / k) ** 2;
      }
    }
  }
  for (let k = 1; k <= n; k++) err[k] /= evalX.length;
  return err;
}

/** Value at sorted position k (0-based) of a; reorders a (quickselect). */
function kthSmallest(a, k) {
  let lo = 0, hi = a.length - 1;
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const t = a[i]; a[i] = a[j]; a[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return a[k];
  }
  return a[k];
}

/**
 * Vote of the k nearest at the center of each cell of a G x G grid.
 *
 * Same rule as vote(). The k nearest are found by quickselect on the
 * distances rather than a full sort, since the grid holds many more points
 * than the samples; only a tied cell pays for the sort.
 *
 * Args:
 *   xdom, ydom (number[]): [lo, hi] extent of the grid; row 0 is ydom[1]
 *
 * Returns:
 *   winner (Uint8Array): (G*G,) estimate per cell, row-major
 */
function gridVotes(G, xdom, ydom, X, y, C, k, metric) {
  const n = X.length, d = new Float64Array(n), buf = new Float64Array(n);
  const winner = new Uint8Array(G * G);
  const counts = new Array(C);
  const q = [0, 0];
  for (let gy = 0; gy < G; gy++) {
    q[1] = ydom[1] - (gy + 0.5) / G * (ydom[1] - ydom[0]);
    for (let gx = 0; gx < G; gx++) {
      q[0] = xdom[0] + (gx + 0.5) / G * (xdom[1] - xdom[0]);
      for (let i = 0; i < n; i++) d[i] = buf[i] = dist(q, X[i], metric);
      const thr = kthSmallest(buf, k - 1);
      counts.fill(0);
      let left = k;
      for (let i = 0; i < n; i++) {
        if (d[i] < thr) {
          counts[y[i]]++;
          left--;
        }
      }
      for (let i = 0; i < n && left > 0; i++) {
        if (d[i] === thr) {
          counts[y[i]]++;
          left--;
        }
      }
      if (isTie(counts)) {
        const order = [...d.keys()].sort((a, b) => d[a] - d[b] || a - b);
        winner[gy * G + gx] = vote(order, y, k, C).winner;
      } else {
        winner[gy * G + gx] = argmax(counts);
      }
    }
  }
  return winner;
}

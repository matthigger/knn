// Seeded toy data sets. Classification samples live in about [-1, 1]^2
// with labels 0..C-1; regression samples have x in [0, 1]. Every sample's
// feature is an array (length 2 or 1) so the k-NN code serves both.

/** Seeded PRNG (mulberry32) so a draw can be replayed. */
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draw (Box-Muller). */
function gauss(r) {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

// Each set draws one sample of class c; noise s in [0, 1] sets the spread.
const CLASS_SETS = {
  blobs: {
    name: "Blobs", C: 3,
    draw(r, c, s) {
      const a = Math.PI / 2 + 2 * Math.PI * c / 3, sd = 0.05 + 0.3 * s;
      return [0.55 * Math.cos(a) + sd * gauss(r),
        0.55 * Math.sin(a) + sd * gauss(r)];
    },
  },
  overlap: {
    name: "Overlap", C: 2,
    draw(r, c, s) {
      const sgn = c ? 1 : -1, sd = 0.15 + 0.4 * s;
      return [0.25 * sgn + sd * gauss(r), 0.1 * sgn + sd * gauss(r)];
    },
  },
  swiss: {
    name: "Swiss roll", C: 3,
    draw(r, c, s) {
      // Three interleaved spiral arms, 1.25 turns each. t = sqrt(u) keeps
      // the density even along an arm, whose length grows with radius.
      const t = Math.sqrt(r()), rad = 0.08 + 0.85 * t;
      const a = 2.5 * Math.PI * t + 2 * Math.PI * c / 3;
      const sd = 0.01 + 0.12 * s;
      return [rad * Math.cos(a) + sd * gauss(r),
        rad * Math.sin(a) + sd * gauss(r)];
    },
  },
  moons: {
    name: "Moons", C: 2,
    draw(r, c, s) {
      const a = Math.PI * r(), sd = 0.02 + 0.22 * s;
      const x = c ? 1 - Math.cos(a) : Math.cos(a);
      const y = c ? 0.5 - Math.sin(a) : Math.sin(a);
      return [0.6 * (x - 0.5) + sd * gauss(r),
        0.6 * (y - 0.25) + sd * gauss(r)];
    },
  },
  rings: {
    name: "Rings", C: 2,
    draw(r, c, s) {
      const a = 2 * Math.PI * r(), sd = 0.02 + 0.15 * s;
      const rad = c ? 0.75 : 0.32;
      return [rad * Math.cos(a) + sd * gauss(r),
        rad * Math.sin(a) + sd * gauss(r)];
    },
  },
};

// True curve f(x); samples are y = f(x) + noise with sd 0.05 + 0.5 s.
const REG_SETS = {
  sine: { name: "Sine", f: x => Math.sin(2 * Math.PI * x) },
  step: { name: "Step", f: x => (x < 0.5 ? -0.6 : 0.6) },
  linear: { name: "Linear", f: x => 2 * x - 1 },
  chirp: {
    name: "Chirp",
    f: x => Math.sin(2 * Math.PI * (x + 2.5 * x * x)),
  },
};

/**
 * Draw nPer samples of each class.
 *
 * Returns:
 *   {X, y}: X (n, 2) features, y (n,) labels, n = C nPer, grouped by class
 */
function makeClass(key, nPer, s, seed) {
  const set = CLASS_SETS[key], r = rng(seed), X = [], y = [];
  for (let c = 0; c < set.C; c++) {
    for (let i = 0; i < nPer; i++) {
      X.push(set.draw(r, c, s));
      y.push(c);
    }
  }
  return { X, y };
}

/**
 * Draw n regression samples, x uniform on [0, 1].
 *
 * Returns:
 *   {X, y}: X (n, 1) features, y (n,) labels
 */
function makeReg(key, n, s, seed) {
  const f = REG_SETS[key].f, r = rng(seed), X = [], y = [];
  for (let i = 0; i < n; i++) {
    const x = r();
    X.push([x]);
    y.push(f(x) + (0.05 + 0.5 * s) * gauss(r));
  }
  return { X, y };
}

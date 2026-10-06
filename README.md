# knn

Interactive teaching demo of k nearest neighbors, in the style of the
[CS231n k-NN demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/).
Drag a training sample to move it and watch the estimate update; turn on
the query point to see its k neighbors and their vote. Two tabs:

- **Classification.** Points in the plane, shaded by the k-NN estimate
  everywhere. Data sets: Blobs, Overlap,
  Swiss roll (three interleaved spiral arms), Moons, Rings. Distance L2 or
  L1 (the neighborhood draws as a circle or a diamond). A tie drops the
  farthest neighbor and votes again (k-1 NN) until it breaks.
- **Regression.** y against x, with the k-NN curve (average of the k
  nearest labels) over the true f(x). Data sets: Sine, Step, Linear, Chirp.
  The plot runs past [0, 1] to show k-NN cannot extrapolate.

Controls: k (1 to n), samples, noise, resample, and two toggles, both off by
default. Show query adds a draggable query point with its neighbors and a
readout. Cross validate holds out a share of the samples (Test slider, 20%
by default; per class for classification) as hollow test samples, and shows
the training and testing error plus a chart of both against k (log scale)
that marks the k with the lowest testing error; click or drag the chart to
set k. Arrow keys step k. "Try this" prompts for each tab sit above the
demo.

Plain HTML/CSS/JS with SVG: no build step and no dependencies.

## Run locally

Open `index.html` in a browser, or serve it:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

Link straight to a tab with `index.html#classification` or
`index.html#regression`.

## Publish on GitHub Pages

Pages is built by `.github/workflows/pages.yml` on every push to `main`. It
stamps `js/version.js` with the commit and build time, which the footer
shows ("local copy" when run locally). One-time setup:

```sh
gh repo create matthigger/knn --public --source . --push
gh api -X POST repos/matthigger/knn/pages -f build_type=workflow
```

The site then lives at <https://matthigger.github.io/knn/>.

## Layout

| file | role |
|------|------|
| `index.html` | page, explanation text, controls |
| `style.css` | layout, class palette, plot styles |
| `js/data.js` | seeded data sets (classification and regression) |
| `js/knn.js` | distances, vote, average, error at every k, region grid |
| `js/app.js` | state, drawing, dragging, controls, footer build stamp |
| `js/version.js` | build stamp, overwritten at deploy |
| `.github/workflows/pages.yml` | Pages deploy |

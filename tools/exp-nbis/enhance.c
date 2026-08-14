/* Hong/Wan/Jain style Gabor enhancement, plain C, no deps.
 * usage: enhance in.pgm out.pgm [upscale_factor] [sigma] [force_freq]
 * Prints the estimated ridge period.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *read_pgm(const char *p, int *w, int *h)
{
  FILE *f = fopen(p, "rb"); char m[3]; int mx; unsigned char *d;
  if (!f) { perror(p); exit(1); }
  if (fscanf(f, "%2s %d %d %d", m, w, h, &mx) != 4) exit(1);
  fgetc(f);
  d = malloc(*w * *h);
  if (fread(d, 1, *w * *h, f) != (size_t)(*w * *h)) exit(1);
  fclose(f); return d;
}
static void write_pgm(const char *p, unsigned char *d, int w, int h)
{
  FILE *f = fopen(p, "wb");
  fprintf(f, "P5\n%d %d\n255\n", w, h);
  fwrite(d, 1, w * h, f); fclose(f);
}
static unsigned char *upscale_bilinear(const unsigned char *s, int w, int h, int k, int *ow, int *oh)
{
  int nw = w * k, nh = h * k, x, y;
  unsigned char *d = malloc(nw * nh);
  for (y = 0; y < nh; y++) for (x = 0; x < nw; x++) {
    double sx = (x + 0.5) / k - 0.5, sy = (y + 0.5) / k - 0.5;
    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    double fx = sx - x0, fy = sy - y0; int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 > w - 1) x1 = w - 1; if (y1 > h - 1) y1 = h - 1;
    if (x0 > w - 1) x0 = w - 1; if (y0 > h - 1) y0 = h - 1;
    d[y * nw + x] = (unsigned char)lrint(
      s[y0*w+x0]*(1-fx)*(1-fy) + s[y0*w+x1]*fx*(1-fy) +
      s[y1*w+x0]*(1-fx)*fy     + s[y1*w+x1]*fx*fy);
  }
  *ow = nw; *oh = nh; return d;
}

int main(int argc, char **argv)
{
  int w, h, i, x, y, k = 1;
  double sigma = 4.0, forcef = 0.0;
  unsigned char *in;
  double *g, *ori, *out, mean = 0, var = 0;
  const double M0 = 128.0, V0 = 100.0;

  if (argc < 3) { fprintf(stderr, "usage: enhance in.pgm out.pgm [k] [sigma] [freq]\n"); return 1; }
  if (argc > 3) k = atoi(argv[3]);
  if (argc > 4) sigma = atof(argv[4]);
  if (argc > 5) forcef = atof(argv[5]);

  in = read_pgm(argv[1], &w, &h);
  if (k > 1) { int nw, nh; unsigned char *u = upscale_bilinear(in, w, h, k, &nw, &nh); free(in); in = u; w = nw; h = nh; }
  g = malloc(sizeof(double) * w * h);
  out = malloc(sizeof(double) * w * h);
  ori = malloc(sizeof(double) * w * h);

  /* ---- 1. global normalisation to mean M0 / var V0 ---- */
  for (i = 0; i < w * h; i++) mean += in[i];
  mean /= w * h;
  for (i = 0; i < w * h; i++) var += (in[i] - mean) * (in[i] - mean);
  var /= w * h;
  for (i = 0; i < w * h; i++) {
    double d = sqrt(V0 * (in[i] - mean) * (in[i] - mean) / var);
    g[i] = in[i] > mean ? M0 + d : M0 - d;
  }

  /* ---- 2. orientation field via gradient covariance, block 8, smoothed ---- */
  {
    int bs = 8, bw = (w + bs - 1) / bs, bh = (h + bs - 1) / bs, bx, by;
    double *vx = calloc(bw * bh, sizeof(double)), *vy = calloc(bw * bh, sizeof(double));
    double *sx_ = calloc(bw * bh, sizeof(double)), *sy_ = calloc(bw * bh, sizeof(double));
    for (by = 0; by < bh; by++) for (bx = 0; bx < bw; bx++) {
      double gxx = 0, gyy = 0, gxy = 0;
      for (y = by * bs; y < (by + 1) * bs && y < h; y++)
        for (x = bx * bs; x < (bx + 1) * bs && x < w; x++) {
          double gx, gy;
          int xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
          int ym = y > 0 ? y - 1 : 0, yp = y < h - 1 ? y + 1 : h - 1;
          gx = (g[y * w + xp] - g[y * w + xm]) / 2.0;
          gy = (g[yp * w + x] - g[ym * w + x]) / 2.0;
          gxx += gx * gx; gyy += gy * gy; gxy += gx * gy;
        }
      vx[by * bw + bx] = 2 * gxy;
      vy[by * bw + bx] = gxx - gyy;
    }
    /* smooth the doubled-angle vector field with a 5x5 box */
    for (by = 0; by < bh; by++) for (bx = 0; bx < bw; bx++) {
      double ax = 0, ay = 0; int n = 0, dy2, dx2;
      for (dy2 = -2; dy2 <= 2; dy2++) for (dx2 = -2; dx2 <= 2; dx2++) {
        int nx = bx + dx2, ny = by + dy2;
        if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
        ax += vx[ny * bw + nx]; ay += vy[ny * bw + nx]; n++;
      }
      sx_[by * bw + bx] = ax; sy_[by * bw + bx] = ay;
    }
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
      int b = (y / bs) * bw + (x / bs);
      ori[y * w + x] = 0.5 * atan2(sx_[b], sy_[b]);   /* ridge direction */
    }
    free(vx); free(vy); free(sx_); free(sy_);
  }

  /* ---- 3. global ridge period from x-signature projected across ridges ---- */
  double period = 0;
  {
    /* average orientation */
    double ax = 0, ay = 0, theta;
    for (i = 0; i < w * h; i++) { ax += sin(2 * ori[i]); ay += cos(2 * ori[i]); }
    theta = 0.5 * atan2(ax, ay);
    /* 1-D autocorrelation of the signal sampled perpendicular to the ridges */
    double best = -1e30; int bestlag = 0, lag, seen_neg = 0;
    double nx = cos(theta), ny = -sin(theta);   /* normal to ridge dir */
    fprintf(stderr, "autocorr:");
    for (lag = 1; lag <= 30; lag++) {
      double s = 0; long n = 0;
      for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
        int x2 = (int)lrint(x + nx * lag), y2 = (int)lrint(y + ny * lag);
        if (x2 < 0 || y2 < 0 || x2 >= w || y2 >= h) continue;
        s += (g[y * w + x] - M0) * (g[y2 * w + x2] - M0); n++;
      }
      if (n < 100) break;
      fprintf(stderr, " %d:%.0f", lag, s / n);
      if (s / n < 0) seen_neg = 1;
      if (seen_neg && s / n > best) { best = s / n; bestlag = lag; }
    }
    fprintf(stderr, "\n");
    period = bestlag ? bestlag : 8;
    fprintf(stderr, "image %dx%d  mean=%.1f sd=%.1f  ridge angle=%.1fdeg  est. period=%d px\n",
            w, h, mean, sqrt(var), theta * 180 / M_PI, bestlag);
  }
  double freq = forcef > 0 ? forcef : 1.0 / period;

  /* ---- 4. Gabor filtering ---- */
  {
    int half = (int)ceil(3 * sigma); if (half > 16) half = 16;
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
      double t = ori[y * w + x], ct = cos(t), st = sin(t), sum = 0, wsum = 0;
      int dx2, dy2;
      for (dy2 = -half; dy2 <= half; dy2++) for (dx2 = -half; dx2 <= half; dx2++) {
        int nx = x + dx2, ny = y + dy2;
        double xp, yp, kern;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        /* rotate so that xp runs across the ridges */
        xp =  dx2 * ct + dy2 * st;   /* wait: t is ridge direction */
        yp = -dx2 * st + dy2 * ct;
        /* xp along ridge, yp across ridge */
        kern = exp(-0.5 * (yp * yp / (sigma * sigma) + xp * xp / (sigma * sigma)))
               * cos(2 * M_PI * freq * yp);
        sum += kern * (g[ny * w + nx] - M0);
        wsum += fabs(kern);
      }
      out[y * w + x] = wsum > 0 ? sum / wsum : 0;
    }
  }

  /* ---- 5. rescale to 0..255 with percentile clipping ---- */
  {
    double lo, hi; int n = w * h;
    double *tmp = malloc(sizeof(double) * n);
    memcpy(tmp, out, sizeof(double) * n);
    int cmp(const void *a, const void *b) { double d = *(double*)a - *(double*)b; return d<0?-1:(d>0?1:0); }
    qsort(tmp, n, sizeof(double), cmp);
    lo = tmp[n / 50]; hi = tmp[n - 1 - n / 50];
    unsigned char *o8 = malloc(n);
    for (i = 0; i < n; i++) {
      double v = (out[i] - lo) / (hi - lo + 1e-9) * 255.0;
      o8[i] = v < 0 ? 0 : (v > 255 ? 255 : (unsigned char)lrint(v));
    }
    write_pgm(argv[2], o8, w, h);
    free(tmp); free(o8);
  }
  return 0;
}

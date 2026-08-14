/* high-pass then directional autocorrelation, to find the ridge period */
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

int main(int argc, char **argv)
{
  int w, h, x, y, r = argc > 2 ? atoi(argv[2]) : 7;
  unsigned char *in = read_pgm(argv[1], &w, &h);
  double *g = malloc(sizeof(double) * w * h);
  int a;

  /* high-pass: subtract local box mean of radius r */
  for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
    double s = 0; int n = 0, dx, dy;
    for (dy = -r; dy <= r; dy++) for (dx = -r; dx <= r; dx++) {
      int nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      s += in[ny * w + nx]; n++;
    }
    g[y * w + x] = in[y * w + x] - s / n;
  }

  printf("%s %dx%d  hp radius %d\n", argv[1], w, h, r);
  for (a = 0; a < 12; a++) {
    double th = a * M_PI / 12.0, nx = cos(th), ny = sin(th);
    int lag; double c0 = 0; long n0 = 0;
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) { c0 += g[y*w+x]*g[y*w+x]; n0++; }
    c0 /= n0;
    printf("  dir %3.0fdeg:", th * 180 / M_PI);
    for (lag = 1; lag <= 16; lag++) {
      double s = 0; long n = 0;
      for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
        int x2 = (int)lrint(x + nx * lag), y2 = (int)lrint(y + ny * lag);
        if (x2 < 0 || y2 < 0 || x2 >= w || y2 >= h) continue;
        s += g[y * w + x] * g[y2 * w + x2]; n++;
      }
      printf(" %+.2f", s / n / c0);
    }
    printf("\n");
  }
  return 0;
}

/* Standalone MINDTCT + BOZORTH3 harness against libfprint's bundled NBIS. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <nbis.h>

static unsigned char *read_pgm(const char *path, int *w, int *h)
{
  FILE *f = fopen(path, "rb");
  char magic[3];
  int maxv;
  unsigned char *d;
  if (!f) { perror(path); exit(1); }
  if (fscanf(f, "%2s %d %d %d", magic, w, h, &maxv) != 4) exit(1);
  fgetc(f);
  d = malloc(*w * *h);
  if (fread(d, 1, *w * *h, f) != (size_t)(*w * *h)) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f);
  return d;
}

/* nearest-neighbour integer upscale */
static unsigned char *upscale(const unsigned char *s, int w, int h, int f, int *ow, int *oh)
{
  int nw = w * f, nh = h * f, x, y;
  unsigned char *d = malloc(nw * nh);
  for (y = 0; y < nh; y++)
    for (x = 0; x < nw; x++)
      d[y * nw + x] = s[(y / f) * w + (x / f)];
  *ow = nw; *oh = nh;
  return d;
}

int main(int argc, char **argv)
{
  int w, h, factor = 1, partial = 0;
  unsigned char *img;
  MINUTIAE *minutiae = NULL;
  int *qmap = NULL, *dmap = NULL, *lcmap = NULL, *lfmap = NULL, *hcmap = NULL;
  int mw, mh, bw, bh, bd, r, i, valid = 0;
  unsigned char *bdata = NULL;
  LFSPARMS lfs;
  double ppmm = 19.685;

  if (argc < 2) { fprintf(stderr, "usage: mdt file.pgm [upscale] [partial] [ppmm]\n"); return 1; }
  if (argc > 2) factor = atoi(argv[2]);
  if (argc > 3) partial = atoi(argv[3]);
  if (argc > 4) ppmm = atof(argv[4]);

  img = read_pgm(argv[1], &w, &h);
  if (factor > 1) { unsigned char *n = upscale(img, w, h, factor, &w, &h); free(img); img = n; }

  lfs = g_lfsparms_V2;
  lfs.remove_perimeter_pts = partial ? TRUE : FALSE;

  r = get_minutiae(&minutiae, &qmap, &dmap, &lcmap, &lfmap, &hcmap,
                   &mw, &mh, &bdata, &bw, &bh, &bd,
                   img, w, h, 8, ppmm, &lfs);
  if (r) { printf("%s x%d partial=%d: get_minutiae FAILED code %d\n", argv[1], factor, partial, r); return 2; }

  for (i = 0; i < mw * mh; i++) if (dmap[i] >= 0) valid++;
  printf("%-22s %3dx%-3d x%d partial=%d ppmm=%.2f  map %dx%d validdir=%d/%d  minutiae=%d",
         argv[1], w, h, factor, partial, ppmm, mw, mh, valid, mw * mh, minutiae->num);
  if (minutiae->num) {
    double rsum = 0;
    for (i = 0; i < minutiae->num; i++) rsum += minutiae->list[i]->reliability;
    printf("  meanrel=%.3f", rsum / minutiae->num);
  }
  printf("%s\n", minutiae->num < 10 ? "   <-- BELOW BOZORTH MIN (10)" : "");
  return 0;
}

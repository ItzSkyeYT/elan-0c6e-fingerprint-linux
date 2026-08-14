/* extract minutiae from each pgm given, then all-pairs bozorth3 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <nbis.h>

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

static int sort_x_y_local(const void *a, const void *b)
{
  struct minutiae_struct *p = (struct minutiae_struct *) a;
  struct minutiae_struct *q = (struct minutiae_struct *) b;
  if (p->col[0] < q->col[0]) return -1;
  if (p->col[0] > q->col[0]) return 1;
  if (p->col[1] < q->col[1]) return -1;
  if (p->col[1] > q->col[1]) return 1;
  return 0;
}

static int extract(const char *path, struct xyt_struct *xyt)
{
  int w, h, mw, mh, bw, bh, bd, r, i, nmin;
  unsigned char *img, *bdata = NULL;
  int *qmap = NULL, *dmap = NULL, *lcmap = NULL, *lfmap = NULL, *hcmap = NULL;
  MINUTIAE *m = NULL;
  LFSPARMS lfs = g_lfsparms_V2;
  struct minutiae_struct c[1000];

  img = read_pgm(path, &w, &h);
  r = get_minutiae(&m, &qmap, &dmap, &lcmap, &lfmap, &hcmap, &mw, &mh,
                   &bdata, &bw, &bh, &bd, img, w, h, 8, 19.685, &lfs);
  if (r) return -1;
  nmin = m->num < MAX_BOZORTH_MINUTIAE ? m->num : MAX_BOZORTH_MINUTIAE;
  for (i = 0; i < nmin; i++) {
    lfs2nist_minutia_XYT(&c[i].col[0], &c[i].col[1], &c[i].col[2], m->list[i], w, h);
    c[i].col[3] = (int) (m->list[i]->reliability * 100.0);
    if (c[i].col[2] > 180) c[i].col[2] -= 360;
  }
  qsort(c, nmin, sizeof(struct minutiae_struct), sort_x_y_local);
  for (i = 0; i < nmin; i++) {
    xyt->xcol[i] = c[i].col[0];
    xyt->ycol[i] = c[i].col[1];
    xyt->thetacol[i] = c[i].col[2];
  }
  xyt->nrows = nmin;
  return nmin;
}

int main(int argc, char **argv)
{
  int n = argc - 1, i, j;
  struct xyt_struct *x = calloc(n, sizeof(struct xyt_struct));
  int *cnt = calloc(n, sizeof(int));

  for (i = 0; i < n; i++) {
    cnt[i] = extract(argv[i + 1], &x[i]);
    printf("%-40s minutiae=%d\n", argv[i + 1], cnt[i]);
  }
  printf("\npairwise bozorth3 scores:\n     ");
  for (j = 0; j < n; j++) printf("%5d", j);
  printf("\n");
  for (i = 0; i < n; i++) {
    int plen = bozorth_probe_init(&x[i]);
    printf("%3d: ", i);
    for (j = 0; j < n; j++)
      printf("%5d", bozorth_to_gallery(plen, &x[i], &x[j]));
    printf("\n");
  }
  return 0;
}

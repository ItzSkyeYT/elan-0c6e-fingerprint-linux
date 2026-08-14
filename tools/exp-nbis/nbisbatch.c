/* Batch MINDTCT + BOZORTH3 harness for the ELAN 0c6e press sensor dataset.
 *
 * usage: nbisbatch manifest.txt [ppmm] [rm_perimeter] [xytdir]
 *
 * manifest.txt: one PGM path per line.
 * stdout:
 *    M <idx> <nminutiae> <meanrel> <path>
 *    S <i> <j> <score>            (full matrix, including i==j)
 * If xytdir is given, writes <xytdir>/<idx>.xyt with "x y theta quality" rows.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <nbis.h>

#define MAXIMG 128

static unsigned char *read_pgm(const char *p, int *w, int *h)
{
  FILE *f = fopen(p, "rb");
  char m[3];
  int mx;
  unsigned char *d;
  if (!f) { perror(p); exit(1); }
  if (fscanf(f, "%2s %d %d %d", m, w, h, &mx) != 4) { fprintf(stderr, "%s: bad header\n", p); exit(1); }
  fgetc(f);
  d = malloc((size_t)*w * *h);
  if (fread(d, 1, (size_t)*w * *h, f) != (size_t)(*w * *h)) { fprintf(stderr, "%s: short read\n", p); exit(1); }
  fclose(f);
  return d;
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

static double g_ppmm = 19.685;
static int g_rmperim = 0;

static int extract(const char *path, struct xyt_struct *xyt, double *meanrel,
                   struct minutiae_struct *out, int *outn)
{
  int w, h, mw, mh, bw, bh, bd, r, i, nmin;
  unsigned char *img, *bdata = NULL;
  int *qmap = NULL, *dmap = NULL, *lcmap = NULL, *lfmap = NULL, *hcmap = NULL;
  MINUTIAE *m = NULL;
  LFSPARMS lfs = g_lfsparms_V2;
  static struct minutiae_struct c[MAX_BOZORTH_MINUTIAE + 8];
  double rsum = 0;

  lfs.remove_perimeter_pts = g_rmperim ? TRUE : FALSE;

  img = read_pgm(path, &w, &h);
  r = get_minutiae(&m, &qmap, &dmap, &lcmap, &lfmap, &hcmap, &mw, &mh,
                   &bdata, &bw, &bh, &bd, img, w, h, 8, g_ppmm, &lfs);
  free(img);
  if (r) return -1;

  nmin = m->num < MAX_BOZORTH_MINUTIAE ? m->num : MAX_BOZORTH_MINUTIAE;
  for (i = 0; i < nmin; i++) {
    lfs2nist_minutia_XYT(&c[i].col[0], &c[i].col[1], &c[i].col[2], m->list[i], w, h);
    c[i].col[3] = (int) (m->list[i]->reliability * 100.0);
    if (c[i].col[2] > 180) c[i].col[2] -= 360;
    rsum += m->list[i]->reliability;
  }
  qsort(c, nmin, sizeof(struct minutiae_struct), sort_x_y_local);
  for (i = 0; i < nmin; i++) {
    xyt->xcol[i] = c[i].col[0];
    xyt->ycol[i] = c[i].col[1];
    xyt->thetacol[i] = c[i].col[2];
    if (out) out[i] = c[i];
  }
  xyt->nrows = nmin;
  if (outn) *outn = nmin;
  *meanrel = nmin ? rsum / nmin : 0.0;

  free_minutiae(m);
  free(qmap); free(dmap); free(lcmap); free(lfmap); free(hcmap); free(bdata);
  return nmin;
}

int main(int argc, char **argv)
{
  char line[4096];
  static char paths[MAXIMG][4096];
  static struct xyt_struct x[MAXIMG];
  static struct minutiae_struct mlist[MAXIMG][MAX_BOZORTH_MINUTIAE + 8];
  static int mn[MAXIMG];
  static double mr[MAXIMG];
  int n = 0, i, j;
  FILE *mf;
  const char *xytdir = NULL;

  if (argc < 2) { fprintf(stderr, "usage: nbisbatch manifest [ppmm] [rmperim] [xytdir]\n"); return 1; }
  if (argc > 2) g_ppmm = atof(argv[2]);
  if (argc > 3) g_rmperim = atoi(argv[3]);
  if (argc > 4) xytdir = argv[4];

  mf = fopen(argv[1], "r");
  if (!mf) { perror(argv[1]); return 1; }
  while (fgets(line, sizeof line, mf)) {
    size_t l = strlen(line);
    while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
    if (!l) continue;
    if (n >= MAXIMG) { fprintf(stderr, "too many images\n"); return 1; }
    strncpy(paths[n], line, sizeof(paths[0]) - 1);
    n++;
  }
  fclose(mf);

  for (i = 0; i < n; i++) {
    int c = extract(paths[i], &x[i], &mr[i], mlist[i], &mn[i]);
    if (c < 0) { mn[i] = 0; x[i].nrows = 0; mr[i] = 0; }
    printf("M %d %d %.4f %s\n", i, x[i].nrows, mr[i], paths[i]);
    if (xytdir) {
      char fn[4200];
      FILE *o;
      snprintf(fn, sizeof fn, "%s/%d.xyt", xytdir, i);
      o = fopen(fn, "w");
      if (o) {
        for (j = 0; j < mn[i]; j++)
          fprintf(o, "%d %d %d %d\n", mlist[i][j].col[0], mlist[i][j].col[1],
                  mlist[i][j].col[2], mlist[i][j].col[3]);
        fclose(o);
      }
    }
  }

  for (i = 0; i < n; i++) {
    int plen = bozorth_probe_init(&x[i]);
    for (j = 0; j < n; j++)
      printf("S %d %d %d\n", i, j, bozorth_to_gallery(plen, &x[i], &x[j]));
  }
  fflush(stdout);
  return 0;
}

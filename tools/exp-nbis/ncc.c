#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static unsigned char *rp(const char*p,int*w,int*h){FILE*f=fopen(p,"rb");char m[3];int mx;unsigned char*d;
 fscanf(f,"%2s %d %d %d",m,w,h,&mx);fgetc(f);d=malloc(*w**h);fread(d,1,*w**h,f);fclose(f);return d;}
int main(int c,char**v){int w,h,w2,h2,i;unsigned char*a=rp(v[1],&w,&h),*b=rp(v[2],&w2,&h2);
 double ma=0,mb=0,n=0,da=0,db=0;int N=w*h;
 for(i=0;i<N;i++){ma+=a[i];mb+=b[i];}ma/=N;mb/=N;
 for(i=0;i<N;i++){double x=a[i]-ma,y=b[i]-mb;n+=x*y;da+=x*x;db+=y*y;}
 printf("%s vs %s  NCC(0,0)=%.4f  meanA=%.1f meanB=%.1f\n",v[1],v[2],n/sqrt(da*db),ma,mb);return 0;}

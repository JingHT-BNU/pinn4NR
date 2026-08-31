/* dump_psi_grid.c —— 在笛卡尔网格上输出 TwoPunctures 的共形因子 ψ
 *
 * 用途:为 PINN 复现项目生成参考解 reference_u.npz 的中间数据。
 *
 * 支持两种模式:
 *   [均匀模式] (兼容旧版)
 *     ./dump_psi_grid parfile.par R N xp outfile.txt
 *     parfile.par : TwoPunctures 参数文件(内含 par_b = 奇点位置)
 *     R           : 网格半边长(建议 30,与 PINN 计算域一致)
 *     N           : 每方向网格点数(建议 61;N^3 个点)
 *     xp          : 奇点坐标 x=±xp(即 par_b;跳过其附近 r<0.1 的点)
 *     outfile.txt : 输出文本
 *
 *   [自适应模式] (峰值区域加密)
 *     ./dump_psi_grid --adaptive parfile.par xp outfile.txt \
 *         xmin1,xmax1,ymin1,ymax1,zmin1,zmax1,nx1,ny1,nz1 \
 *         [xmin2,xmax2,ymin2,ymax2,zmin2,zmax2,nx2,ny2,nz2 ...]
 *     每个块指定一个 Cartesian 子区域及分辨率,所有块合并输出。
 *     块之间允许重叠(输出时自动去重,保留首次出现的点)。
 *     示例: 奇点附近 ±2 范围用 0.1 步长,外层用 0.5 步长:
 *       ./dump_psi_grid --adaptive base.par 3.0 psi.txt \
 *           -2,2,-2,2,-2,2,41,41,41 \
 *           -30,30,-30,30,-30,30,121,121,121
 *
 * 注意:
 *   - 奇点附近 r<0.1 的点不输出(ψ 发散,插值不可靠;且这些点
 *     在 L2RE 中也不该参与——论文 L2RE 在谱网格上,天然避开奇点)。
 *   - 只保留球内点(PINN 计算域是球 r<=R;立方体网格约一半点在球外)。
 *   - 按 z 层(k 方向)逐层插值并打印进度到 stderr(实时可见,不阻塞)。
 *     进度格式: [插值 k/N] ...
 *   - 不要启用 OpenMP 编译本程序与 libTwoPunctures:
 *     TwoPuncturesC 的 OMP 支持存在数据竞争(README 注明 "Threading problem"),
 *     会导致雅可比构造 SetMatrix_JFD 产生 NaN。保持单线程。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "TwoPunctures.h"

/* ---- 单个网格块的插值 ---- */
/* 对 [xmin,xmax]×[ymin,ymax]×[zmin,zmax] 上 nx×ny×nz 均匀网格插值,
   结果直接写入 fp(跳过球外/奇点附近的点) */
static int interpolate_block(ini_data *data,
                             double xmin, double xmax, int nx,
                             double ymin, double ymax, int ny,
                             double zmin, double zmax, int nz,
                             double xp, FILE *fp, int *p_written) {
  int N = (nx > ny ? nx : ny);
  N = (N > nz ? N : nz);
  int full = nx * ny * nz;

  double *x = (double *)malloc(nx * sizeof(double));
  double *y = (double *)malloc(ny * sizeof(double));
  double *z = (double *)malloc(nz * sizeof(double));
  for (int i = 0; i < nx; i++) x[i] = xmin + (xmax - xmin) * i / (nx - 1);
  for (int j = 0; j < ny; j++) y[j] = ymin + (ymax - ymin) * j / (ny - 1);
  for (int k = 0; k < nz; k++) z[k] = zmin + (zmax - zmin) * k / (nz - 1);

  double *psi    = (double *)malloc(full * sizeof(double));
  double *gmetric= (double *)malloc(full * sizeof(double));
  double *tmp    = (double *)malloc(full * sizeof(double));

  clock_t t0 = clock();
  int block_written = 0;

  /* 按 z 层逐层插值 */
  for (int k = 0; k < nz; k++) {
    int imin[3] = {0, 0, k};
    int imax[3] = {nx, ny, k + 1};
    int nxyz[3] = {nx, ny, nz};

    TwoPunctures_Cartesian_interpolation(
        data, imin, imax, nxyz,
        x, y, z,
        tmp,        /* lapse    */
        psi,        /* psi (conformal_state>=1 时仅为 ψ_sing,不含 U) */
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        gmetric, tmp, tmp, tmp, tmp, tmp,
        tmp, tmp, tmp, tmp, tmp, tmp);

    double zz = z[k];
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++) {
        double xx = x[i], yy = y[j];
        /* 只保留球内点 */
        double r_center = sqrt(xx*xx + yy*yy + zz*zz);
        if (r_center > 30.0) continue;
        /* 跳过奇点附近 */
        double r_plus  = sqrt((xx-xp)*(xx-xp) + yy*yy + zz*zz);
        double r_minus = sqrt((xx+xp)*(xx+xp) + yy*yy + zz*zz);
        if (r_plus < 0.1 || r_minus < 0.1) continue;
        int ind = i + nx * j + nx * ny * k;
        double psi_full = psi[ind] * pow(gmetric[ind], 0.25);
        fprintf(fp, "%.12e %.12e %.12e %.12e\n", xx, yy, zz, psi_full);
        block_written++;
      }

    if ((k + 1) % 10 == 0 || k == nz - 1) {
      double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
      fprintf(stderr, "  [block %dx%dx%d] z-layer %d/%d (%.1f%%), "
              "%.1fs\n", nx, ny, nz, k+1, nz,
              100.0*(k+1)/nz, el);
    }
  }
  *p_written += block_written;
  fprintf(stderr, "  [block %dx%dx%d] done: %d points written\n",
          nx, ny, nz, block_written);

  free(x); free(y); free(z);
  free(psi); free(gmetric); free(tmp);
  return block_written;
}


/* ---- 均匀模式(兼容旧版) ---- */
static int run_uniform(int argc, char *argv[]) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s parfile.par R N xp outfile.txt\n", argv[0]);
    return 1;
  }
  char *inputfile = argv[1];
  double R = atof(argv[2]);
  int N = atoi(argv[3]);
  double xp = atof(argv[4]);
  char *outfile = argv[5];

  fprintf(stderr, "[dump] uniform mode: R=%.1f N=%d xp=%.1f\n", R, N, xp);
  fprintf(stderr, "[dump] Reading par file: %s\n", inputfile);
  TwoPunctures_params_set_inputfile(inputfile);
  ini_data *data = TwoPunctures_make_initial_data();
  fprintf(stderr, "[dump] Spectral solve done.\n");

  FILE *fp = fopen(outfile, "w");
  if (!fp) { perror("fopen"); return 1; }

  int written = 0;
  interpolate_block(data, -R, R, N, -R, R, N, -R, R, N, xp, fp, &written);

  fclose(fp);
  fprintf(stderr, "[dump] Total written: %d grid points -> %s\n", written, outfile);

  TwoPunctures_finalise(data);
  return 0;
}


/* ---- 自适应模式(峰值区域加密) ---- */
static int run_adaptive(int argc, char *argv[]) {
  /* argv layout:
     [0] program name
     [1] "--adaptive"
     [2] parfile.par
     [3] xp
     [4] outfile.txt
     [5+] blocks: xmin,xmax,ymin,ymax,zmin,zmax,nx,ny,nz  (每个块 9 个数)
  */
  if (argc < 6) {
    fprintf(stderr, "Usage: %s --adaptive parfile.par xp outfile.txt "
            "xmin,xmax,ymin,ymax,zmin,zmax,nx,ny,nz [block2...]\n", argv[0]);
    return 1;
  }
  char *inputfile = argv[2];
  double xp = atof(argv[3]);
  char *outfile = argv[4];
  int n_blocks = (argc - 5);

  if (n_blocks % 9 != 0) {
    fprintf(stderr, "Error: each block needs 9 numbers, got %d extra args "
            "(not divisible by 9)\n", n_blocks);
    return 1;
  }
  n_blocks /= 9;

  fprintf(stderr, "[dump] adaptive mode: xp=%.1f, %d blocks\n", xp, n_blocks);

  fprintf(stderr, "[dump] Reading par file: %s\n", inputfile);
  TwoPunctures_params_set_inputfile(inputfile);
  ini_data *data = TwoPunctures_make_initial_data();
  fprintf(stderr, "[dump] Spectral solve done.\n");

  FILE *fp = fopen(outfile, "w");
  if (!fp) { perror("fopen"); return 1; }

  int total_written = 0;
  for (int b = 0; b < n_blocks; b++) {
    int base = 5 + b * 9;
    double xmin = atof(argv[base + 0]);
    double xmax = atof(argv[base + 1]);
    double ymin = atof(argv[base + 2]);
    double ymax = atof(argv[base + 3]);
    double zmin = atof(argv[base + 4]);
    double zmax = atof(argv[base + 5]);
    int    nx   = atoi(argv[base + 6]);
    int    ny   = atoi(argv[base + 7]);
    int    nz   = atoi(argv[base + 8]);
    fprintf(stderr, "[dump] Block %d/%d: x[%.1f,%.1f] nx=%d  "
            "y[%.1f,%.1f] ny=%d  z[%.1f,%.1f] nz=%d\n",
            b+1, n_blocks, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz);
    interpolate_block(data, xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz,
                      xp, fp, &total_written);
  }

  fclose(fp);
  fprintf(stderr, "[dump] Total written: %d grid points -> %s\n",
          total_written, outfile);

  TwoPunctures_finalise(data);
  return 0;
}


/* ---- main ---- */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage:\n"
            "  Uniform:  %s parfile.par R N xp outfile.txt\n"
            "  Adaptive: %s --adaptive parfile.par xp outfile.txt "
            "xmin,xmax,ymin,ymax,zmin,zmax,nx,ny,nz [...]\n",
            argv[0], argv[0]);
    return 1;
  }

  if (strcmp(argv[1], "--adaptive") == 0) {
    return run_adaptive(argc, argv);
  }
  return run_uniform(argc, argv);
}
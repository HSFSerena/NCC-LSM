// ============================================================================
//  摄影测量学（数字部分）编程题 1
//  相关系数匹配 + 最小二乘匹配（LSM）精化
//
//  编译（Release x64，可直接运行文件夹里的 build.bat，等价的命令是）：
//      cl /O2 /MT /EHsc /I "C:\opencv\04_opencv_4100\opencv\build\include" ^
//         matching.cpp /Fe:Matching.exe ^
//         /link /LIBPATH:"C:\opencv\04_opencv_4100\opencv\build\x64\vc16\lib" ^
//         opencv_world4100.lib
//
//  注意：exe 运行时需要同文件夹里的 opencv_world4100.dll，
//  提交时把 exe 和这个 dll 放在一起即可。
//
//  用法（三种都行）：
//      1) 双击 Matching.exe，按提示输入影像路径和参数，回车表示用默认值
//      2) 把左、右两张影像一起拖到 Matching.exe 图标上
//      3) 命令行：Matching.exe 左影像 右影像
//
//  输出：tiepoints.csv（同名点坐标）  result.bmp（匹配结果图）
// ============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
using namespace std;

// OpenCV 只用来做一件事：把 bmp / jpg / png 文件读成灰度图（imread）。
// png、jpg 是压缩格式，必须解码才能拿到像素；金字塔、相关系数、
// 最小二乘、画结果图这些算法全部用下面的代码自己实现。
#include <opencv2/opencv.hpp>

// ============================================================================
//  一、数据结构
// ============================================================================

// 灰度影像：宽 w、高 h，像素按行存放，pix[y*w + x] 就是 (x,y) 处的灰度
struct GrayImage
{
    int w;
    int h;
    vector<unsigned char> pix;

    GrayImage() { w = 0; h = 0; }

    // 取 (x,y) 处的灰度
    unsigned char at(int x, int y) const { return pix[y * w + x]; }
};

// 彩色影像，只用来画结果图（每个像素 3 个字节：R、G、B）
struct RgbImage
{
    int w;
    int h;
    vector<unsigned char> pix;

    RgbImage() { w = 0; h = 0; }

    void set(int x, int y, unsigned char r, unsigned char g, unsigned char b)
    {
        if (x < 0 || y < 0 || x >= w || y >= h) return;   // 越界就不画
        int i = (y * w + x) * 3;
        pix[i] = r; pix[i + 1] = g; pix[i + 2] = b;
    }
};

// 一个同名点
struct TiePoint
{
    double xl, yl;        // 左影像坐标（格网点）
    double xr, yr;        // 右影像坐标（相关系数匹配的结果，整像素）
    double ncc0;          // 相关系数

    double xrL, yrL;      // 最小二乘精化后的右影像坐标（亚像素）
    double nccL;          // 精化后重新算的相关系数
    int    iter;          // 迭代次数
    bool   lsmOk;         // 精化是否成功

    bool   keep;          // 是否通过粗差检查

    TiePoint()
    {
        xl = yl = xr = yr = ncc0 = 0;
        xrL = yrL = nccL = 0;
        iter = 0;
        lsmOk = false;
        keep = true;
    }
};

// 最小二乘匹配的结果
struct LsmResult
{
    double a[3];          // 几何变形参数：x' = a0 + a1*u + a2*v
    double b[3];          //                 y' = b0 + b1*u + b2*v
    double h0, h1;        // 辐射变形参数：g = h0 + h1*(f - f均值)
    double xr, yr;        // 同名点坐标（u=v=0 时的 x'、y'）
    double ncc;           // 精化后的相关系数
    int    iter;          // 实际迭代次数
    bool   ok;            // 是否成功

    LsmResult()
    {
        a[0] = b[0] = 0; a[1] = b[2] = 1; a[2] = b[1] = 0;
        h0 = 0; h1 = 1;
        xr = yr = ncc = 0;
        iter = 0;
        ok = false;
    }
};

// 粗定位的结果：整体位移（右影像相对左影像平移了多少像素）
struct ShiftResult
{
    int dx, dy;
    double ncc;

    ShiftResult() { dx = 0; dy = 0; ncc = -2.0; }
};

// ============================================================================
//  二、读图 / 存图
// ============================================================================

// 读入一张影像（bmp / jpg / png 都行），转成 8 位灰度。
// 压缩格式的解码交给 OpenCV 的 imread（第二个参数 IMREAD_GRAYSCALE
// 表示读进来直接转灰度），像素拷到我们自己的数组里，后面的算法
// 全部用自己的代码实现。
bool readImage(const char* filename, GrayImage& img)
{
    cv::Mat m = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (m.empty()) return false;                  // 文件不存在或格式不支持

    img.w = m.cols;
    img.h = m.rows;
    img.pix.resize(img.w * img.h);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            img.pix[y * img.w + x] = m.at<unsigned char>(y, x);
    return true;
}

// ---- 存 24 位 BMP（结果图用）----
// BMP 格式没有压缩，文件里就是一行行的像素，写起来最简单，
// 任何看图软件都能打开。文件开头是两个头，照着 BMP 格式说明写：

// 文件头（14 字节）：文件类型、大小、像素数据从第几字节开始
#pragma pack(push, 1)                             // 这两个头必须一个字节都不多
struct BmpFileHead
{
    unsigned short type;      // "BM" 两个字符
    unsigned int   size;      // 整个文件的字节数
    unsigned short r1, r2;    // 保留位，填 0
    unsigned int   offset;    // 像素数据离文件开头的字节数（14 + 40 = 54）
};

// 信息头（40 字节）：图像的宽、高、每个像素几个比特等
struct BmpInfoHead
{
    unsigned int   size;      // 本头的字节数（40）
    int            width;     // 图像宽（像素）
    int            height;    // 图像高（正数表示像素自下而上存放）
    unsigned short planes;    // 固定填 1
    unsigned short bits;      // 每像素比特数，这里用 24（真彩色）
    unsigned int   comp;      // 压缩方式，0 表示不压缩
    unsigned int   imgSize;   // 像素数据的字节数
    int            xppm, yppm;       // 分辨率，随便填
    unsigned int   clrUsed, clrImp;  // 调色板，24 位图用不到，填 0
};
#pragma pack(pop)

bool saveBmp(const char* filename, const RgbImage& img)
{
    if (img.w <= 0 || img.h <= 0) return false;

    int rowBytes = (img.w * 3 + 3) / 4 * 4;       // 每行字节数按 4 对齐
    int dataSize = rowBytes * img.h;

    BmpFileHead fh;                               // 文件头
    fh.type = 0x4D42;                             // 0x42='B'，0x4D='M'
    fh.size = 54 + dataSize;
    fh.r1 = 0; fh.r2 = 0;
    fh.offset = 54;

    BmpInfoHead ih;                               // 信息头
    ih.size = 40;
    ih.width = img.w;
    ih.height = img.h;
    ih.planes = 1;
    ih.bits = 24;
    ih.comp = 0;
    ih.imgSize = dataSize;
    ih.xppm = 0; ih.yppm = 0;
    ih.clrUsed = 0; ih.clrImp = 0;

    ofstream out(filename, ios::binary);
    if (!out) return false;

    out.write((char*)&fh, sizeof(fh));
    out.write((char*)&ih, sizeof(ih));

    vector<unsigned char> row(rowBytes, 0);
    for (int y = img.h - 1; y >= 0; --y) {        // BMP 从最后一行开始存
        for (int x = 0; x < img.w; ++x) {
            int i = (y * img.w + x) * 3;
            row[x * 3 + 0] = img.pix[i + 2];      // BMP 里是 B、G、R 的顺序
            row[x * 3 + 1] = img.pix[i + 1];
            row[x * 3 + 2] = img.pix[i + 0];
        }
        out.write((char*)row.data(), rowBytes);
    }
    return true;
}

// ============================================================================
//  三、找整体位移（粗定位）
// ============================================================================
// 两张立体影像一般整体错开了几十到几百个像素（本实验的矿区像对约 400 像素），
// 如果每个点都在 ±400 像素的范围内搜索，太慢了。所以先做一次粗定位：
//   把两张图都缩小 4 倍，在缩小图上把右影像上下左右滑动，每滑到一个位置
//   就把两张图重叠区域的相关系数算一遍，相关系数最大的位置就是整体位移。

// 缩小：把 factor x factor 的方块平均成 1 个像素
GrayImage shrink(const GrayImage& im, int factor)
{
    GrayImage d;
    d.w = im.w / factor;
    d.h = im.h / factor;
    d.pix.assign(d.w * d.h, 0);

    for (int y = 0; y < d.h; ++y) {
        for (int x = 0; x < d.w; ++x) {
            int sum = 0;
            for (int v = 0; v < factor; ++v)
                for (int u = 0; u < factor; ++u)
                    sum += im.at(x * factor + u, y * factor + v);
            d.pix[y * d.w + x] = (unsigned char)(sum / (factor * factor));
        }
    }
    return d;
}

// 整幅影像重叠区域的相关系数（粗定位时用来比较"平移多少像素"最合适）。
// 两张图错开 (dx,dy) 后重叠的部分求相关系数；sampleStep 是抽样间隔，
// 隔几个像素取一个，省时间。
double globalNcc(const GrayImage& A, const GrayImage& B, int dx, int dy,
                 int sampleStep, int& count)
{
    // B 相对 A 平移 (dx,dy) 以后，两者重叠的范围
    int x0 = 0, y0 = 0, x1 = A.w, y1 = A.h;
    if (-dx > x0) x0 = -dx;
    if (-dy > y0) y0 = -dy;
    if (B.w - dx < x1) x1 = B.w - dx;
    if (B.h - dy < y1) y1 = B.h - dy;

    double sf = 0, sg = 0, sff = 0, sgg = 0, sfg = 0;
    int n = 0;

    for (int y = y0; y < y1; y += sampleStep) {
        for (int x = x0; x < x1; x += sampleStep) {
            double f = A.at(x, y);
            double g = B.at(x + dx, y + dy);
            sf += f;   sg += g;
            sff += f * f;
            sgg += g * g;
            sfg += f * g;
            ++n;
        }
    }

    count = n;
    if (n < 200) return -2.0;                     // 重叠太少，结果不可信

    double num = sfg - sf * sg / n;               // Σ(f-f均值)(g-g均值)
    double d1  = sff - sf * sf / n;               // Σ(f-f均值)^2
    double d2  = sgg - sg * sg / n;               // Σ(g-g均值)^2
    if (d1 < 1e-6 || d2 < 1e-6) return -2.0;

    return num / sqrt(d1 * d2);
}

// 在 ±radX、±radY 的范围内穷举整数位移，找相关系数最大的那个
ShiftResult searchShift(const GrayImage& A, const GrayImage& B,
                        int radX, int radY, int sampleStep)
{
    ShiftResult best;

    for (int dy = -radY; dy <= radY; ++dy) {
        for (int dx = -radX; dx <= radX; ++dx) {
            int cnt = 0;
            double c = globalNcc(A, B, dx, dy, sampleStep, cnt);
            if (c > best.ncc) {
                best.ncc = c;
                best.dx = dx;
                best.dy = dy;
            }
        }
    }
    return best;
}

// ============================================================================
//  四、相关系数匹配（逐点搜索）
// ============================================================================

// 计算左影像以 (lx,ly) 为中心、右影像以 (cx,cy) 为中心的两个窗口的相关系数。
// 窗口里几乎没有灰度变化（纯色）时没有信息，返回 -2 表示无效。
double zncc(const GrayImage& F, int lx, int ly, int r,
            const GrayImage& G, int cx, int cy)
{
    // 右影像窗口越界就直接判无效
    if (cx - r < 0 || cy - r < 0 || cx + r >= G.w || cy + r >= G.h) return -2.0;

    double sf = 0, sg = 0;                        // Σf、Σg
    double sff = 0, sgg = 0, sfg = 0;             // Σf^2、Σg^2、Σfg
    int n = 0;

    for (int v = -r; v <= r; ++v) {
        for (int u = -r; u <= r; ++u) {
            double f = F.at(lx + u, ly + v);
            double g = G.at(cx + u, cy + v);
            sf += f;   sg += g;
            sff += f * f;
            sgg += g * g;
            sfg += f * g;
            ++n;
        }
    }

    double num = sfg - sf * sg / n;               // Σ(f-f均值)(g-g均值)
    double d1  = sff - sf * sf / n;               // Σ(f-f均值)^2
    double d2  = sgg - sg * sg / n;               // Σ(g-g均值)^2
    if (d1 < 1e-6 || d2 < 1e-6) return -2.0;      // 纯色窗口，没有纹理

    return num / sqrt(d1 * d2);
}

// 在右影像里，以预测位置 (px,py) 为中心、半径 sr 的方框内搜索左点 (lx,ly) 的同名点。
// 相关系数只在整像素位置上一个个算，峰值所在的像素就是匹配结果；
// 更精确的亚像素位置交给第五节的最小二乘精化去求。
bool matchByNcc(const GrayImage& F, const GrayImage& G,
                int lx, int ly, int r, int sr,
                int px, int py, double thresh,
                double& outX, double& outY, double& outNcc)
{
    double best = -2.0;
    int bx = 0, by = 0;

    for (int j = -sr; j <= sr; ++j) {
        for (int i = -sr; i <= sr; ++i) {
            double c = zncc(F, lx, ly, r, G, px + i, py + j);
            if (c > best) {
                best = c;
                bx = px + i;
                by = py + j;
            }
        }
    }

    if (best < thresh) return false;              // 相关系数不够，认为没匹配上
    // 峰值落在搜索范围的最边上，说明真正的峰可能在范围之外，不可信
    if (bx == px - sr || bx == px + sr || by == py - sr || by == py + sr)
        return false;

    outX = bx;
    outY = by;
    outNcc = best;
    return true;
}

// ============================================================================
//  五、最小二乘匹配（LSM）
// ============================================================================
//
// 模型（在一个窗口内）：
//   几何变形  x' = a0 + a1*u + a2*v      u、v 是模板窗口内的局部坐标
//             y' = b0 + b1*u + b2*v      （以格网点为原点）
//   辐射变形  g(x',y') = h0 + h1·(f(u,v) - f均值)
//
// 误差方程  v = h0 + h1·fc - g(x',y')     fc = f - f均值
// 未知数    6 个几何参数 a0,a1,a2,b0,b1,b2，加 2 个辐射参数 h0、h1
// 按 Σv^2 = min 组成法方程，解出改正数 Δ，更新参数后重复迭代。
//
// 每一轮迭代（和理论课的流程一一对应）：
//   ① 几何变形改正：用当前几何参数把窗口里的 (u,v) 映射到右影像 (x',y')
//   ② 重采样：按 (x',y') 从右影像取灰度 g（双三次卷积内插）
//   ③ 辐射畸变改正：解出辐射参数 h0、h1（它们有现成公式，不用迭代）
//   ④ 计算是否需要迭代：看本轮的改正数是不是已经足够小
//   ⑤ 求解变形参数改正值：组法方程，解 6 个几何参数的改正数
//   ⑥ 计算变形参数：把改正数加到参数上
//   ⑦ 循环结束后：u = v = 0 对应的 (x', y') = (a0, b0) 就是最佳匹配点位
//
// 求偏导（gx、gy 是右影像在 (x',y') 处的灰度梯度）：
//   dv/da0 = -gx    dv/da1 = -u·gx    dv/da2 = -v·gx
//   dv/db0 = -gy    dv/db1 = -u·gy    dv/db2 = -v·gy

// ---- 双三次卷积重采样 ----
// 双线性内插只用周围 4 个像素，插值面在像素边界上是折面，导数不连续，
// 最小二乘迭代到边界附近会来回摆动、收敛很慢（试过：一半的点 30 次都
// 收敛不了）。双三次卷积用周围 4x4 = 16 个像素按距离加权，插值面光滑、
// 导数连续，迭代 5~8 次就收敛。权函数（Keys 核，a = -0.5）：
//      |t| <= 1  :  W = 1.5|t|^3 - 2.5|t|^2 + 1
//      1 < |t| < 2:  W = -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2
// 下面对小数部分 t 求 4 个权 w[0..3]（对应 ix-1、ix、ix+1、ix+2 四个点），
// 同时给出权函数的导数 dw，用来算插值面在该点的梯度。
void cubicWeights(double t, double w[4], double dw[4])
{
    double t2 = t * t, t3 = t2 * t;
    double s1 = 1 - t, s1_2 = s1 * s1, s1_3 = s1_2 * s1;
    double s2 = 2 - t, s2_2 = s2 * s2, s2_3 = s2_2 * s2;

    w[0] = -0.5 * (1 + t) * (1 + t) * (1 + t) + 2.5 * (1 + t) * (1 + t) - 4 * (1 + t) + 2;
    w[1] = 1.5 * t3 - 2.5 * t2 + 1.0;
    w[2] = 1.5 * s1_3 - 2.5 * s1_2 + 1.0;
    w[3] = -0.5 * s2_3 + 2.5 * s2_2 - 4 * s2 + 2;

    dw[0] = -1.5 * (1 + t) * (1 + t) + 5 * (1 + t) - 4;
    dw[1] = 4.5 * t2 - 5 * t;
    dw[2] = -(4.5 * s1_2 - 5 * s1);
    dw[3] = 1.5 * s2_2 - 5 * s2 + 4;
}

// 在浮点坐标 (x,y) 处重采样，顺便把对 x、y 的偏导数一起算出来。
// 注意：误差方程里用的梯度必须来自同一种重采样方式，否则迭代会来回振荡。
double resample(const GrayImage& img, double x, double y,
                double& dgdx, double& dgdy, bool& ok)
{
    ok = false;
    dgdx = 0;
    dgdy = 0;

    int ix = (int)floor(x);                       // 左上角那个整数像素
    int iy = (int)floor(y);
    // 要取周围 4x4 个像素，靠边就取不全了
    if (ix - 1 < 0 || iy - 1 < 0 || ix + 2 >= img.w || iy + 2 >= img.h) return 0.0;

    double wx[4], dwx[4], wy[4], dwy[4];
    cubicWeights(x - ix, wx, dwx);                // x 方向的权和导数
    cubicWeights(y - iy, wy, dwy);                // y 方向的权和导数

    double g = 0, gx = 0, gy = 0;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double p = img.at(ix - 1 + i, iy - 1 + j);
            g  += wx[i] * wy[j] * p;
            gx += dwx[i] * wy[j] * p;
            gy += wx[i] * dwy[j] * p;
        }
    }

    ok = true;
    dgdx = gx;
    dgdy = gy;
    return g;
}

// 解 6 元线性方程组 A·x = b（高斯消元 + 选主元）。A、b 会被改掉。
bool solveLinear(double A[6][6], double b[6], double x[6])
{
    for (int k = 0; k < 6; ++k) {
        // 选绝对值最大的那一行做主元，避免除很小的数
        int p = k;
        for (int i = k + 1; i < 6; ++i)
            if (fabs(A[i][k]) > fabs(A[p][k])) p = i;
        if (fabs(A[p][k]) < 1e-12) return false;  // 法方程奇异
        if (p != k) {
            for (int j = 0; j < 6; ++j) swap(A[k][j], A[p][j]);
            swap(b[k], b[p]);
        }
        // 用第 k 行把下面各行的第 k 列消成 0
        for (int i = k + 1; i < 6; ++i) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j < 6; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    // 回代
    for (int i = 5; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < 6; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return true;
}

// 最小二乘精化：以 (x0,y0) 为初值，在左点 (lx,ly) 的 r 邻域窗口上迭代。
// 辐射参数 h0、h1 用闭式解（③），先解它们再解几何参数（⑤）：8 个未知数
// 一起解的话 h0、h1 会和几何参数强相关，法方程接近奇异，迭代会来回振荡。
LsmResult lsmRefine(const GrayImage& F, const GrayImage& G,
                    int lx, int ly, int r, double x0, double y0, int maxIter)
{
    LsmResult res;
    res.a[0] = x0; res.a[1] = 1; res.a[2] = 0;    // 初值：只平移，不旋转不缩放
    res.b[0] = y0; res.b[1] = 0; res.b[2] = 1;

    int total = (2 * r + 1) * (2 * r + 1);        // 窗口内像素个数

    // 模板窗口的平均灰度。辐射改正写成 g = h0 + h1*(f - f均值)，
    // 常数项和 f - f均值 差不多正交，法方程的性质会好很多。
    double fmean = 0;
    for (int v = -r; v <= r; ++v)
        for (int u = -r; u <= r; ++u) fmean += F.at(lx + u, ly + v);
    fmean /= total;
    res.h0 = fmean;
    res.h1 = 1;

    double d[6] = {0, 0, 0, 0, 0, 0};             // 本轮解出的改正数

    for (int iter = 0; iter < maxIter; ++iter) {

        // ================= ①② 几何变形改正 + 重采样，③ 辐射畸变改正 =========
        // 窗口里每个 (u,v)：先用当前几何参数算出右影像上的位置（①），
        // 双线性内插取灰度 g（②），并累加解 h0、h1 要用的几个和。
        double sg = 0, sfc = 0, sfg = 0, sff = 0;
        int used = 0;
        for (int v = -r; v <= r; ++v) {
            for (int u = -r; u <= r; ++u) {
                double fc = F.at(lx + u, ly + v) - fmean;
                double xr = res.a[0] + res.a[1] * u + res.a[2] * v;
                double yr = res.b[0] + res.b[1] * u + res.b[2] * v;
                bool ok = false;
                double gx, gy;
                double g = resample(G, xr, yr, gx, gy, ok);
                if (!ok) continue;
                sg += g;  sfc += fc;  sfg += fc * g;  sff += fc * fc;
                ++used;
            }
        }
        if (used < total * 3 / 4) return res;     // 有效像素太少，匹配失败
        if (sff < 1e-6) return res;               // 模板没有纹理

        // ③ 辐射参数的最小二乘公式（对 h0、h1 分别求偏导等于零解出来的）
        double gmean = sg / used;
        double h1 = (sfg - gmean * sfc) / sff;    // Σfc(g-g均值) / Σfc^2
        double h0 = gmean - h1 * sfc / used;
        res.h0 = h0;
        res.h1 = h1;

        // ================= ⑤ 求解变形参数改正值：组法方程 =================
        // 误差方程 v = h0 + h1·fc - g(x',y')，对 6 个几何参数求偏导
        // （偏导就是 -gx、-gy 的各种组合），法方程 (JTJ)Δ = -JTv。
        double A[6][6], L[6];
        for (int i = 0; i < 6; ++i) {
            L[i] = 0;
            for (int j = 0; j < 6; ++j) A[i][j] = 0;
        }

        for (int v = -r; v <= r; ++v) {
            for (int u = -r; u <= r; ++u) {
                double fc = F.at(lx + u, ly + v) - fmean;
                double xr = res.a[0] + res.a[1] * u + res.a[2] * v;
                double yr = res.b[0] + res.b[1] * u + res.b[2] * v;
                bool ok = false;
                double gx, gy;
                double g = resample(G, xr, yr, gx, gy, ok);
                if (!ok) continue;

                double vres = h0 + h1 * fc - g;   // 误差方程的常数项
                double J[6];                      // 偏导（雅可比矩阵的一行）
                J[0] = -gx;  J[1] = -gx * u;  J[2] = -gx * v;
                J[3] = -gy;  J[4] = -gy * u;  J[5] = -gy * v;

                for (int i = 0; i < 6; ++i) {
                    L[i] += J[i] * vres;
                    for (int j = i; j < 6; ++j) A[i][j] += J[i] * J[j];
                }
            }
        }
        for (int i = 0; i < 6; ++i)               // 上三角补齐成对称矩阵
            for (int j = 0; j < i; ++j) A[i][j] = A[j][i];
        for (int i = 0; i < 6; ++i) L[i] = -L[i]; // 法方程右端 = -JTv

        if (!solveLinear(A, L, d)) return res;    // 解 6 元法方程

        // ================= ⑥ 计算变形参数：改正数加到参数上 ===============
        res.a[0] += d[0]; res.a[1] += d[1]; res.a[2] += d[2];
        res.b[0] += d[3]; res.b[1] += d[4]; res.b[2] += d[5];
        res.iter = iter + 1;

        // ================ ④ 计算是否需要迭代 ================================
        // 位置改正数小于 0.005 像素、变形参数改正数也很小，就认为收敛
        bool conv = fabs(d[0]) < 0.005 && fabs(d[3]) < 0.005
                 && fabs(d[1]) < 1e-4  && fabs(d[2]) < 1e-4
                 && fabs(d[4]) < 1e-4  && fabs(d[5]) < 1e-4;
        if (conv) break;
    }

    // 跑飞检查：精化结果不应该离初值太远
    if (fabs(res.a[0] - x0) > 6.0 || fabs(res.b[0] - y0) > 6.0) return res;

    // ================= ⑦ 最佳匹配点位：u=v=0 时的 (x',y') =================
    res.xr = res.a[0];
    res.yr = res.b[0];

    // 用最终的变形参数把右影像窗口重采样一遍，和模板算相关系数，作为精度指标
    double sf = 0, sg = 0, sff = 0, sgg = 0, sfg = 0;
    int n = 0;
    for (int v = -r; v <= r; ++v) {
        for (int u = -r; u <= r; ++u) {
            double f = F.at(lx + u, ly + v);
            double xr = res.a[0] + res.a[1] * u + res.a[2] * v;
            double yr = res.b[0] + res.b[1] * u + res.b[2] * v;
            bool ok = false;
            double gx, gy;
            double g = resample(G, xr, yr, gx, gy, ok);
            if (!ok) continue;
            sf += f;   sg += g;
            sff += f * f;
            sgg += g * g;
            sfg += f * g;
            ++n;
        }
    }
    if (n < total * 3 / 4) return res;

    double num = sfg - sf * sg / n;
    double d1 = sff - sf * sf / n;
    double d2 = sgg - sg * sg / n;
    if (d1 < 1e-6 || d2 < 1e-6) return res;

    res.ncc = num / sqrt(d1 * d2);
    res.ok = true;
    return res;
}

// ============================================================================
//  六、结果图
// ============================================================================

// 画线（简单的 DDA，步长按 x、y 里变化大的那个算）
void drawLine(RgbImage& im, int x1, int y1, int x2, int y2,
              unsigned char r, unsigned char g, unsigned char b)
{
    int dx = x2 - x1;
    int dy = y2 - y1;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    int n = dx > dy ? dx : dy;

    if (n == 0) { im.set(x1, y1, r, g, b); return; }

    for (int i = 0; i <= n; ++i) {
        double t = (double)i / n;
        int x = (int)(x1 + (x2 - x1) * t + 0.5);
        int y = (int)(y1 + (y2 - y1) * t + 0.5);
        im.set(x, y, r, g, b);
    }
}

// 画空心圆
void drawCircle(RgbImage& im, int cx, int cy, int radius,
                unsigned char r, unsigned char g, unsigned char b)
{
    for (int y = cy - radius; y <= cy + radius; ++y) {
        for (int x = cx - radius; x <= cx + radius; ++x) {
            double d = sqrt((double)(x - cx) * (x - cx) + (double)(y - cy) * (y - cy));
            if (d <= radius && d >= radius - 1.2) im.set(x, y, r, g, b);
        }
    }
}

// 画十字
void drawCross(RgbImage& im, int cx, int cy, int len,
               unsigned char r, unsigned char g, unsigned char b)
{
    for (int i = -len; i <= len; ++i) {
        im.set(cx + i, cy, r, g, b);
        im.set(cx, cy + i, r, g, b);
    }
}

// 把左右影像左右拼在一起，画上同名点（红圈 + 黄色十字）和连线（绿线）
void drawResult(const GrayImage& L, const GrayImage& R,
                const vector<TiePoint>& pts, const char* filename)
{
    int gap = 12;                                 // 两幅图之间的间隔
    int H = L.h > R.h ? L.h : R.h;

    RgbImage im;
    im.w = L.w + gap + R.w;
    im.h = H;
    im.pix.assign(im.w * im.h * 3, 30);           // 先涂成深灰底色

    // 把灰度影像贴上去
    for (int y = 0; y < L.h; ++y)
        for (int x = 0; x < L.w; ++x) {
            unsigned char v = L.at(x, y);
            im.set(x, y, v, v, v);
        }
    for (int y = 0; y < R.h; ++y)
        for (int x = 0; x < R.w; ++x) {
            unsigned char v = R.at(x, y);
            im.set(L.w + gap + x, y, v, v, v);
        }

    // 画同名点和连线
    for (size_t i = 0; i < pts.size(); ++i) {
        const TiePoint& p = pts[i];
        if (!p.keep) continue;

        int xl = (int)(p.xl + 0.5);
        int yl = (int)(p.yl + 0.5);
        double rx = p.lsmOk ? p.xrL : p.xr;       // 有精化结果就用精化的
        double ry = p.lsmOk ? p.yrL : p.yr;
        int xr = (int)(rx + 0.5);
        int yr = (int)(ry + 0.5);

        drawLine(im, xl, yl, L.w + gap + xr, yr, 0, 220, 0);       // 绿线
        drawCircle(im, xl, yl, 7, 255, 60, 60);                    // 左图红圈
        drawCircle(im, L.w + gap + xr, yr, 7, 255, 60, 60);        // 右图红圈
        drawCross(im, xl, yl, 3, 255, 255, 0);                     // 黄色十字
        drawCross(im, L.w + gap + xr, yr, 3, 255, 255, 0);
    }

    saveBmp(filename, im);
}

// ============================================================================
//  七、主程序
// ============================================================================

// 提示并读入一个数，直接回车就用默认值
double askNumber(const char* prompt, double def, double lo, double hi)
{
    cout << prompt << " [默认 " << def << "]: ";
    string s;
    getline(cin, s);
    if (s.empty()) return def;

    double v = atof(s.c_str());
    if (v >= lo && v <= hi) return v;

    cout << "  输入不合法，已使用默认值 " << def << endl;
    return def;
}

// 提示并读入一个路径，去掉首尾空格和引号（从资源管理器"复制路径"会带引号）
string askPath(const char* prompt)
{
    cout << prompt;
    string s;
    getline(cin, s);

    while (!s.empty() && (s[0] == ' ' || s[0] == '\t')) s.erase(0, 1);
    while (!s.empty() && (s[s.size() - 1] == ' ' || s[s.size() - 1] == '\t')) s.erase(s.size() - 1);
    if (s.size() >= 2 && s[0] == '"' && s[s.size() - 1] == '"') s = s.substr(1, s.size() - 2);

    return s;
}

int main(int argc, char* argv[])
{
    cout << "   相关系数匹配 + 最小二乘匹配（摄影测量学 编程题 1）" << endl;

    // argc 是命令行参数的个数，argv 是这些参数。
    // 把两张影像拖到 exe 图标上时，系统会自动把两个路径变成 argv[1]、argv[2]。
    string leftPath, rightPath;
    if (argc >= 3) {
        leftPath = argv[1];
        rightPath = argv[2];
        cout << "左影像：" << leftPath << endl;
        cout << "右影像：" << rightPath << endl;
    } else {
        leftPath = askPath("请输入【左】影像的路径（可以直接把文件拖进窗口）：");
        rightPath = askPath("请输入【右】影像的路径：");
    }
    if (leftPath.empty() || rightPath.empty()) {
        cout << "没有输入影像路径，程序结束。" << endl;
        return 1;
    }

    // ---------- 匹配参数 ----------
    cout << endl << "-------- 匹配参数（直接回车用默认值）--------" << endl;
    int r         = (int)askNumber("模板窗口半径 r（窗口大小 = 2r+1）", 11, 3, 40);
    int searchR   = (int)askNumber("搜索半径（在预测位置附近 ±多少个像素）", 25, 2, 100);
    int gridStep  = (int)askNumber("格网点间距（像素）", 60, 10, 400);
    double thresh = askNumber("相关系数阈值（低于该值视为匹配失败）", 0.80, 0.0, 1.0);
    int doLsm     = (int)askNumber("是否做最小二乘精化（1 = 做，0 = 不做）", 1, 0, 1);
    int maxIter   = (int)askNumber("最小二乘最大迭代次数", 20, 1, 30);

    cout << fixed << setprecision(3);             // 后面输出小数统一保留 3 位

    clock_t t0 = clock();

    // ---------- 1. 读影像 ----------
    GrayImage L, R;
    if (!readImage(leftPath.c_str(), L)) {
        cout << "左影像读取失败，请检查路径是否正确。" << endl;
        return 1;
    }
    if (!readImage(rightPath.c_str(), R)) {
        cout << "右影像读取失败，请检查路径是否正确。" << endl;
        return 1;
    }
    cout << endl << "[1/5] 读入影像：左 " << L.w << " x " << L.h
         << "，右 " << R.w << " x " << R.h << endl;

    // ---------- 2. 粗定位：找整体位移 ----------
    // 两张图缩小 4 倍以后全范围滑动搜索，找到整体位移（粗定位）。
    // 整体位移 + 局部视差 = 真正的同名点位置，所以逐点搜索只需要
    // 在"左点 + 整体位移"附近小范围（±searchR）进行。
    GrayImage Ls = shrink(L, 4);
    GrayImage Rs = shrink(R, 4);
    ShiftResult s = searchShift(Ls, Rs, Ls.w / 2, Ls.h / 4, 3);

    double gdx = s.dx * 4.0;                      // 换算回原图的像素数
    double gdy = s.dy * 4.0;
    cout << "[2/5] 粗定位：整体位移 dx = " << gdx << "，dy = " << gdy
         << "（相关系数 " << s.ncc << "）" << endl;

    // ---------- 3. 逐点相关系数匹配 ----------
    vector<TiePoint> pts;
    int candidates = 0, nccOk = 0;

    // 影像四周留出一圈不取点：航片边缘有黑框和框标，往往匹配错
    int maxSize = L.w > L.h ? L.w : L.h;
    int margin = r + 2 + maxSize / 40;

    for (int y = margin; y < L.h - margin; y += gridStep) {
        for (int x = margin; x < L.w - margin; x += gridStep) {
            // 预测的同名点位置 = 左点 + 整体位移
            int px = (int)(x + gdx + 0.5);
            int py = (int)(y + gdy + 0.5);
            // 预测位置太靠边，右影像上取不到完整窗口
            if (px - searchR - r < 0 || py - searchR - r < 0
             || px + searchR + r >= R.w || py + searchR + r >= R.h) continue;

            // 模板是纯色（灰度方差很小）就没有匹配的价值
            double sf = 0, sff = 0;
            int n = 0;
            for (int v = -r; v <= r; ++v)
                for (int u = -r; u <= r; ++u) {
                    double f = L.at(x + u, y + v);
                    sf += f;
                    sff += f * f;
                    ++n;
                }
            double var = sff - sf * sf / n;
            ++candidates;
            if (var / n < 9.0) continue;          // 灰度标准差小于 3，纹理太少

            TiePoint p;
            p.xl = x;
            p.yl = y;
            if (matchByNcc(L, R, x, y, r, searchR, px, py, thresh,
                           p.xr, p.yr, p.ncc0)) {
                pts.push_back(p);
                ++nccOk;
            }
        }
    }
    cout << "[3/5] 相关系数匹配：候选 " << candidates << " 点，成功 " << nccOk
         << " 点（阈值 " << thresh << "）" << endl;

    if (pts.empty()) {
        cout << "没有匹配到任何点，请检查影像或放宽参数。" << endl;
        return 1;
    }

    // ---------- 4. 最小二乘精化 ----------
    int lsmOk = 0;
    double sumIter = 0, sumNcc0 = 0, sumNccL = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        TiePoint& p = pts[i];
        sumNcc0 += p.ncc0;

        if (!doLsm) {                             // 不做精化，就直接用相关系数的结果
            p.xrL = p.xr;
            p.yrL = p.yr;
            p.nccL = p.ncc0;
            p.lsmOk = false;
            continue;
        }

        LsmResult res = lsmRefine(L, R, (int)p.xl, (int)p.yl, r, p.xr, p.yr, maxIter);
        if (res.ok) {
            p.xrL = res.xr;
            p.yrL = res.yr;
            p.nccL = res.ncc;
            p.iter = res.iter;
            p.lsmOk = true;
            ++lsmOk;
            sumIter += res.iter;
            sumNccL += res.ncc;
        }
    }
    if (doLsm) {
        double avgIter = lsmOk ? sumIter / lsmOk : 0;
        double avgNccL = lsmOk ? sumNccL / lsmOk : 0;
        cout << "[4/5] 最小二乘精化：成功 " << lsmOk << " / " << (int)pts.size()
             << " 点，平均迭代 " << avgIter << " 次，平均相关系数 "
             << sumNcc0 / pts.size() << " -> " << avgNccL << endl;
    }

    // ---------- 5. 粗差剔除 ----------
    // 同一立体像对的视差是缓慢变化的，偏离中位视差太多的点判为粗差。
    // 尺度用中位数绝对偏差（MAD）估计，限差取 5 倍 MAD，最小 3 像素。
    vector<double> dxs, dys;
    for (size_t i = 0; i < pts.size(); ++i) {
        const TiePoint& p = pts[i];
        if (doLsm && !p.lsmOk) continue;
        dxs.push_back((p.lsmOk ? p.xrL : p.xr) - p.xl);
        dys.push_back((p.lsmOk ? p.yrL : p.yr) - p.yl);
    }

    if (!dxs.empty()) {
        vector<double> sx = dxs, sy = dys;
        sort(sx.begin(), sx.end());
        sort(sy.begin(), sy.end());
        double mx = sx[sx.size() / 2];            // 视差中位数
        double my = sy[sy.size() / 2];

        vector<double> ax(dxs.size()), ay(dys.size());
        for (size_t i = 0; i < dxs.size(); ++i) {
            ax[i] = fabs(dxs[i] - mx);
            ay[i] = fabs(dys[i] - my);
        }
        sort(ax.begin(), ax.end());
        sort(ay.begin(), ay.end());
        double madx = ax[ax.size() / 2] + 0.5;    // +0.5 防止 MAD 为 0
        double mady = ay[ay.size() / 2] + 0.5;

        double limx = 5.0 * madx;                 // 限差
        double limy = 5.0 * mady;
        if (limx < 3.0) limx = 3.0;
        if (limy < 3.0) limy = 3.0;

        int kept = 0, removed = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            TiePoint& p = pts[i];
            if (doLsm && !p.lsmOk) { p.keep = false; ++removed; continue; }

            double ddx = (p.lsmOk ? p.xrL : p.xr) - p.xl;
            double ddy = (p.lsmOk ? p.yrL : p.yr) - p.yl;
            if (fabs(ddx - mx) > limx || fabs(ddy - my) > limy) {
                p.keep = false;
                ++removed;
                continue;
            }
            ++kept;
        }
        cout << "[5/5] 粗差检查：保留 " << kept << " 点，剔除 " << removed
             << " 点（视差中位数 " << mx << " , " << my
             << "，限差 " << limx << " , " << limy << " 像素）" << endl;
    }

    // ---------- 输出同名点文件 ----------
    int kept = 0;
    double meanNcc = 0, rms = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!pts[i].keep) continue;
        ++kept;
        meanNcc += pts[i].lsmOk ? pts[i].nccL : pts[i].ncc0;

        // 与整体位移的较差，用来看视差在图内的变化
        double ex = (pts[i].lsmOk ? pts[i].xrL : pts[i].xr) - pts[i].xl - gdx;
        double ey = (pts[i].lsmOk ? pts[i].yrL : pts[i].yr) - pts[i].yl - gdy;
        rms += ex * ex + ey * ey;
    }

    ofstream csv("tiepoints.csv");
    if (csv) {
        csv << "id,x_left,y_left,x_right,y_right,ncc_coarse,"
               "x_right_lsm,y_right_lsm,ncc_lsm,iterations\n";
        csv << fixed << setprecision(3);

        int id = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            const TiePoint& p = pts[i];
            if (!p.keep) continue;
            ++id;
            csv << id << ","
                << (int)p.xl << "," << (int)p.yl << ","          // 左点是格网点，本来就是整数
                << setprecision(3) << p.xr << "," << p.yr << ","
                << setprecision(5) << p.ncc0 << ","
                << setprecision(3) << (p.lsmOk ? p.xrL : p.xr) << ","
                << (p.lsmOk ? p.yrL : p.yr) << ","
                << setprecision(5) << (p.lsmOk ? p.nccL : p.ncc0) << ","
                << p.iter << "\n";
        }
    }

    // ---------- 输出结果图 ----------
    drawResult(L, R, pts, "result.bmp");

    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    double rmse = kept ? sqrt(rms / (2.0 * kept)) : 0;

    cout << endl << "================= 结果汇总 =================" << endl;
    cout << "  同名点个数    : " << kept << endl;
    cout << "  平均相关系数  : " << (kept ? meanNcc / kept : 0.0) << endl;
    cout << "  视差较差中误差: " << rmse << " 像素（相对整体位移）" << endl;
    cout << "  输出文件      : tiepoints.csv（同名点坐标）" << endl;
    cout << "                  result.bmp（匹配结果图，左图右图并排）" << endl;
    cout << "  总耗时        : " << sec << " 秒" << endl;
    cout << "============================================" << endl << endl;

    cout << "按回车键退出..." << endl;
    cin.get();
    return 0;
}

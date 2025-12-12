// -----------------------------------------------------------------------------
// 3x3 Singular Value Decomposition (SVD) for HLSL
//
// Written by abecombe on 2025/12/15
//
// This implementation is based on the following paper:
//
//   "Computing the Singular Value Decomposition of 3×3 matrices
//    with minimal branching and elementary floating point operations"
//   Aleka McAdams, Andrew Selle, Rasmus Tamstorf,
//   Joseph Teran, Eftychios Sifakis
//   Technical Report #1690, University of Wisconsin–Madison, May 2011
//
// The algorithm follows the quaternion-based approximate Jacobi
// eigenanalysis for AᵀA, sorting of singular values, and QR-based
// extraction of U and Σ, as described in the paper.
//
// Reference:
// https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf
// -----------------------------------------------------------------------------

#ifndef SVD_HLSL
#define SVD_HLSL

#define SVD_EPSILON 1e-6
#define SVD_GAMMA   5.8284271247 // 3 + 2 * sqrt(2)
#define SVD_C_STAR  0.9238795325 // cos(pi/8)
#define SVD_S_STAR  0.3826834323 // sin(pi/8)
// q = (w, x, y, z)

inline float4 QuaternionMultiply(in float4 q1, in float4 q2)
{
    return float4(
        q1.x * q2.x - dot(q1.yzw, q2.yzw),
        q1.x * q2.yzw + q2.x * q1.yzw + cross(q1.yzw, q2.yzw)
    );
}

inline float3x3 QuaternionToMatrix(in float4 q)
{
    float x2 = q.y + q.y; float y2 = q.z + q.z; float z2 = q.w + q.w;
    float xx = q.y * x2;  float xy = q.y * y2;  float xz = q.y * z2;
    float yy = q.z * y2;  float yz = q.z * z2;  float zz = q.w * z2;
    float wx = q.x * x2;  float wy = q.x * y2;  float wz = q.x * z2;

    return float3x3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

inline float AccurateRSqrt(in float x)
{
    float y = rsqrt(x);
    return y * (1.5 - 0.5 * x * y * y);
}

inline void CondSwap(in bool condition, inout float x, inout float y)
{
    float temp = x;
    x = condition ? y : x;
    y = condition ? temp : y;
}

inline void CondSwap(in bool condition, inout float3 x, inout float3 y)
{
    float3 temp = x;
    x = condition ? y : x;
    y = condition ? temp : y;
}

inline void CondNegSwap(in bool condition, inout float3 x, inout float3 y)
{
    float3 temp = -x;
    x = condition ? y : x;
    y = condition ? temp : y;
}

inline void SortSingularValues(inout float3x3 B, inout float3x3 V)
{
    float3 v0 = float3(V[0][0], V[1][0], V[2][0]);
    float3 v1 = float3(V[0][1], V[1][1], V[2][1]);
    float3 v2 = float3(V[0][2], V[1][2], V[2][2]);
    float3 b0 = float3(B[0][0], B[1][0], B[2][0]);
    float3 b1 = float3(B[0][1], B[1][1], B[2][1]);
    float3 b2 = float3(B[0][2], B[1][2], B[2][2]);
    float pho0 = length(b0), pho1 = length(b1), pho2 = length(b2);

    bool c = pho0 < pho1;
    CondNegSwap(c, b0, b1); CondNegSwap(c, v0, v1);
    CondSwap(c, pho0, pho1);
    c = pho0 < pho2;
    CondNegSwap(c, b0, b2); CondNegSwap(c, v0, v2);
    CondSwap(c, pho0, pho2);
    c = pho1 < pho2;
    CondNegSwap(c, b1, b2); CondNegSwap(c, v1, v2);

    B[0] = float3(b0.x, b1.x, b2.x);
    B[1] = float3(b0.y, b1.y, b2.y);
    B[2] = float3(b0.z, b1.z, b2.z);

    V[0] = float3(v0.x, v1.x, v2.x);
    V[1] = float3(v0.y, v1.y, v2.y);
    V[2] = float3(v0.z, v1.z, v2.z);
}

inline void ApproxGivensQuaternion(in float a11, in float a12, in float a22, out float ch, out float sh)
{
    ch = 2.0 * (a11 - a22);
    sh = a12;
    bool b = SVD_GAMMA * sh * sh < ch * ch;
    float w = AccurateRSqrt(ch * ch + sh * sh);
    ch = b ? w * ch : SVD_C_STAR;
    sh = b ? w * sh : SVD_S_STAR;
}

inline void QrGivensQuaternion(in float a1, in float a2, out float ch, out float sh)
{
    float rho = sqrt(a1 * a1 + a2 * a2);
    ch = abs(a1) + max(rho, SVD_EPSILON);
    sh = rho > SVD_EPSILON ? a2 : 0.0;
    CondSwap(a1 < 0.0, sh, ch);
    float w = AccurateRSqrt(ch * ch + sh * sh);
    ch = w * ch;
    sh = w * sh;
}

inline void RotateSymmetricMatrix(inout float3x3 mat, in float ch, in float sh, in int p, in int q)
{
    float a = 1.0 - 2.0 * sh * sh;
    float b = 2.0 * ch * sh;

    float aa = a * a; float bb = b * b; float ab = a * b;

    int k = 3 - (p + q);

    float s_pp = mat[p][p];
    float s_qq = mat[q][q];
    float s_pq = mat[p][q];
    float s_kp = mat[k][p];
    float s_kq = mat[k][q];

    float n_pp = aa * s_pp + 2.0 * ab * s_pq + bb * s_qq;
    float n_qq = bb * s_pp - 2.0 * ab * s_pq + aa * s_qq;
    float n_pq = (aa - bb) * s_pq + ab * (s_qq - s_pp);
    float n_kp = a * s_kp + b * s_kq;
    float n_kq = -b * s_kp + a * s_kq;

    mat[p][p] = n_pp;
    mat[q][q] = n_qq;
    mat[p][q] = mat[q][p] = n_pq;
    mat[k][p] = mat[p][k] = n_kp;
    mat[k][q] = mat[q][k] = n_kq;
}

inline void PremultiplyTransposeR(inout float3x3 mat, in float ch, in float sh, in int p, in int q)
{
    float a = 1.0 - 2.0 * sh * sh;
    float b = 2.0 * ch * sh;

    float3 row_p = mat[p];
    float3 row_q = mat[q];

    mat[p] = a * row_p + b * row_q;
    mat[q] = -b * row_p + a * row_q;
}

// -----------------------------------------------------------------------------
// Main SVD function
// input      : A
// output     : U, S(diagonal), V
// definition : A = U * diag(S) * V^T
// -----------------------------------------------------------------------------
void SVD(in float3x3 A, out float3x3 U, out float3 S, out float3x3 V)
{
    float3x3 S_mat = mul(transpose(A), A);
    float4 q = float4(1, 0, 0, 0);

    float ch, sh;
    float4 rot;
    // float3x3 R;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        ApproxGivensQuaternion(S_mat[0][0], S_mat[0][1], S_mat[1][1], ch, sh);
        rot = float4(ch, 0, 0, sh);
        q = QuaternionMultiply(q, rot);
        // R = QuaternionToMatrix(rot);
        // S_mat = mul(transpose(R), mul(S_mat, R));
        RotateSymmetricMatrix(S_mat, ch, sh, 0, 1);

        ApproxGivensQuaternion(S_mat[0][0], S_mat[0][2], S_mat[2][2], ch, sh);
        rot = float4(ch, 0, -sh, 0);
        q = QuaternionMultiply(q, rot);
        // R = QuaternionToMatrix(rot);
        // S_mat = mul(transpose(R), mul(S_mat, R));
        RotateSymmetricMatrix(S_mat, ch, -sh, 2, 0);

        ApproxGivensQuaternion(S_mat[1][1], S_mat[1][2], S_mat[2][2], ch, sh);
        rot = float4(ch, sh, 0, 0);
        q = QuaternionMultiply(q, rot);
        // R = QuaternionToMatrix(rot);
        // S_mat = mul(transpose(R), mul(S_mat, R));
        RotateSymmetricMatrix(S_mat, ch, sh, 1, 2);
    }

    float q_norm = AccurateRSqrt(dot(q, q));
    q *= q_norm;
    V = QuaternionToMatrix(q);

    float3x3 B = mul(A, V);

    SortSingularValues(B, V);

    float4 u_q = float4(1, 0, 0, 0);

    QrGivensQuaternion(B[0][0], B[1][0], ch, sh);
    rot = float4(ch, 0, 0, sh);
    u_q = QuaternionMultiply(u_q, rot);
    // R = QuaternionToMatrix(rot);
    // B = mul(transpose(R), B);
    PremultiplyTransposeR(B, ch, sh, 0, 1);

    QrGivensQuaternion(B[0][0], B[2][0], ch, sh);
    rot = float4(ch, 0, -sh, 0);
    u_q = QuaternionMultiply(u_q, rot);
    // R = QuaternionToMatrix(rot);
    // B = mul(transpose(R), B);
    PremultiplyTransposeR(B, ch, -sh, 2, 0);

    QrGivensQuaternion(B[1][1], B[2][1], ch, sh);
    rot = float4(ch, sh, 0, 0);
    u_q = QuaternionMultiply(u_q, rot);
    // R = QuaternionToMatrix(rot);
    // B = mul(transpose(R), B);
    PremultiplyTransposeR(B, ch, sh, 1, 2);

    float u_q_norm = AccurateRSqrt(dot(u_q, u_q));
    u_q *= u_q_norm;
    U = QuaternionToMatrix(u_q);

    S = float3(B[0][0], B[1][1], B[2][2]);

    float3 sign_s = sign(S);
    S = abs(S);
    U[0] *= sign_s;
    U[1] *= sign_s;
    U[2] *= sign_s;
}

#endif /* SVD_HLSL */

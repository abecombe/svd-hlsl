# 3x3 SVD (HLSL)

## Overview

This repository provides a **compact and GPU-friendly Singular Value Decomposition (SVD) implementation for 3×3 matrices written in HLSL**.

The algorithm avoids trigonometric functions and heavy branching, making it suitable for **ComputeShaders**, **real-time graphics**, and **SIMD-style execution**.

The decomposition follows:

```
A = U · diag(S) · Vᵀ
```

Where `U` and `V` are orthonormal rotation matrices and `S` contains the singular values.

In addition, this repository provides a **Polar Decomposition** built on top of the SVD, allowing the matrix to be decomposed into **rotation** and **stretch** components.

---

## Usage

### Singular Value Decomposition

```hlsl
float3x3 U, V;
float3   S;

SVD(A, U, S, V);
```

* `A` : input 3×3 matrix
* `U` : left singular vectors (rotation matrix)
* `S` : singular values (non-negative, sorted descending)
* `V` : right singular vectors (rotation matrix)

---

### Polar Decomposition (SVD-based)

```hlsl
float3x3 U, V, R;
float3   S;

SVD_PolarDecomposition(A, U, S, V, R);
```

The decomposition follows:

```
A = U · diag(S) · Vᵀ
R = U · Vᵀ
```

* `A` : input 3×3 matrix
* `U` : left singular vectors
* `S` : singular values (stretch component)
* `V` : right singular vectors
* `R` : **pure rotation matrix** (`det(R) = +1`)

This form is commonly used in **deformation analysis**, **physics simulation (FEM)**, and **corotational models**, where separating rotation from stretch is required.

---

## Reference

This implementation is based on:

> **Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations**
> Aleka McAdams, Andrew Selle, Rasmus Tamstorf, Joseph Teran, Eftychios Sifakis
> Technical Report #1690, University of Wisconsin–Madison, May 2011

PDF:
[https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf](https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf)

---

**Author**: abecombe
**Date**: 2025-12-15
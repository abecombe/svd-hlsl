# 3x3 SVD (HLSL)

## Overview

This repository provides a **compact and GPU-friendly Singular Value Decomposition (SVD) implementation for 3×3 matrices written in HLSL**.

The algorithm avoids trigonometric functions and heavy branching, making it suitable for **ComputeShaders**, **real-time graphics**, and **SIMD-style execution**.

The decomposition follows:

```
A = U · diag(S) · Vᵀ
```

Where `U` and `V` are orthonormal rotation matrices and `S` contains the singular values.

---

## Usage

```hlsl
float3x3 U, V;
float3   S;

SVD(A, U, S, V);
```

* `A` : input 3×3 matrix
* `U` : left singular vectors (rotation matrix)
* `S` : singular values
* `V` : right singular vectors (rotation matrix)

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

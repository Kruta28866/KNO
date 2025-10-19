"""
KNO — Tensor Operations (TensorFlow + NumPy)

This single-file CLI implements Tasks 1–5:
• Rotate 2D points around the origin (0, 0).
• Return a TensorFlow 2×2 rotation matrix R(a).
• Solve linear systems Ax = b using matrix methods with basic existence checks.
• Clean CLI interface (argparse) with a proper __main__ entry point.
• Optional quick self-test to verify core functionality.

Requirements: TensorFlow 2.x and NumPy.
"""
from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf

DTYPE = tf.float64  # prefer higher precision for linear algebra



def rotation_matrix_tf(
    angle: float, *, degrees: bool = True, dtype: tf.dtypes.DType = DTYPE
) -> tf.Tensor:
    """Return a 2×2 TensorFlow rotation matrix R(a).

    R(a) = [[cos a, -sin a],
            [sin a,  cos a]]

    Args:
        angle: rotation angle.
        degrees: if True, `angle` is given in degrees; otherwise radians.
        dtype: TensorFlow floating dtype (defaults to float64).

    Returns:
        tf.Tensor of shape (2, 2).
    """
    theta = tf.convert_to_tensor(angle, dtype=dtype)
    if degrees:
        theta = theta * (math.pi / 180.0)
    c = tf.cos(theta)
    s = tf.sin(theta)
    return tf.stack([tf.stack([c, -s]), tf.stack([s, c])])



def rotate_points_tf(
    points: Iterable[Iterable[float]] | np.ndarray | tf.Tensor,
    angle: float,
    *,
    degrees: bool = True,
    dtype: tf.dtypes.DType = DTYPE,
) -> tf.Tensor:
    """Rotate N 2D points (x, y) around (0, 0) by a given angle.

    We represent points as rows: P' = P @ R^T
    where P has shape (N, 2) and R is 2×2 from `rotation_matrix_tf`.
    """
    P = tf.convert_to_tensor(points, dtype=dtype)
    P = tf.reshape(P, (-1, 2))  # (N, 2)
    R = rotation_matrix_tf(angle, degrees=degrees, dtype=dtype)  # (2, 2)
    return tf.linalg.matmul(P, tf.transpose(R))  # (N, 2)



def _infer_n_from_flat_matrix_len(flat_len: int) -> int:
    root = int(round(math.sqrt(flat_len)))
    if root * root != flat_len:
        raise ValueError(
            f"Length of A (= {flat_len}) is not a perfect square. Provide --n."
        )
    return root


def solve_linear_tf(
    A_flat: Sequence[float],
    b_vec: Sequence[float],
    *,
    n: int | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    dtype: tf.dtypes.DType = DTYPE,
) -> tuple[np.ndarray, str]:
    """Solve a linear system Ax = b using TensorFlow.

    We check existence via ranks: rank(A) ?= rank([A|b]).
    • If rank(A) != rank([A|b]) → no solutions.
    • If equal ranks and A is square full-rank → unique solution via `tf.linalg.solve`.
    • Otherwise return a minimum-norm solution via `tf.linalg.lstsq`.

    Returns: (x, status) — x as NumPy 1D array, status ∈ {"unique", "infinite_min_norm", "none"}.
    """
    if n is None:
        n = _infer_n_from_flat_matrix_len(len(A_flat))

    A = tf.convert_to_tensor(A_flat, dtype=dtype)
    A = tf.reshape(A, (n, -1))  # n×m (usually n×n)

    b = tf.convert_to_tensor(b_vec, dtype=dtype)
    if b.shape.rank == 0:
        b = tf.reshape(b, (1,))
    if b.shape[-1] != n:
        b = tf.reshape(b, (n,))
    b = tf.reshape(b, (n, 1))  # n×1

    # Ranks and existence check
    rank_A = tf.linalg.matrix_rank(A, tol=atol)
    Ab = tf.concat([A, b], axis=1)
    rank_Ab = tf.linalg.matrix_rank(Ab, tol=atol)

    if int(rank_A.numpy()) != int(rank_Ab.numpy()):
        return np.array([], dtype=np.float64), "none"

    n_rows = int(A.shape[0])
    n_cols = int(A.shape[1])

    if n_rows == n_cols and int(rank_A.numpy()) == n_cols:
        # Unique solution
        x = tf.linalg.solve(A, b)  # n×1
        return tf.reshape(x, (-1,)).numpy(), "unique"

    # Infinite solutions — return minimum-norm solution
    x = tf.linalg.lstsq(A, b, fast=False)  # (n_cols × 1)
    x = tf.reshape(x, (n_cols,))
    return x.numpy(), "infinite_min_norm"



def _add_common_angle_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--angle", type=float, required=True, help="Rotation angle")
    deg_group = p.add_mutually_exclusive_group()
    deg_group.add_argument("--degrees", action="store_true", help="Angle in degrees (default)")
    deg_group.add_argument("--radians", action="store_true", help="Angle in radians")


def cmd_rotate_point(args: argparse.Namespace) -> int:
    deg = not args.radians
    P = tf.constant([[args.x, args.y]], dtype=DTYPE)  # 1×2
    P_rot = rotate_points_tf(P, args.angle, degrees=deg)
    x, y = [float(v) for v in tf.reshape(P_rot, (-1,)).numpy()]
    print(f"Rotated: ({x:.10f}, {y:.10f})")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    if len(args.points) % 2 != 0:
        print("ERROR: The number of values in --points must be even (x1 y1 x2 y2 ...).")
        return 2
    pts = np.array(args.points, dtype=np.float64).reshape(-1, 2)
    deg = not args.radians
    out = rotate_points_tf(pts, args.angle, degrees=deg).numpy()
    np.set_printoptions(suppress=False, precision=10)
    print(out)
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    if args.n is None:
        try:
            n = _infer_n_from_flat_matrix_len(len(args.A))
        except ValueError as e:
            print(f"ERROR: {e}")
            return 2
    else:
        n = args.n

    if len(args.b) != n:
        print(f"ERROR: Expected vector b of length {n}, got {len(args.b)}.")
        return 2

    x, status = solve_linear_tf(args.A, args.b, n=n, rtol=args.rtol, atol=args.atol)

    if status == "none":
        print("No solutions: rank(A) != rank([A|b]).")
        return 2

    if status == "unique":
        print("Unique solution x =", x)
        return 0

    if status == "infinite_min_norm":
        print("Infinitely many solutions — returning a minimum-norm solution x =", x)
        return 0

    print("Unexpected status:", status)
    return 3


def cmd_self_test(_: argparse.Namespace) -> int:
    # 1) Rotate (1, 0) by 90° → (0, 1)
    v = rotate_points_tf([[1.0, 0.0]], 90.0, degrees=True).numpy().ravel()
    assert np.allclose(v, [0.0, 1.0], atol=1e-9), f"Rotation failed: {v}"

    # 2) R(0°) should be identity
    R0 = rotation_matrix_tf(0.0, degrees=True).numpy()
    assert np.allclose(R0, np.eye(2)), f"Rotation matrix 0° failed: {R0}"

    # 3) System with a unique solution
    A = [3.0, 2.0, 1.0, 2.0]  # [[3,2],[1,2]]
    b = [5.0, 5.0]
    x, status = solve_linear_tf(A, b)
    assert status == "unique"
    assert np.allclose(np.dot(np.array(A).reshape(2, 2), x), b)

    # 4) Inconsistent system: x + y = 1, 2x + 2y = 3
    A = [1.0, 1.0, 2.0, 2.0]
    b = [1.0, 3.0]
    x, status = solve_linear_tf(A, b)
    assert status == "none"

    # 5) Infinitely many solutions: x + y = 1, 2x + 2y = 2
    A = [1.0, 1.0, 2.0, 2.0]
    b = [1.0, 2.0]
    x, status = solve_linear_tf(A, b)
    assert status == "infinite_min_norm"
    A_np = np.array(A).reshape(2, 2)
    assert np.allclose(A_np @ x, np.array(b), atol=1e-9)

    print("Self-test OK ✔")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tensor operations: 2D rotations and linear systems (TF + NumPy)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # rotate-point
    rp = sub.add_parser("rotate-point", help="Rotate a single point (x, y) around (0, 0)")
    rp.add_argument("--x", type=float, required=True)
    rp.add_argument("--y", type=float, required=True)
    _add_common_angle_flags(rp)
    rp.set_defaults(func=cmd_rotate_point)

    # rotate — multiple points
    r = sub.add_parser("rotate", help="Rotate multiple points (list: x1 y1 x2 y2 ...)")
    r.add_argument("--points", type=float, nargs="+", required=True)
    _add_common_angle_flags(r)
    r.set_defaults(func=cmd_rotate)

    # solve — Ax=b
    s = sub.add_parser("solve", help="Solve a linear system Ax=b")
    s.add_argument("--A", type=float, nargs="+", required=True, help="Matrix A flattened row-wise")
    s.add_argument("--b", type=float, nargs="+", required=True, help="Right-hand side vector b")
    s.add_argument("--n", type=int, help="Matrix size (if A is not an n×n square, provide n)")
    s.add_argument("--rtol", type=float, default=1e-7)
    s.add_argument("--atol", type=float, default=1e-9)
    s.set_defaults(func=cmd_solve)

    # self-test
    t = sub.add_parser("self-test", help="Run a quick correctness check")
    t.set_defaults(func=cmd_self_test)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # only an entry point, no top-level side effects
    raise SystemExit(main())

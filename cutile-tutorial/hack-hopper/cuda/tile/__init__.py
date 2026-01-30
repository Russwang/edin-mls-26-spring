"""
cuTile Compatibility Layer for Non-Blackwell GPUs (Hopper Hack)

This module provides a drop-in replacement for cuda.tile that works on
older GPUs (Ada Lovelace sm_89, Ampere sm_80, etc.) by using CuPy.

Strategy: Interpret student's kernel code by simulating block execution.
For each block in the grid, we execute the kernel with appropriate bid values
and translate ct.load/ct.store to CuPy array operations.
"""

import builtins
import cupy as cp
import numpy as np
import math
from typing import Callable, Tuple, Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
import threading

# Save Python builtins before we override them
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_sum = builtins.sum
_builtin_pow = builtins.pow

# =============================================================================
# Data Types
# =============================================================================

class DType:
    """Base class for cuTile data types."""
    pass

# Data type singletons
class _Int8(DType):
    name = "int8"
    ctype = "signed char"
    nptype = np.int8
int8 = _Int8()

class _Int16(DType):
    name = "int16"
    ctype = "short"
    nptype = np.int16
int16 = _Int16()

class _Int32(DType):
    name = "int32"
    ctype = "int"
    nptype = np.int32
int32 = _Int32()

class _Int64(DType):
    name = "int64"
    ctype = "long long"
    nptype = np.int64
int64 = _Int64()

class _UInt8(DType):
    name = "uint8"
    ctype = "unsigned char"
    nptype = np.uint8
uint8 = _UInt8()

class _UInt16(DType):
    name = "uint16"
    ctype = "unsigned short"
    nptype = np.uint16
uint16 = _UInt16()

class _UInt32(DType):
    name = "uint32"
    ctype = "unsigned int"
    nptype = np.uint32
uint32 = _UInt32()

class _UInt64(DType):
    name = "uint64"
    ctype = "unsigned long long"
    nptype = np.uint64
uint64 = _UInt64()

class _Float16(DType):
    name = "float16"
    ctype = "__half"
    nptype = np.float16
float16 = _Float16()

class _Float32(DType):
    name = "float32"
    ctype = "float"
    nptype = np.float32
float32 = _Float32()

class _Float64(DType):
    name = "float64"
    ctype = "double"
    nptype = np.float64
float64 = _Float64()

class _BFloat16(DType):
    name = "bfloat16"
    ctype = "__nv_bfloat16"
    nptype = np.float16
bfloat16 = _BFloat16()

class _TFloat32(DType):
    name = "tfloat32"
    ctype = "float"
    nptype = np.float32
tfloat32 = _TFloat32()

class _Bool(DType):
    name = "bool"
    ctype = "bool"
    nptype = np.bool_
bool_ = _Bool()

class _Float8E4M3FN(DType):
    name = "float8_e4m3fn"
    ctype = "__nv_fp8_e4m3"
    nptype = np.float16
float8_e4m3fn = _Float8E4M3FN()

class _Float8E5M2(DType):
    name = "float8_e5m2"
    ctype = "__nv_fp8_e5m2"
    nptype = np.float16
float8_e5m2 = _Float8E5M2()


def _dtype_to_nptype(dtype):
    """Convert cuTile dtype to numpy dtype."""
    if isinstance(dtype, DType):
        return dtype.nptype
    if dtype is None:
        return None
    return np.dtype(dtype)


# =============================================================================
# Type Annotations
# =============================================================================

class Constant:
    """Type annotation for compile-time constants."""
    def __class_getitem__(cls, item):
        return item


class ConstantAnnotation:
    """Marker for constant annotations."""
    pass


class Array:
    """Type annotation for arrays."""
    def __class_getitem__(cls, item):
        return item


class Scalar:
    """Type annotation for scalars."""
    def __class_getitem__(cls, item):
        return item


class Tile:
    """Type annotation for tiles."""
    def __class_getitem__(cls, item):
        return item


class ByTarget:
    """Target-specific configuration."""
    def __class_getitem__(cls, item):
        return item


# =============================================================================
# Enums
# =============================================================================

class MemoryOrder:
    relaxed = "relaxed"
    acquire = "acquire"
    release = "release"
    acq_rel = "acq_rel"
    seq_cst = "seq_cst"


class MemoryScope:
    system = "system"
    device = "device"
    block = "block"


class PaddingMode:
    zeros = "zeros"
    reflect = "reflect"
    replicate = "replicate"


class RoundingMode:
    nearest = "nearest"
    down = "down"
    up = "up"
    truncate = "truncate"


# =============================================================================
# Exceptions
# =============================================================================

class TileCompilerError(Exception):
    """Base class for tile compiler errors."""
    pass


class TileCompilerExecutionError(TileCompilerError):
    """Raised when tile compiler execution fails."""
    pass


class TileCompilerTimeoutError(TileCompilerError):
    """Raised when tile compiler times out."""
    pass


class TileInternalError(TileCompilerError):
    """Raised for internal errors."""
    pass


class TileSyntaxError(TileCompilerError):
    """Raised for syntax errors in tile code."""
    pass


class TileTypeError(TileCompilerError):
    """Raised for type errors in tile code."""
    pass


class TileValueError(TileCompilerError):
    """Raised for value errors in tile code."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def cdiv(a: int, b: int) -> int:
    """Ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


# =============================================================================
# Execution Context - Thread-local storage for current block ID
# =============================================================================

class _ExecutionContext(threading.local):
    """Thread-local execution context for kernel simulation."""
    def __init__(self):
        self.block_id = (0, 0, 0)  # Current block ID (x, y, z)
        self.grid = (1, 1, 1)       # Grid dimensions
        self.in_kernel = False      # Whether we're inside a kernel

_ctx = _ExecutionContext()


@contextmanager
def _kernel_context(block_id: Tuple[int, int, int], grid: Tuple[int, int, int]):
    """Context manager for kernel execution."""
    old_block_id = _ctx.block_id
    old_grid = _ctx.grid
    old_in_kernel = _ctx.in_kernel

    _ctx.block_id = block_id
    _ctx.grid = grid
    _ctx.in_kernel = True
    try:
        yield
    finally:
        _ctx.block_id = old_block_id
        _ctx.grid = old_grid
        _ctx.in_kernel = old_in_kernel


# =============================================================================
# Tile Operations - These are called during kernel execution
# =============================================================================

def bid(dim: int) -> int:
    """Get block ID in given dimension."""
    if not _ctx.in_kernel:
        raise RuntimeError("bid() can only be called within a kernel")
    return _ctx.block_id[dim]


def num_blocks(dim: int) -> int:
    """Get number of blocks in given dimension."""
    if not _ctx.in_kernel:
        raise RuntimeError("num_blocks() can only be called within a kernel")
    return _ctx.grid[dim]


def num_tiles(dim: int) -> int:
    """Get number of tiles in given dimension."""
    return num_blocks(dim)


def load(array, index: Tuple, shape: Tuple, **kwargs):
    """
    Load a tile from global memory.

    index: tuple of tile indices (not element indices)
    shape: shape of the tile to load

    For 1D: load(arr, index=(pid,), shape=(tile_size,))
        -> arr[pid*tile_size : (pid+1)*tile_size]

    For 2D: load(arr, index=(pid_y, pid_x), shape=(tile_h, tile_w))
        -> arr[pid_y*tile_h:(pid_y+1)*tile_h, pid_x*tile_w:(pid_x+1)*tile_w]
    """
    if not _ctx.in_kernel:
        raise RuntimeError("load() can only be called within a kernel")

    ndim = len(index)
    slices = []

    for i in range(ndim):
        tile_idx = index[i]
        tile_size = shape[i]
        start = tile_idx * tile_size
        end = start + tile_size

        # Handle boundary: clamp to array size
        if i < array.ndim:
            end = _builtin_min(end, array.shape[i])

        slices.append(slice(start, end))

    tile = array[tuple(slices)]

    # If tile is smaller than requested shape (boundary), pad with zeros
    actual_shape = tile.shape
    if actual_shape != shape:
        padded = cp.zeros(shape, dtype=tile.dtype)
        # Copy what we have
        copy_slices = tuple(slice(0, s) for s in actual_shape)
        padded[copy_slices] = tile
        tile = padded

    return tile


def store(array, index: Tuple, tile):
    """
    Store a tile to global memory.

    index: tuple of tile indices
    tile: the tile data to store
    """
    if not _ctx.in_kernel:
        raise RuntimeError("store() can only be called within a kernel")

    ndim = len(index)
    shape = tile.shape
    slices = []
    tile_slices = []

    for i in range(ndim):
        tile_idx = index[i]
        tile_size = shape[i]
        start = tile_idx * tile_size
        end = start + tile_size

        # Handle boundary: clamp to array size
        if i < array.ndim:
            actual_end = _builtin_min(end, array.shape[i])
            slices.append(slice(start, actual_end))
            tile_slices.append(slice(0, actual_end - start))
        else:
            slices.append(slice(start, end))
            tile_slices.append(slice(None))

    array[tuple(slices)] = tile[tuple(tile_slices)]


def full(shape: Tuple, value, dtype=None):
    """Create a tile filled with a value."""
    if not _ctx.in_kernel:
        raise RuntimeError("full() can only be called within a kernel")

    np_dtype = _dtype_to_nptype(dtype) if dtype else None
    return cp.full(shape, value, dtype=np_dtype)


def zeros(shape: Tuple, dtype=None):
    """Create a tile filled with zeros."""
    if not _ctx.in_kernel:
        raise RuntimeError("zeros() can only be called within a kernel")

    np_dtype = _dtype_to_nptype(dtype) if dtype else cp.float32
    return cp.zeros(shape, dtype=np_dtype)


def ones(shape: Tuple, dtype=None):
    """Create a tile filled with ones."""
    if not _ctx.in_kernel:
        raise RuntimeError("ones() can only be called within a kernel")

    np_dtype = _dtype_to_nptype(dtype) if dtype else cp.float32
    return cp.ones(shape, dtype=np_dtype)


def arange(start, stop=None, step=1, dtype=None):
    """Create a tile with evenly spaced values."""
    if not _ctx.in_kernel:
        raise RuntimeError("arange() can only be called within a kernel")

    np_dtype = _dtype_to_nptype(dtype)
    if stop is None:
        return cp.arange(start, step=step, dtype=np_dtype)
    return cp.arange(start, stop, step, dtype=np_dtype)


def astype(tile, dtype):
    """Convert tile to specified data type."""
    if not _ctx.in_kernel:
        raise RuntimeError("astype() can only be called within a kernel")

    np_dtype = _dtype_to_nptype(dtype)
    return tile.astype(np_dtype)


def transpose(tile, axes=None):
    """Transpose a tile."""
    if not _ctx.in_kernel:
        raise RuntimeError("transpose() can only be called within a kernel")
    return cp.transpose(tile, axes)


def permute(tile, axes):
    """Permute tile dimensions."""
    if not _ctx.in_kernel:
        raise RuntimeError("permute() can only be called within a kernel")
    return cp.transpose(tile, axes)


def reshape(tile, shape):
    """Reshape a tile."""
    if not _ctx.in_kernel:
        raise RuntimeError("reshape() can only be called within a kernel")
    return cp.reshape(tile, shape)


def broadcast_to(tile, shape):
    """Broadcast tile to shape."""
    if not _ctx.in_kernel:
        raise RuntimeError("broadcast_to() can only be called within a kernel")
    return cp.broadcast_to(tile, shape)


def expand_dims(tile, axis):
    """Expand tile dimensions."""
    if not _ctx.in_kernel:
        raise RuntimeError("expand_dims() can only be called within a kernel")
    return cp.expand_dims(tile, axis)


def cat(tiles, axis=0):
    """Concatenate tiles."""
    if not _ctx.in_kernel:
        raise RuntimeError("cat() can only be called within a kernel")
    return cp.concatenate(tiles, axis)


def bitcast(tile, dtype):
    """Bitcast tile to dtype."""
    if not _ctx.in_kernel:
        raise RuntimeError("bitcast() can only be called within a kernel")
    np_dtype = _dtype_to_nptype(dtype)
    return tile.view(np_dtype)


def extract(tile, indices):
    """Extract elements from tile."""
    if not _ctx.in_kernel:
        raise RuntimeError("extract() can only be called within a kernel")
    return tile[indices]


def gather(array, indices, axis=0):
    """Gather elements from array."""
    if not _ctx.in_kernel:
        raise RuntimeError("gather() can only be called within a kernel")
    return cp.take(array, indices, axis=axis)


def scatter(array, indices, tile, axis=0):
    """Scatter tile to array."""
    if not _ctx.in_kernel:
        raise RuntimeError("scatter() can only be called within a kernel")
    cp.put(array, indices, tile)


def where(condition, x, y):
    """Conditional selection."""
    if not _ctx.in_kernel:
        raise RuntimeError("where() can only be called within a kernel")
    return cp.where(condition, x, y)


# =============================================================================
# Math Functions
# =============================================================================

def exp(x, **kwargs):
    if not _ctx.in_kernel:
        raise RuntimeError("exp() can only be called within a kernel")
    return cp.exp(x)

def exp2(x, **kwargs):
    if not _ctx.in_kernel:
        raise RuntimeError("exp2() can only be called within a kernel")
    return cp.exp2(x)

def log(x):
    if not _ctx.in_kernel:
        raise RuntimeError("log() can only be called within a kernel")
    return cp.log(x)

def log2(x):
    if not _ctx.in_kernel:
        raise RuntimeError("log2() can only be called within a kernel")
    return cp.log2(x)

def sqrt(x):
    if not _ctx.in_kernel:
        raise RuntimeError("sqrt() can only be called within a kernel")
    return cp.sqrt(x)

def rsqrt(x):
    if not _ctx.in_kernel:
        raise RuntimeError("rsqrt() can only be called within a kernel")
    return 1.0 / cp.sqrt(x)

def sin(x):
    if not _ctx.in_kernel:
        raise RuntimeError("sin() can only be called within a kernel")
    return cp.sin(x)

def cos(x):
    if not _ctx.in_kernel:
        raise RuntimeError("cos() can only be called within a kernel")
    return cp.cos(x)

def tan(x):
    if not _ctx.in_kernel:
        raise RuntimeError("tan() can only be called within a kernel")
    return cp.tan(x)

def sinh(x):
    if not _ctx.in_kernel:
        raise RuntimeError("sinh() can only be called within a kernel")
    return cp.sinh(x)

def cosh(x):
    if not _ctx.in_kernel:
        raise RuntimeError("cosh() can only be called within a kernel")
    return cp.cosh(x)

def tanh(x):
    if not _ctx.in_kernel:
        raise RuntimeError("tanh() can only be called within a kernel")
    return cp.tanh(x)

def floor(x):
    if not _ctx.in_kernel:
        raise RuntimeError("floor() can only be called within a kernel")
    return cp.floor(x)

def ceil(x):
    if not _ctx.in_kernel:
        raise RuntimeError("ceil() can only be called within a kernel")
    return cp.ceil(x)

def pow(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("pow() can only be called within a kernel")
    return cp.power(x, y)

def abs(x):
    if not _ctx.in_kernel:
        raise RuntimeError("abs() can only be called within a kernel")
    return cp.abs(x)


# =============================================================================
# Reduction Functions
# =============================================================================

def sum(x, axis=None, keepdims=False):
    if not _ctx.in_kernel:
        raise RuntimeError("sum() can only be called within a kernel")
    return cp.sum(x, axis=axis, keepdims=keepdims)

def prod(x, axis=None):
    if not _ctx.in_kernel:
        raise RuntimeError("prod() can only be called within a kernel")
    return cp.prod(x, axis=axis)

def min(x, axis=None, keepdims=False):
    if not _ctx.in_kernel:
        raise RuntimeError("min() can only be called within a kernel")
    return cp.min(x, axis=axis, keepdims=keepdims)

def max(x, axis=None, keepdims=False):
    if not _ctx.in_kernel:
        raise RuntimeError("max() can only be called within a kernel")
    return cp.max(x, axis=axis, keepdims=keepdims)

def argmin(x, axis=None):
    if not _ctx.in_kernel:
        raise RuntimeError("argmin() can only be called within a kernel")
    return cp.argmin(x, axis=axis)

def argmax(x, axis=None):
    if not _ctx.in_kernel:
        raise RuntimeError("argmax() can only be called within a kernel")
    return cp.argmax(x, axis=axis)

def cumsum(x, axis=None):
    if not _ctx.in_kernel:
        raise RuntimeError("cumsum() can only be called within a kernel")
    return cp.cumsum(x, axis=axis)

def cumprod(x, axis=None):
    if not _ctx.in_kernel:
        raise RuntimeError("cumprod() can only be called within a kernel")
    return cp.cumprod(x, axis=axis)

def minimum(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("minimum() can only be called within a kernel")
    return cp.minimum(x, y)

def maximum(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("maximum() can only be called within a kernel")
    return cp.maximum(x, y)


# =============================================================================
# Binary Operations
# =============================================================================

def add(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("add() can only be called within a kernel")
    return x + y

def sub(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("sub() can only be called within a kernel")
    return x - y

def mul(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("mul() can only be called within a kernel")
    return x * y

def truediv(x, y, **kwargs):
    if not _ctx.in_kernel:
        raise RuntimeError("truediv() can only be called within a kernel")
    return x / y

def floordiv(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("floordiv() can only be called within a kernel")
    return x // y

def mod(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("mod() can only be called within a kernel")
    return x % y

def negative(x):
    if not _ctx.in_kernel:
        raise RuntimeError("negative() can only be called within a kernel")
    return -x


# =============================================================================
# Comparison Operations
# =============================================================================

def equal(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("equal() can only be called within a kernel")
    return x == y

def not_equal(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("not_equal() can only be called within a kernel")
    return x != y

def less(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("less() can only be called within a kernel")
    return x < y

def less_equal(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("less_equal() can only be called within a kernel")
    return x <= y

def greater(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("greater() can only be called within a kernel")
    return x > y

def greater_equal(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("greater_equal() can only be called within a kernel")
    return x >= y


# =============================================================================
# Bitwise Operations
# =============================================================================

def bitwise_and(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_and() can only be called within a kernel")
    return x & y

def bitwise_or(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_or() can only be called within a kernel")
    return x | y

def bitwise_xor(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_xor() can only be called within a kernel")
    return x ^ y

def bitwise_not(x):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_not() can only be called within a kernel")
    return ~x

def bitwise_lshift(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_lshift() can only be called within a kernel")
    return x << y

def bitwise_rshift(x, y):
    if not _ctx.in_kernel:
        raise RuntimeError("bitwise_rshift() can only be called within a kernel")
    return x >> y


# =============================================================================
# Matrix Operations
# =============================================================================

def matmul(a, b):
    if not _ctx.in_kernel:
        raise RuntimeError("matmul() can only be called within a kernel")
    return cp.matmul(a, b)

def mma(a, b, c):
    """Matrix multiply-accumulate: c += a @ b"""
    if not _ctx.in_kernel:
        raise RuntimeError("mma() can only be called within a kernel")
    return c + cp.matmul(a, b)


# =============================================================================
# Atomic Operations
# =============================================================================

def atomic_add(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_add() can only be called within a kernel")
    # For simulation, just do regular add (not truly atomic in Python)
    array[index] += value
    return array[index]

def atomic_and(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_and() can only be called within a kernel")
    array[index] &= value
    return array[index]

def atomic_or(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_or() can only be called within a kernel")
    array[index] |= value
    return array[index]

def atomic_xor(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_xor() can only be called within a kernel")
    array[index] ^= value
    return array[index]

def atomic_min(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_min() can only be called within a kernel")
    array[index] = _builtin_min(array[index], value)
    return array[index]

def atomic_max(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_max() can only be called within a kernel")
    array[index] = _builtin_max(array[index], value)
    return array[index]

def atomic_xchg(array, index, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_xchg() can only be called within a kernel")
    old = array[index]
    array[index] = value
    return old

def atomic_cas(array, index, compare, value):
    if not _ctx.in_kernel:
        raise RuntimeError("atomic_cas() can only be called within a kernel")
    old = array[index]
    if old == compare:
        array[index] = value
    return old


# =============================================================================
# Debug Functions
# =============================================================================

def printf(fmt, *args):
    if not _ctx.in_kernel:
        raise RuntimeError("printf() can only be called within a kernel")
    print(fmt % args)

def assert_(condition, msg=""):
    if not _ctx.in_kernel:
        raise RuntimeError("assert_() can only be called within a kernel")
    assert condition, msg


# =============================================================================
# Kernel Wrapper and Launch
# =============================================================================

class _KernelWrapper:
    """Wrapper for cuTile kernels."""

    def __init__(self, func: Callable, **options):
        self.func = func
        self.name = func.__name__
        self.options = options

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")


def kernel(func: Callable = None, /, **kwargs) -> _KernelWrapper:
    """Decorator to mark a function as a cuTile kernel."""
    if func is None:
        def decorator(f):
            return _KernelWrapper(f, **kwargs)
        return decorator
    return _KernelWrapper(func, **kwargs)


def function(func=None, /, *, host=False, tile=True):
    """Decorator for tile functions."""
    def decorator(func):
        if host:
            return func
        else:
            @wraps(func)
            def wrapped(*args, **kwargs):
                # Allow calling if we're inside a kernel
                if _ctx.in_kernel:
                    return func(*args, **kwargs)
                raise RuntimeError('Tile functions can only be called from tile code.')
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


def launch(stream, grid: Tuple[int, ...], kernel_func: _KernelWrapper, args: Tuple):
    """
    Launch a cuTile kernel by simulating execution for each block.

    This interprets the student's kernel code directly by:
    1. Iterating over all blocks in the grid
    2. Setting up the execution context with current block ID
    3. Calling the kernel function which uses ct.bid(), ct.load(), ct.store()
    """
    if not isinstance(kernel_func, _KernelWrapper):
        raise TypeError("kernel_func must be decorated with @ct.kernel")

    # Normalize grid to 3D
    grid_x = grid[0] if len(grid) > 0 else 1
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    grid_3d = (grid_x, grid_y, grid_z)

    # Execute kernel for each block
    for bz in range(grid_z):
        for by in range(grid_y):
            for bx in range(grid_x):
                block_id = (bx, by, bz)
                with _kernel_context(block_id, grid_3d):
                    kernel_func.func(*args)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core
    "kernel", "function", "launch", "cdiv",

    # Type annotations
    "Constant", "ConstantAnnotation", "Array", "Scalar", "Tile", "ByTarget",

    # Data types
    "DType", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "bfloat16", "tfloat32", "bool_",
    "float8_e4m3fn", "float8_e5m2",

    # Enums
    "MemoryOrder", "MemoryScope", "PaddingMode", "RoundingMode",

    # Exceptions
    "TileCompilerError", "TileCompilerExecutionError",
    "TileCompilerTimeoutError", "TileInternalError",
    "TileSyntaxError", "TileTypeError", "TileValueError",

    # Tile operations
    "bid", "num_blocks", "num_tiles",
    "load", "store", "full", "zeros", "ones", "arange",
    "astype", "transpose", "permute", "reshape",
    "broadcast_to", "expand_dims", "cat", "bitcast",
    "extract", "gather", "scatter", "where",

    # Math
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "floor", "ceil", "pow", "abs",

    # Reductions
    "sum", "prod", "min", "max", "argmin", "argmax",
    "cumsum", "cumprod", "minimum", "maximum",

    # Binary ops
    "add", "sub", "mul", "truediv", "floordiv", "mod", "negative",

    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

    # Bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_lshift", "bitwise_rshift",

    # Matrix
    "matmul", "mma",

    # Atomic
    "atomic_add", "atomic_and", "atomic_or", "atomic_xor",
    "atomic_min", "atomic_max", "atomic_xchg", "atomic_cas",

    # Debug
    "printf", "assert_",
]

# Print info on import
import sys
if not hasattr(sys, '_cutile_compat_warned'):
    print("[cuTile Compat] Using Hopper compatibility layer (interpreter mode)")
    sys._cutile_compat_warned = True

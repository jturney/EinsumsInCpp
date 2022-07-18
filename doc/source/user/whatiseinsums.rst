.. _whatiseinsums:

****************
What is Einsums?
****************

Einsums is a package for scientific computing in C++. It is a C++
library that provides a multidimensional array object and an
assortment of routines for fast operations on array, including
mathematical, shape manipulation, sorting, I/O, basic linear algebra,
and much more.

At the core of the Einsums package, is the `Tensor` object. This
encapsulates *n*-dimensional array of homogeneous data types, with
many operations being analyzed at compile-time for performance.

As a simple example, consider the case of multiplying each element
in a 1-D sequence with the corresponding element in another sequence
of the same length. If the data are stored in two C arrays,
``a`` and ``b``, we could iterate over each element (for clarity
we neglect variable declarations and initializations, memory
allocation, etc.)::

  for (i = 0; i < rows; i++) {
    c[i] = a[i]*b[i];
  }

However,, the coding work required increases with the dimensionality
of our data. In the case of a 2-D array, for example, the C code
(abridged as before) expands to::

  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++) {
      c[i][j] = a[i][j]*b[i][j];
    }
  }

By default, in C/C++, this would execute in a single thread. Einsums
allows us to describe the operations. The framework will at compile-
time determine an algorithm for computing the quantity (be it a BLAS
call or a threaded internal algorithm). In Einsums,::

  einsum(Indices{i, j}, &c, Indices{i, j}, a, Indices{i, j}, b);

does what the earlier examples do, at near-optimal speeds.
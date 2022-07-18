==================
Einsums Quickstart
==================

.. currentmodule:: einsums

Prerequisites
=============

You'll need to know a bit of C++.

**Learning Objective**

After reading, you should be able to:

- Understand the difference between one-, two-, and n-dimensional arrays in Einsums;
- Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;

.. _quickstart.the-basics:

The Basics
==========

Einsums' main object is the homogeneous multidimensional array, a.k.a.
tensor. It is a list of elements, all of the same type, indexed by a
set of non-negative integers.

Einsums' array class is called ``Tensor`` a templated C++ class. Some
important attributes of the ``Tensor`` object are:

Tensor::dims(int d)
    the number of elements per dimension ``d`` of the tensor.
Tensor::size()
    the total number of elements of the array. This is equal to the
    product of the elements of ``shape``.
Tensor::stride(int d)
    the number of elements that need to be skipped per dimension to
    access the next element of that dimension.
Tensor::name()
    a descriptive name for the tensor. Used when printing to aid the user.

An example
----------

::

  #include "einsums/Tensor.hpp"
  #include "einsums/Utilities.hpp"
  #include "einsums/Print.hpp"

  auto a = einsums::create_incremented_tensor("a", 3, 5);
  println(a);

::

  Name: a
      Type: In Core Tensor
      Data Type: double
      Dims{3 5 }
      Strides{5 1 }

      (0, 0-5):         0.00000000     1.00000000     2.00000000     3.00000000     4.00000000

      (1, 0-5):         5.00000000     6.00000000     7.00000000     8.00000000     9.00000000

      (2, 0-5):        10.00000000    11.00000000    12.00000000    13.00000000    14.00000000

Here is this simple example, we included three header files. The first, ``einsums/Tensor.hpp``
provides access to einsums Tensor object. The second, ``einsums/Utilties.hpp`` provides
conveniant tensor creation functions. In this example, we use the ``create_incremented_tensor``
function which returns a rank 2 tensor of doubles with each element being one more than the
element before it. Finally, ``einsums/Print.hpp`` provides the printing functionality of einsums.
It provides wrappers for printing Tensors and other objects. It is also, thread-safe. When it
detects that ``println`` was called in an OpenMP thread block, it will print the corresponding
thread-id. In our example here, ``println`` reported, the name, type, data type, dimensions, strides,
and individual element information about our tensor.
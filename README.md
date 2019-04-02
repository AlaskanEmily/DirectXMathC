DirectX Math (for C)
====================

This is a modified version of the DirectX Math library, which was originally
released by Microsoft under the MIT license for C++.

This port is still incomplete, but has been tested for some basic uses.

This version has been translated to C, had any Visual C/C++ specific
features put into conditional defines to allow compiling with any C compiler
(include with ANSI options), and has been tested on platforms other than
Windows.

The intent is to allow the use of DirectX Math vectors and matrices to be
used in cross-platform projects, wrapped for use in languages that can only
interface with C (such as Mercury and Python), and to allow use with older
or less capable C/C++ compilers such as GCC 2 on Haiku or Watcom C/C++.

It's plausible to add inline x86/x87 ASM for use with Watcom and additional
intrinsics or inline ASM for other platforms. Work to use inline assembly
for Watcom hasn't started yet.

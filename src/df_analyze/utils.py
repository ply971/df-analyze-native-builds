import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat
from typing import Callable


class Debug:
    """Printing mixin a la https://doc.rust-lang.org/std/fmt/trait.Debug.html"""

    def __repr__(self) -> str:
        return "".join(pformat(self.__dict__, indent=2, width=80, compact=False))


"""
https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

A common task in Python (especially while testing or debugging) is to
redirect sys.stdout to a stream or a file while executing some piece of code.
However, simply "redirecting stdout" is sometimes not as easy as one would
expect; hence the slightly strange title of this post. In particular, things
become interesting when you want C code running within your Python process
(including, but not limited to, Python modules implemented as C extensions)
to also have its stdout redirected according to your wish. This turns out to
be tricky and leads us into the interesting world of file descriptors,
buffers and system calls.


# Detour - on file descriptors and streams

This section dives into some internals of the operating system, the C
library, and Python [3]. If you just want to know how to properly redirect
printouts from C in Python, you can safely skip to the next section (though
understanding how the redirection works will be difficult).

Files are opened by the OS, which keeps a system-wide table of open files,
some of which may point to the same underlying disk data (two processes can
have the same file open at the same time, each reading from a different
place, etc.)

File descriptors are another abstraction, which is managed per-process. Each
process has its own table of open file descriptors that point into the
system-wide table. Here's a schematic, taken from The Linux Programming
Interface: File descriptor diagram

File descriptors allow sharing open files between processes (for example when
creating child processes with fork). They're also useful for redirecting from
one entry to another, which is relevant to this post. Suppose that we make
file descriptor 5 a copy of file descriptor 4. Then all writes to 5 will
behave in the same way as writes to 4. Coupled with the fact that the
standard output is just another file descriptor on Unix (usually index 1),
you can see where this is going. The full code is given in the next section.

File descriptors are not the end of the story, however. You can read and
write to them with the read and write system calls, but this is not the way
things are typically done. The C runtime library provides a convenient
abstraction around file descriptors - streams. These are exposed to the
programmer as the opaque FILE structure with a set of functions that act on
it (for example fprintf and fgets).

FILE is a fairly complex structure, but the most important things to know
about it is that it holds a file descriptor to which the actual system calls
are directed, and it provides buffering, to ensure that the system call
(which is expensive) is not called too often. Suppose you emit stuff to a
binary file, a byte or two at a time. Unbuffered writes to the file
descriptor with write would be quite expensive because each write invokes a
system call. On the other hand, using fwrite is much cheaper because the
typical call to this function just copies your data into its internal buffer
and advances a pointer. Only occasionally (depending on the buffer size and
flags) will an actual write system call be issued.

With this information in hand, it should be easy to understand what stdout
actually is for a C program. stdout is a global FILE object kept for us by
the C library, and it buffers output to file descriptor number 1. Calls to
functions like printf and puts add data into this buffer. fflush forces its
flushing to the file descriptor, and so on.

But we're talking about Python here, not C. So how does Python translate
calls to sys.stdout.write to actual output?

Python uses its own abstraction over the underlying file descriptor - a file
object. Moreover, in Python 3 this file object is further wrapped in an
io.TextIOWrapper, because what we pass to print is a Unicode string, but the
underlying write system calls accept binary data, so encoding has to happen
en route.

The important take-away from this is: Python and a C extension loaded by it
(this is similarly relevant to C code invoked via ctypes) run in the same
process, and share the underlying file descriptor for standard output.
However, while Python has its own high-level wrapper around it - sys.stdout,
the C code uses its own FILE object. Therefore, simply replacing sys.stdout
cannot, in principle, affect output from C code. To make the replacement
deeper, we have to touch something shared by the Python and C runtimes - the
file descriptor.


Redirecting with file descriptor duplication

Without further ado, here is an improved stdout_redirector that also
redirects output from C code [4]:

from contextlib import contextmanager
import ctypes
import io
import os, sys
import tempfile

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):

        '''Redirect stdout to the given file descriptor.'''
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)

There are a lot of details here (such as managing the temporary file into
which output is redirected) that may obscure the key approach: using dup and
dup2 to manipulate file descriptors. These functions let us duplicate file
descriptors and make any descriptor point at any file. I won't spend more
time on them - go ahead and read their documentation, if you're interested.
The detour section should provide enough background to understand it.

Let's try this:

f = io.BytesIO()

with stdout_redirector(f):
    print('foobar')
    print(12)
    libc.puts(b'this comes from C')
    os.system('echo and this is from echo')
print('Got stdout: "{0}"'.format(f.getvalue().decode('utf-8')))

Gives us:

Got stdout: "and this is from echo
this comes from C
foobar
12
"

Success! A few things to note:

    The output order may not be what we expected. This is due to buffering.
    If it's important to preserve order between different kinds of output
    (i.e. between C and Python), further work is required to disable
    buffering on all relevant streams. You may wonder why the output of echo
    was redirected at all? The answer is that file descriptors are inherited
    by subprocesses. Since we rigged fd 1 to point to our file instead of the
    standard output prior to forking to echo, this is where its output went.
    We use a BytesIO here. This is because on the lowest level, the file
    descriptors are binary. It may be possible to do the decoding when
    copying from the temporary file into the given stream, but that can hide
    problems. Python has its in-memory understanding of Unicode, but who
    knows what is the right encoding for data printed out from underlying C
    code? This is why this particular redirection approach leaves the
    decoding to the caller. The above also makes this code specific to Python
    3. There's no magic involved, and porting to Python 2 is trivial, but
    some assumptions made here don't hold (such as sys.stdout being a
    io.TextIOWrapper).

# Redirecting the stdout of a child process

We've just seen that the file descriptor duplication approach lets us grab
the output from child processes as well. But it may not always be the most
convenient way to achieve this task. In the general case, you typically use
the subprocess module to launch child processes, and you may launch several
such processes either in a pipe or separately. Some programs will even juggle
multiple subprocesses launched this way in different threads. Moreover, while
these subprocesses are running you may want to emit something to stdout and
you don't want this output to be captured.

So, managing the stdout file descriptor in the general case can be messy; it
is also unnecessary, because there's a much simpler way.

The subprocess module's swiss knife Popen class (which serve as the basis for
much of the rest of the module) accepts a stdout parameter, which we can use
to ask it to get access to the child's stdout:

import subprocess

echo_cmd = ['echo', 'this', 'comes', 'from', 'echo']
proc = subprocess.Popen(echo_cmd, stdout=subprocess.PIPE)
output = proc.communicate()[0]
print('Got stdout:', output)

The subprocess.PIPE argument can be used to set up actual child process pipes
(a la the shell), but in its simplest incarnation it captures the process's
output.

If you only launch a single child process at a time and are interested in its
output, there's an even simpler way:

output = subprocess.check_output(echo_cmd)
print('Got stdout:', output)

check_output will capture and return the child's standard output to you; it
will also raise an exception if the child exist with a non-zero return code.
Conclusion

I hope I covered most of the common cases where "stdout redirection" is
needed in Python. Naturally, all of the same applies to the other standard
output stream - stderr. Also, I hope the background on file descriptors was
sufficiently clear to explain the redirection code; squeezing this topic in
such a short space is challenging. Let me know if any questions remain or if
there's something I could have explained better.

Finally, while it is conceptually simple, the code for the redirector is
quite long; I'll be happy to hear if you find a shorter way to achieve the
same effect.

[1] Do not despair. As of February 2015, a sizable chunk of the worldwide
Python programmers are in the same boat.

[2] Note that bytes passed to puts. This being Python 3, we have to be
careful since libc doesn't understand Python's unicode strings.

[3] The following description focuses on Unix/POSIX systems; also, it's
necessarily partial. Large book chapters have been written on this topic -
I'm just trying to present some key concepts relevant to stream redirection.

For a variant that works on Windows, take a look at this gist:
https://gist.github.com/natedileas/8eb31dc03b76183c0211cdde57791005

[4] The approach taken here is inspired by this Stack Overflow answer.


"""


"""
See below for how hard it is to stream outputs from C programs

https://github.com/tensorflow/community/pull/14

Working example: (Wurlitzer)

https://github.com/minrk/wurlitzer/blob/main/wurlitzer.py
"""


def spammer(id: int) -> None:
    time.sleep(0.05)
    print(f"I should be in stderr. Process={id:>4d}", file=sys.stderr, flush=True)
    print(f"I should be in stdout. Process={id:>4d}", file=sys.stdout, flush=True)


def run_captured(pipe_write_fd: int, target: Callable[[], None]):
    # Route all stdout/stderr to the inherited pipe write end
    os.set_inheritable(pipe_write_fd, True)
    # close stdout and stderr, so writes to those files go to the pipe
    os.dup2(pipe_write_fd, 1, inheritable=True)
    os.dup2(pipe_write_fd, 2, inheritable=True)
    os.set_inheritable(1, True)
    os.set_inheritable(2, True)

    try:
        target()
        os._exit(0)
    except SystemExit as e:
        os._exit(e.code if isinstance(e.code, int) else 1)
    except Exception:
        traceback.print_exc()
        os._exit(1)


def tee(target, log_path="outputs.log") -> int:
    """
    Run `target(*args, **kwargs)` in a child process.
    Mirror everything written to its stdout/stderr (and all its children)
    to both the parent's terminal and `log_path`. No threads.
    Returns the child's exit code.
    """
    CHUNK = 65536
    # Pipe for child's stdout/stderr
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    # os.dup2(write_fd, 1)
    # os.dup2(write_fd, 2)

    # Save parent's real terminal stdout FD
    stdout_fd = os.dup(1)

    # Start child
    ctx = mp.get_context()  # respects platform defaults (spawn on Win/macOS)
    p = ctx.Process(target=run_captured, args=(write_fd, target))
    p.start()

    # Parent: close write end; pump read -> tty + log
    os.close(write_fd)
    with open(log_path, "ab", buffering=0) as log:
        try:
            while True:
                chunk = os.read(read_fd, CHUNK)
                if not chunk:
                    break
                # os.write(sys.stdout.fileno(), chunk)
                os.write(stdout_fd, chunk)
                log.write(chunk)
                log.flush()
        finally:
            try:
                os.close(read_fd)
            except OSError:
                traceback.print_exc()
                pass
            try:
                ...
                # os.close(stdout_fd)
            except OSError:
                traceback.print_exc()
                pass

    p.join()
    if p.exitcode is None:
        return 0
    return p.exitcode


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull):
    stdout = sys.stdout
    stderr = sys.stderr

    stdout_fd = fileno(stdout)
    stderr_fd = fileno(stderr)  # noqa: F841
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to

        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def main():
    print("parent start")
    with mp.get_context().Pool(4) as pool:
        list(pool.imap(spammer, range(12)))
    print("parent done")


if __name__ == "__main__":
    # rc = tee_run(main, log_path="run.log", merge_stderr=True)
    # sys.exit(rc)
    # main()

    # read_fd, write_fd = os.pipe()
    # print("Read:  ", read_fd)
    # print("Write: ", write_fd)
    # os.close(read_fd)
    # os.close(write_fd)

    # tee(main)
    logfile = Path(".") / "outputs.log"
    main()

    logged = logfile.read_text()
    print("")
    if logged.strip() == "":
        print("Capture failed.")
    else:
        print("Outputs:")
        print(logged)

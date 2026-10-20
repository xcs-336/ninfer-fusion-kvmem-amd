"""Run one GPU test under a shared Windows byte-range lock and a kill-on-close job.

The lock is held until the test and its children have exited. An outer CTest or
terminal cancellation closes the job handle, preventing an orphan GPU process
from outliving the lock owner. No other process or model service is stopped.
"""
import argparse
import ctypes
from ctypes import wintypes
import errno
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--lock-file', required=True, type=Path)
parser.add_argument('--wait-seconds', type=float, default=300)
parser.add_argument('--timeout', type=float, required=True)
parser.add_argument('--yield-seconds', type=float, default=1.5)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command or args.timeout <= 0 or args.wait_seconds < 0 or args.yield_seconds < 0:
    parser.error('a command, positive timeout and nonnegative wait are required')

class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
                ('max_working_set', ctypes.c_size_t), ('active_processes', wintypes.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]

class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops',
                                                   'read_bytes', 'write_bytes', 'other_bytes')]

class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io', IoCounters), ('process_memory', ctypes.c_size_t),
                ('job_memory', ctypes.c_size_t), ('peak_process_memory', ctypes.c_size_t),
                ('peak_job_memory', ctypes.c_size_t)]

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel.CreateJobObjectW.restype = wintypes.HANDLE
kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
kernel.SetInformationJobObject.restype = wintypes.BOOL
kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel.AssignProcessToJobObject.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL

def run_test():
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process = None
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=sys.stdout,
                                   stderr=sys.stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        if not kernel.AssignProcessToJobObject(job, int(process._handle)):
            error = ctypes.WinError(ctypes.get_last_error())
            process.kill()
            process.wait()
            raise error
        print(f'GPU lock acquired; test PID={process.pid}; timeout={args.timeout:g}s', flush=True)
        try:
            return process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f'FAIL: GPU test exceeded {args.timeout:g}s; terminating only its job', flush=True)
            return 124
    finally:
        # Closing the job also stops any surviving children before releasing the lock.
        kernel.CloseHandle(job)
        if process is not None:
            process.wait()

def main():
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open('a+b') as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b'\0')
            lock.flush()
        deadline = time.monotonic() + args.wait_seconds
        acquired = False
        while not acquired:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    print('NOT_RUN: shared GPU lock is busy; no test process was launched', flush=True)
                    return 77
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        try:
            result = run_test()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    # Leave a slot for the other project before a serial CTest suite reacquires.
    time.sleep(args.yield_seconds)
    return result

if __name__ == '__main__':
    sys.exit(main())

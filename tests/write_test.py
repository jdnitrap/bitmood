#!/usr/bin/env python3
"""Drives `cmix-bit write` through a pseudo-terminal, as a person would type.

usage: write_test.py <cmix-bit> <memory.bin> <out.txt>
Exits 0 if the written text and the learned byte count are what the keys imply.
"""
import os
import pty
import re
import select
import subprocess
import sys
import time

binary, memory, out = sys.argv[1:4]

before = 0
if os.path.exists(memory):
    info = subprocess.run([binary, "info", memory], capture_output=True, text=True).stdout
    before = int(re.search(r"bytes learned\s+(\d+)", info).group(1))

pid, fd = pty.fork()
if pid == 0:
    os.execv(binary, [binary, "write", "--state", memory, "--out", out, "--suggest", "12"])

screen = bytearray()


def pump(seconds=0.3):
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            return
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            return
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return
        screen.extend(chunk)


def send(keys):
    os.write(fd, keys)
    pump()


# Wait until write is ready (the header line is drawn), up to 60 s.
deadline = time.time() + 60
while b"Tab: accept" not in screen and time.time() < deadline:
    pump(0.5)
send(b"The ")          # finished word: learned
send(b"\t")            # accept the whole suggestion
send(b" mox")          # word in progress
send(b"\x7f")          # backspace inside the word
send(b"del ")          # -> "model "
send(b"\x1b[C")        # right arrow: accept one word
send(b"\r")            # newline
send(b"end")
send(b"\x04")          # Ctrl-D: save and quit
pump(1.0)
_, status = os.waitpid(pid, 0)

text = open(out, "rb").read().decode("utf-8", "replace")
problems = []
if os.WEXITSTATUS(status) != 0:
    problems.append("exit status %d" % os.WEXITSTATUS(status))
if not text.startswith("The "):
    problems.append("text should start with 'The ': %r" % text)
if " model " not in text or "mox" in text:
    problems.append("backspace inside a word failed: %r" % text)
if "\n" not in text or not text.endswith("end"):
    problems.append("newline / final word missing: %r" % text)
if b"\x1b[90m" not in screen:
    problems.append("no grey suggestion was drawn")

info = subprocess.run([binary, "info", memory], capture_output=True, text=True).stdout
after = int(re.search(r"bytes learned\s+(\d+)", info).group(1))
if after - before != len(text.encode()):
    problems.append("learned %d bytes, wrote %d" % (after - before, len(text.encode())))

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("wrote %r" % text)

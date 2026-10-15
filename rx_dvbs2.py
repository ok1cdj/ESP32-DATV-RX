#!/usr/bin/env python3
"""rx_dvbs2.py: receive a DVB-S2 transmission with an ESP32-S3 and show it.

    ESP32-S3 (espdr-rx, rtl_tcp, 333 ksps) -> leandvb (DVB-S2) -> ffmpeg (retime) -> ffplay
    ESP32-S3 (--wide: espdr.nb, packed 4-bit I/Q, 500 ksps) -> leandvb -> ffmpeg -> ffplay

It starts espdr-rx when no rtl_tcp server is listening (espdr-rx loads the receiver firmware into the
board's RAM when needed), looks for the signal near the given frequency, tunes to its centre and plays
the transport stream. Ctrl-C or closing the video window ends it.

--wide receives up to about 333 kS/s. It needs the receiver firmware with the cs4 format (esp32-sdr-trx
1.0.0rc4+cs4, installed by tools/install.sh); 333 ksps int8 through rtl_tcp is limited to about 200 kS/s.
"""
import argparse
import math
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

try:
    import numpy as np
except ImportError:  # main() re-executes inside .venv
    np = None


def log(*a):
    print(*a, file=sys.stderr, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 333333  # espdr-rx --decim 3
FS_WIDE = 500000  # espdr.nb --decim 2 --format cs4

# DVB-S2 MODCOD numbers (EN 302 307-1, table 12), for leandvb's --modcods filter.
MODCODS = {("QPSK", r): i + 1 for i, r in enumerate(("1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "5/6", "8/9", "9/10"))}
MODCODS.update({("8PSK", r): i + 12 for i, r in enumerate(("3/5", "2/3", "3/4", "5/6", "8/9", "9/10"))})
MODCODS.update({("16APSK", r): i + 18 for i, r in enumerate(("2/3", "3/4", "4/5", "5/6", "8/9", "9/10"))})
MODCODS.update({("32APSK", r): i + 24 for i, r in enumerate(("3/4", "4/5", "5/6", "8/9", "9/10"))})


def plframe_symbols(const, normal=True, pilots=True):
    """Symbols of one PLFRAME: PLHEADER, data slots of 90 symbols, a 36-symbol pilot block after every 16 slots."""
    slots = (64800 if normal else 16200) // {"QPSK": 2, "8PSK": 3, "16APSK": 4, "32APSK": 5}[const] // 90
    return 90 + 90 * slots + (36 * ((slots - 1) // 16) if pilots else 0)


def find_tool(name, *candidates):
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which(name)


class RtlTcp:
    """Minimal rtl_tcp client for espdr-rx."""

    def __init__(self, host, port):
        self.s = socket.create_connection((host, port), timeout=10)
        hdr = self._recv_exact(12)
        if hdr[:4] != b"RTL0":
            raise RuntimeError("not an rtl_tcp server")

    def _recv_exact(self, n):
        b = bytearray()
        while len(b) < n:
            d = self.s.recv(n - len(b))
            if not d:
                raise EOFError("rtl_tcp server closed the connection")
            b += d
        return bytes(b)

    def cmd(self, c, v):
        self.s.sendall(struct.pack(">BI", c, int(v) & 0xFFFFFFFF))

    def setup(self, freq, gain_db):
        self.cmd(0x02, FS)
        self.cmd(0x01, round(freq))
        self.cmd(0x03, 1)  # manual gain
        self.cmd(0x08, 0)  # AGC off
        self.cmd(0x04, round(gain_db * 10))

    def read(self, n):
        return self._recv_exact(n)

    def close(self):
        self.s.close()


class DcBlock:
    """cu8 I/Q from rtl_tcp -> complex64 in leandvb's --u8 units (u8 - 128) without the receiver's own DC.

    The 8-bit stream keeps the ESP32-S3's DC (LO leakage, ADC offset) at 0 Hz; the cs4 mode removes it on the chip. When
    the transmitter's carrier comes within a few kHz of the LO (the C3 drifts by tens of kHz) the line sits inside the
    signal and nothing decodes. A running mean (blocks of ~0.1 s, smoothed) is taken off: a notch a few Hz wide."""

    def __init__(self):
        self.dc = None
        self.odd = b""                                     # half an I/Q pair left from the previous block

    def process(self, data):
        data = self.odd + data
        k = len(data) // 2 * 2
        self.odd = data[k:]
        if not k:
            return np.zeros(0, np.complex64)
        b = np.frombuffer(data[:k], np.uint8).astype(np.float32) - 128.0
        x = (b[0::2] + 1j * b[1::2]).astype(np.complex64)
        m = x.mean()
        self.dc = m if self.dc is None else self.dc + (m - self.dc) * min(1.0, 0.2 * len(x) / 30000)
        return (x - self.dc).astype(np.complex64)


class Nco:
    """Software tuning that follows the transmitter's drift: shifts the samples by -freq (Hz) before leandvb, with a
    continuous phase across blocks. The ESP32-C3 has no TCXO and drifts by some 260 Hz/s while warming up after a
    reset; leandvb itself only follows +-sr/4 (--freq-tol 0.25) and loses the signal beyond it (seen on the bench).
    A first-order loop on leandvb's FREQ (the offset left after the NCO): the NCO frequency ramps at K x residual Hz/s,
    at most RATE_MAX, smoothly block by block - no steps, which upset leandvb's lock at low symbol rates. A linear drift
    of r Hz/s leaves r / K behind (260 Hz/s: about 500 Hz, well inside leandvb's range)."""

    K = 0.5                                                # 1/s
    RATE_MAX = 600.0                                       # Hz/s

    def __init__(self, fs):
        self.fs = fs
        self.freq = 0.0                                    # Hz
        self.rate = 0.0                                    # Hz/s, set by track(), applied by process()
        self.phase = 0.0                                   # cycles
        self.updates = 0
        self.last = None

    def process(self, x):
        n = len(x)
        if self.rate == 0.0 and self.freq == 0.0:
            return x
        dt = n / self.fs
        f0, f1 = self.freq, self.freq + self.rate * dt     # a linear ramp over the block: a continuous frequency
        t = np.arange(n) / self.fs
        ph = self.phase - (f0 * t + 0.5 * (f1 - f0) / dt * t * t)
        self.phase = (self.phase - (f0 + f1) / 2 * dt) % 1.0
        self.freq = f1
        return (x * np.exp(2j * np.pi * ph)).astype(np.complex64)

    def track(self, residual, mer, framelock):
        """Called with leandvb's FREQ, MER and FRAMELOCK. Holds (rate 0) unless the signal is live: on noise a stale lock
        would walk the NCO away."""
        if not framelock or mer is None or not mer > 3:
            self.rate = 0.0
            return
        self.rate = max(-self.RATE_MAX, min(self.RATE_MAX, self.K * residual))
        self.updates += 1


class Decimator:
    """Streaming lowpass + decimation by d of rtl_tcp cu8 I/Q, out as cf32 in leandvb's --u8 units (u8 - 128).

    Low symbol rates arrive with many samples per symbol (333 ksps: 5 at 66.7 kS/s, 8 at 40 kS/s). Decimating to 2..2.5
    samples per symbol looked better with 3 noise seeds, but not with 10 (ESP32-ATV rx/test_rx_decim.py: 40 and 66.7 kS/s,
    Es/N0 9 and 6 dB, about even); what loses lock at low rates is mostly the residual carrier offset. Kept as an option."""

    def __init__(self, fs, sr, d):
        self.d = d
        fp, fstop = 0.675 * sr, fs / d - 0.675 * sr      # keep the signal (roll-off 0.35); above fstop aliases into it
        n = int(np.ceil(5.0 * fs / (fstop - fp))) | 1      # Kaiser window, beta 8: about 80 dB stopband
        t = np.arange(n) - (n - 1) / 2
        fc = (fp + fstop) / 2 / fs
        h = 2 * fc * np.sinc(2 * fc * t) * np.kaiser(n, 8.0)
        self.h = (h / h.sum()).astype(np.float32)
        self.hist = np.zeros(n - 1, np.complex64)          # the last n - 1 input samples
        self.phase = 0                                     # index, in the next block, of the next sample to keep
        self.odd = b""                                     # half an I/Q pair left from the previous block

    def process(self, samples):
        """complex64 samples (DcBlock output) -> decimated cf32 bytes."""
        if not len(samples):
            return b""
        x = np.concatenate((self.hist, samples))
        m = len(samples)
        y = np.convolve(x, self.h, "valid")                # y[i]: filter output at new sample i
        out = y[self.phase::self.d]
        self.phase = (self.phase - m) % self.d
        self.hist = x[len(x) - len(self.hist):]
        return out.astype(np.complex64).tobytes()


class NarrowSource:
    """espdr-rx's rtl_tcp server: 333 ksps cu8 -> without its DC -> cf32 for leandvb (optionally decimated)."""

    fs = FS

    def __init__(self, args):
        self.args = args
        self.dec = None
        self.nco = Nco(FS)

    def decimate(self, sr, d):
        self.dec = Decimator(FS, sr, d)

    def grab(self, freq, seconds):
        rx = RtlTcp(self.args.host, self.args.port)
        rx.setup(freq, self.args.gain)
        rx.read(int(0.3 * FS) * 2)  # retune settling
        b = np.frombuffer(rx.read(int(seconds * FS) * 2), np.uint8).astype(np.float32) - 127.5
        rx.close()
        x = b[0::2] + 1j * b[1::2]
        return x - x.mean()                                # the receiver's DC would look like a narrow signal

    def leandvb_input(self):
        if self.dec:
            return ["--f32", "--float-scale", "1", "-f", str(round(FS / self.dec.d))]
        return ["--f32", "--float-scale", "1", "-f", str(FS)]        # u8 - 128 units, as --u8 would have it

    def start(self, freq, sink):
        """Streams to sink.write() from a thread until the server closes or sink fails."""
        def pump():
            rx = RtlTcp(self.args.host, self.args.port)
            rx.setup(freq, self.args.gain)
            dc = DcBlock()
            try:
                while d := rx.s.recv(65536):
                    x = self.nco.process(dc.process(d))
                    sink.write(self.dec.process(x) if self.dec else x.tobytes())
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                rx.close()
        threading.Thread(target=pump, daemon=True).start()

    def stop(self):
        pass


class WideSource:
    """espdr.nb directly: 500 ksps packed 4-bit I/Q with a gain per packet, delivered as cf32."""

    fs = FS_WIDE

    def __init__(self, args):
        self.args = args
        self.proc = None
        try:
            from espdr import board, nb
        except ImportError:
            sys.exit("espdr not found: run tools/install.sh first (or activate the venv)")
        if "cs4" not in nb.FORMATS:
            sys.exit("the installed espdr has no cs4 format: run tools/install.sh again (it installs esp32-sdr-trx 1.0.0rc4+cs4)")
        self.port = board.ensure(board.RX, port=args.device, log=log)  # loads the receiver firmware into RAM if needed
        link = nb.open_link(self.port)
        status, _ = link.command(nb.NB_SET_FORMAT, nb.FORMATS["cs4"], allow=(0, 2))
        link.ser.close()
        if status != 0:
            log("the running firmware has no cs4 format; loading the one of the installed package")
            self.port = board.ensure(board.RX, port=args.device, load="always", log=log)
        # espdr-rx maps 0..49.6 dB to the gain selector 30..80
        self.selector = round(30 + max(0.0, min(49.6, args.gain)) * 50 / 49.6)
        self.nco = Nco(FS_WIDE)

    def _cmd(self, freq, seconds):
        return [sys.executable, "-m", "espdr.nb", "-p", self.port, "run", "--freq", str(round(freq)), "--decim", "2",
                "--format", "cs4", "--gain", str(self.selector), "--seconds", str(seconds), "--convert", "cf32", "-o", "-"]

    def grab(self, freq, seconds):
        p = subprocess.run(self._cmd(freq, int(seconds) + 1), capture_output=True, timeout=30)
        if p.returncode:
            sys.exit("espdr.nb failed:\n" + p.stderr.decode(errors="ignore"))
        x = np.frombuffer(p.stdout, np.complex64)
        return x[int(0.3 * FS_WIDE):]

    def leandvb_input(self):
        # cf32 in full-scale units (about 0.02 RMS at the usual gain); leandvb's AGC does the rest
        return ["--f32", "--float-scale", "512", "-f", str(FS_WIDE)]

    def start(self, freq, sink):
        self.proc = subprocess.Popen(self._cmd(freq, 0), stdout=subprocess.PIPE,
                                     stderr=None if self.args.verbose else subprocess.DEVNULL)

        def pump():
            odd = b""                                      # bytes of an incomplete cf32 sample
            try:
                while d := self.proc.stdout.read1(65536):
                    d = odd + d
                    k = len(d) // 8 * 8
                    odd = d[k:]
                    sink.write(self.nco.process(np.frombuffer(d[:k], np.complex64)).tobytes())
            except (BrokenPipeError, OSError, ValueError):
                pass
        threading.Thread(target=pump, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # espdr.nb stops the ESP's run cleanly on SIGINT
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def server_up(host, port):
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False


def start_espdr(args):
    espdr = find_tool("espdr-rx", os.path.join(HERE, ".venv", "bin", "espdr-rx"))
    if not espdr:
        sys.exit("espdr-rx not found: run tools/install.sh first (or activate the venv)")
    cmd = [espdr, "--decim", "3", "--listen", f"{args.host}:{args.port}"]
    if args.device:
        cmd += ["--port", args.device]
    if args.ppm:
        cmd += ["--ppm", str(args.ppm)]
    log("starting:", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while time.time() - t0 < 60:
        if p.poll() is not None:
            sys.exit("espdr-rx ended:\n" + p.stdout.read())
        if server_up(args.host, args.port):
            threading.Thread(target=lambda: [None for _ in p.stdout], daemon=True).start()  # drain its output
            return p
        time.sleep(0.5)
    p.terminate()
    sys.exit("espdr-rx did not start within 60 s")


def search(src, freq):
    """Find the signal near freq; return its centre frequency in Hz."""
    fs = src.fs
    x = src.grab(freq, 1.0)
    n = 1024
    seg = len(x) // n
    p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(x[: seg * n].reshape(seg, n) * np.hanning(n), axis=1), axes=1)) ** 2, 0)
    p = np.convolve(10 * np.log10(p + 1e-9), np.ones(9) / 9, "same")
    f = (np.arange(n) - n / 2) * fs / n
    usable = np.abs(f) < 0.45 * fs
    floor = np.percentile(p[usable], 10)
    peak = np.percentile(p[usable], 95)
    if peak - floor < 6:
        log(f"search: no signal near {freq / 1e6:.4f} MHz (only {peak - floor:.1f} dB above the floor)")
        return None
    occ = usable & (p > peak - 3)
    lo, hi = f[occ].min(), f[occ].max()
    centre = freq + (lo + hi) / 2
    log(f"search: signal {peak - floor:.0f} dB above the floor, -3 dB width {(hi - lo) / 1e3:.0f} kHz, "
          f"centre {centre / 1e6:.4f} MHz ({(centre - freq) / 1e3:+.1f} kHz)")
    if hi - lo > 0.8 * fs:
        log(f"search: the signal is wider than the receiver ({fs / 1e3:.0f} ksps)"
            + ("" if src.fs == FS_WIDE else ": use --wide, or set the transmitter to 125 kS/s"))
    return centre


def main():
    # Run inside the project's venv (numpy, espdr) when started as ./rx_dvbs2.py with the system Python.
    venv_python = os.path.join(HERE, ".venv", "bin", "python")
    if os.path.exists(venv_python) and os.path.realpath(sys.prefix) != os.path.realpath(os.path.join(HERE, ".venv")):
        os.execv(venv_python, [venv_python, os.path.abspath(__file__)] + sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=2300.5, help="transmitter frequency in MHz (default 2300.5)")
    ap.add_argument("--sr", type=float, default=125e3,
                    help="symbol rate in S/s (default 125000; at most about 200000, with --wide about 333000)")
    ap.add_argument("--wide", action="store_true", help="500 ksps 4-bit mode for symbol rates above 200 kS/s")
    ap.add_argument("--cr", default="3/4", help="code rate (default 3/4)")
    ap.add_argument("--frame", choices=("normal", "short", "auto"), default="normal",
                    help="DVB-S2 FECFRAME size to decode: normal (64800 bits, default), short (16200) or auto (both; a "
                         "corrupted PLS header may then start an ldpc_tool for the other size)")
    ap.add_argument("--const", default="QPSK", help="constellation (default QPSK)")
    ap.add_argument("--gain", type=float, default=10, help="receiver gain in dB, 0..49.6 (default 10)")
    ap.add_argument("--ppm", type=float, default=0, help="crystal correction passed to espdr-rx")
    ap.add_argument("--no-search", action="store_true", help="tune exactly to --freq, do not look for the signal")
    ap.add_argument("--no-track", action="store_true", help="do not follow the transmitter's drift (software NCO)")
    ap.add_argument("--sr-tol", type=float, default=0.002,
                    help="symbol rate tolerance (leandvb --sr-tol, default 0.002 = 2000 ppm; leandvb's own 100 ppm misses an "
                         "ESP32-C3 loop from the flash at 333333 Bd when tuned to 333000)")
    ap.add_argument("--record", metavar="FILE.ts", help="also write the transport stream to this file")
    ap.add_argument("--no-play", action="store_true", help="do not start ffplay (use with --record or --stdout)")
    ap.add_argument("--stdout", action="store_true", help="write the transport stream to stdout")
    ap.add_argument("--device", help="the receiver's native USB port, e.g. /dev/serial/by-id/usb-Espressif_..._<MAC>-if00 "
                                     "(needed when another ESP32 with the same USB id is plugged in, such as the transmitter)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--leandvb", help="path of leandvb (default: build/leansdr/src/apps/leandvb, then PATH)")
    ap.add_argument("--ldpc", choices=("auto", "tool", "bitflip"), default="auto",
                    help="LDPC decoder: ldpc_tool (soft decision, about 5 dB better, 32-frame latency) or leandvb's "
                         "bit-flip one; auto takes ldpc_tool when it is installed")
    ap.add_argument("--sampler", choices=("auto", "linear", "rrc"), default="auto",
                    help="leandvb symbol sampler: auto = rrc below 100 kS/s (many samples per symbol: the default linear sampler "
                         "keeps all the out-of-band noise), else leandvb's default")
    ap.add_argument("--decimate", action="store_true",
                    help="experimental: below 100 kS/s (without --wide) lowpass and decimate the 333 ksps stream by the largest D "
                         "that leaves at least 2 samples per symbol, before leandvb (no clear gain in simulation)")
    ap.add_argument("--rrc-rej", type=float, default=50, help="leandvb RRC filter rejection with --sampler rrc (default 50)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show leandvb's messages")
    args = ap.parse_args()

    leandvb = find_tool("leandvb", args.leandvb, os.path.join(HERE, "build", "leansdr", "src", "apps", "leandvb"))
    ldpc_tool = None if args.ldpc == "bitflip" else find_tool("ldpc_tool", os.path.join(HERE, "build", "LDPC", "ldpc_tool"))
    if args.ldpc == "tool" and not ldpc_tool:
        sys.exit("ldpc_tool not found: run tools/install.sh first")
    modcod = MODCODS.get((args.const.upper(), args.cr))
    if modcod is None:
        sys.exit(f"unknown DVB-S2 modulation {args.const} {args.cr}")
    if not leandvb:
        sys.exit("leandvb not found: run tools/install.sh first, or give --leandvb")
    if not args.no_play and not (shutil.which("ffmpeg") and shutil.which("ffplay")):
        sys.exit("ffmpeg/ffplay not found: sudo apt install ffmpeg")

    espdr = None
    if args.wide:
        if server_up(args.host, args.port):
            sys.exit("an rtl_tcp server (espdr-rx) is running and holds the board: stop it for --wide")
        src = WideSource(args)
    else:
        espdr = None if server_up(args.host, args.port) else start_espdr(args)
        src = NarrowSource(args)
    procs = []
    try:
        freq = args.freq * 1e6
        if not args.no_search:
            # A signal far off centre is cut by the +-166 kHz window: retune to the estimate and look again.
            for _ in range(3):
                found = search(src, freq)
                if found is None:
                    break
                step, freq = found - freq, found
                if abs(step) < 5e3:
                    break

        fs_in = FS_WIDE if args.wide else FS
        if not args.wide and args.decimate and args.sr < 100e3 and FS // (2 * args.sr) >= 2:
            d = int(FS // (2 * args.sr))
            fs_in = FS / d
            src.decimate(args.sr, d)
            log(f"decimating {FS / 1e3:.0f} ksps by {d} to {FS / d / 1e3:.1f} ksps ({FS / d / args.sr:.2f} samples per symbol)")
        sampler = args.sampler if args.sampler != "auto" else ("rrc" if args.sr < 100e3 else "linear")
        # --fd-info on a pipe of its own: MER, CNR, FREQ (residual carrier offset, Hz) for the status line
        info_r, info_w = os.pipe()
        dvb = subprocess.Popen([leandvb] + src.leandvb_input() + ["--sr", str(int(args.sr)), "--standard", "DVB-S2",
                                "--const", args.const, "--cr", args.cr, "--fastdrift", "-d",
                                # only our MODCOD and normal frames: a corrupted PLS header would otherwise start an
                                # ldpc_tool for a random MODCOD, which then waits for its batch of 32 frames forever
                                "--modcods", str(1 << modcod),
                                "--framesizes", {"normal": "1", "short": "2", "auto": "3"}[args.frame]]
                               + (["--ldpc-helper", ldpc_tool] if ldpc_tool else ["--ldpc-bf", "100"])
                               + (["--sampler", "rrc", "--rrc-rej", str(args.rrc_rej)] if sampler == "rrc"
                                  else ["--sampler", "linear"] if args.sampler == "linear" else [])
                               # below 100 kS/s: no ~5 normal frames of data (4 s at 40 kS/s) thrown away before each new
                               # search; leandvb's first lock on normal frames is often a false one (MER < 0) and this gets
                               # out of it sooner (ESP32-ATV rx/sim_lock.py: first TS 31-40 s instead of 34-44 s or none)
                               + (["--fastlock"] if args.sr < 100e3 else [])
                               + ["--sr-tol", str(args.sr_tol)]
                               + ["--fd-info", str(info_w)] + (["--cnr"] if fs_in > 3 * args.sr else []),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(info_w,))
        os.close(info_w)
        procs.append(dvb)
        sinks = []
        if not args.no_play:
            ffmpeg = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "fatal", "-probesize", "64000",
                                       "-analyzeduration", "3000000", "-fflags", "+igndts+genpts", "-i", "-",
                                       "-vf", "setpts=N/25/TB", "-an", "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
                                       "-f", "nut", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # SDL's GLX path fails on some X11 setups; EGL works there and elsewhere.
            env = dict(os.environ, SDL_VIDEO_X11_FORCE_EGL=os.environ.get("SDL_VIDEO_X11_FORCE_EGL", "1"))
            ffplay = subprocess.Popen(["ffplay", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer",
                                       "-window_title", f"DVB-S2 {freq / 1e6:.4f} MHz (ESP32-S3 RX)", "-i", "-"],
                                      stdin=ffmpeg.stdout, env=env)
            ffmpeg.stdout.close()
            procs += [ffmpeg, ffplay]
            sinks.append(ffmpeg.stdin)
        if args.record:
            sinks.append(open(args.record, "wb"))
        if args.stdout:
            sinks.append(sys.stdout.buffer)

        stats = {"good": 0, "bad": 0, "locks": 0, "ts": 0}
        info = {}

        def pump_info():
            with os.fdopen(info_r, "r", errors="ignore") as f:
                for line in f:
                    k, _, v = line.strip().partition(" ")
                    if k in ("MER", "CNR", "FREQ", "LOCK", "FRAMELOCK"):
                        info[k] = v
                    if k == "FREQ" and not args.no_track:
                        try:
                            src.nco.track(float(v), float(info.get("MER", "nan")), info.get("FRAMELOCK") == "1")
                        except ValueError:
                            pass

        def pump_ts():
            while d := dvb.stdout.read1(65536):
                stats["ts"] += len(d) // 188
                for s in list(sinks):
                    try:
                        s.write(d)
                        s.flush()
                    except (BrokenPipeError, OSError):
                        sinks.remove(s)

        def pump_log():
            # leandvb -d prints one character per frame ('_' good, '!' with errors) without a newline.
            while d := dvb.stderr.read1(4096):
                t = d.decode(errors="ignore")
                stats["good"] += t.count("_")
                stats["bad"] += t.count("!")
                stats["locks"] += t.count("LOCKED")
                if args.verbose:
                    sys.stderr.write(t)

        src.start(freq, dvb.stdin)
        for fn in (pump_ts, pump_log, pump_info):
            threading.Thread(target=fn, daemon=True).start()

        log(f"receiving {freq / 1e6:.4f} MHz, {args.sr / 1e3:g} kS/s {args.const} {args.cr}, gain {args.gain:g} dB, "
            f"LDPC {'ldpc_tool' if ldpc_tool else 'bit-flip'}. Ctrl-C ends.")
        if ldpc_tool:
            log(f"ldpc_tool decodes 32 frames at a time: the first picture comes after about "
                f"{32 * plframe_symbols(args.const, args.frame != 'short') / args.sr:.0f} s")
        last = None
        while dvb.poll() is None and (args.no_play or procs[-1].poll() is None):
            time.sleep(5)
            now = (stats["good"], stats["bad"], stats["locks"], stats["ts"])
            q = "".join(f", {k} {info[k]}" + (" dB" if k in ("MER", "CNR") else " Hz" if k == "FREQ" else "")
                        for k in ("MER", "CNR", "FREQ") if k in info)
            if src.nco.updates:
                q += f", NCO {src.nco.freq:+.0f} Hz ({src.nco.rate:+.0f} Hz/s)"
            if now != last or q:
                log(f"frames ok {stats['good']}, with errors {stats['bad']}, locks {stats['locks']}, "
                    f"TS packets {stats['ts']}{q}")
                last = now
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(3)
            except subprocess.TimeoutExpired:
                p.kill()
        src.stop()
        for p in procs:
            for f in (p.stdin, p.stdout):
                try:
                    if f:
                        f.close()
                except OSError:  # its reader or writer is gone already
                    pass
        if espdr:
            espdr.terminate()
            espdr.wait(10)


if __name__ == "__main__":
    main()

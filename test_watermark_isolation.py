"""Decode in a fresh interpreter with no app state, run files, or model imports."""

import json
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

from token_auction_lab import watermark as wm

ROOT = Path(__file__).resolve().parent
DECODER_SOURCES = (
    "auction.py",
    "token_auction_lab/__init__.py",
    "token_auction_lab/watermark.py",
    "experiments/__init__.py",
    "experiments/ad_stego/__init__.py",
    "experiments/ad_stego/prototype.py",
    "experiments/ad_stego/race.py",
    "experiments/ad_stego/tilt.py",
    "experiments/ad_stego/tilt_race.py",
    "experiments/ad_stego/word_window.py",
    "experiments/ad_stego/tokenizer_check.py",
    "experiments/ad_stego/synthetic.py",
)

CHILD = r"""
import sys, json, os
from pathlib import Path
stage, purelib, platlib = map(Path, sys.argv[1:])
# -I -S excludes cwd, user-site, PYTHONPATH, and editable-install .pth hooks.
sys.path[:0] = [str(stage/'code'), str(purelib), str(platlib)]
allowed = [stage.resolve(), purelib.resolve(), platlib.resolve(),
           (Path(sys.base_prefix)/'lib').resolve()]
opened, denied, blocked_network = set(), [], []

def audit(event, args):
    if event.startswith('socket.'):
        # urllib3 probes IPv6 during import; it catches the failure and decoding
        # continues. Socket creation remains blocked, so no channel is opened.
        blocked_network.append(event)
        raise RuntimeError('Network denied: ' + event)
    if event in ('subprocess.Popen', 'os.system', 'os.exec'):
        denied.append(event)
        raise RuntimeError('Subprocess denied: ' + event)
    if event in ('open', 'os.listdir', 'os.scandir') and args and not isinstance(args[0], int):
        path = Path(os.fsdecode(args[0] if args[0] is not None else '.')).resolve()
        if not any(path.is_relative_to(root) for root in allowed):
            denied.append(str(path))
            raise RuntimeError('Non-decoder file access denied: ' + str(path))
        if event == 'open':
            opened.add(str(path))

sys.addaudithook(audit)
from token_auction_lab.watermark import decode_response
text = sys.stdin.read()  # The only response-specific input.
result = decode_response(text)
for name in ('torch', 'transformers', 'token_auction_lab.live',
             'token_auction_lab.runner', 'token_auction_lab.decoder'):
    assert name not in sys.modules, name
assert not denied, denied
print(json.dumps({'result': result, 'denied_accesses': denied,
                  'opened_files': sorted(opened), 'blocked_network_attempts': blocked_network,
                  'model_or_live_modules_loaded': False}))
"""


def isolated_decode(text):
    """Only text crosses stdin. Other inputs are code, libraries, and pinned tokenizer."""
    with tempfile.TemporaryDirectory(prefix="watermark-text-only-") as tmp:
        stage = Path(tmp)
        for name in DECODER_SOURCES:
            target = stage / "code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        cache = stage / "cache"
        tokenizer = (
            cache
            / "models--Qwen--Qwen3-4B-Instruct-2507"
            / "snapshots"
            / wm.REVISION
            / "tokenizer.json"
        )
        tokenizer.parent.mkdir(parents=True)
        shutil.copyfile(wm.tokenizer_path(), tokenizer)
        paths = sysconfig.get_paths()
        process = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-S",
                "-c",
                CHILD,
                str(stage),
                paths["purelib"],
                paths["platlib"],
            ],
            input=text,
            text=True,
            capture_output=True,
            timeout=60,
            cwd=stage,
            env={
                "HF_HUB_CACHE": str(cache),
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TMPDIR": str(stage),
            },
        )
        if process.returncode:
            raise AssertionError(process.stderr)
        return json.loads(process.stdout)


class DecoderIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            wm.get_tokenizer()
        except (ImportError, ValueError):
            raise unittest.SkipTest("Requires the locally cached tokenizer; never downloads it.")

    def test_fresh_text_only_process_matches_success_false_positive_and_short_result(self):
        for example in wm.examples():
            with self.subTest(example=example["id"]):
                expected = wm.decode_response(example["text"])
                result = isolated_decode(example["text"])
                self.assertEqual(result["result"], expected)
                self.assertFalse(result["denied_accesses"])
                self.assertFalse(result["model_or_live_modules_loaded"])
                self.assertFalse(
                    any("/artifacts/" in p or "/evidence/" in p for p in result["opened_files"])
                )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from download_manifest_checkpoints import file_sha256, verify_checkpoint


class ManifestDownloadTests(unittest.TestCase):
    def test_verifies_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "best_model.pt"
            checkpoint.write_bytes(b"checkpoint")
            verify_checkpoint(checkpoint, file_sha256(checkpoint))

    def test_rejects_checkpoint_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "best_model.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_checkpoint(checkpoint, "0" * 64)


if __name__ == "__main__":
    unittest.main()

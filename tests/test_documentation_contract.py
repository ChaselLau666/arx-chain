from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "act"))

from pipeline_contract import ACTION_DIM, ACTION_SEMANTICS, FPS, HTTP_PROTOCOL_VERSION, SCHEMA_VERSION


class DocumentationContractTests(unittest.TestCase):
    def test_manual_contains_runtime_contract(self):
        manual = (REPO_ROOT / "docs/ARX_LIFT2S_CUSTOM_PIPELINE.md").read_text(encoding="utf-8")
        for value in (
            SCHEMA_VERSION,
            HTTP_PROTOCOL_VERSION,
            ACTION_SEMANTICS,
            f"{FPS} FPS",
            f"{ACTION_DIM} 维",
            "`R`：开始",
            "`E`：结束",
            "`S`：保存",
            "`D`：丢弃",
            "GET  /healthz",
            "GET  /v1/schema",
            "POST /v1/reset",
            "POST /v1/infer",
        ):
            self.assertIn(value, manual)

    def test_customizations_links_both_manual_forms(self):
        index = (REPO_ROOT / "CUSTOMIZATIONS.md").read_text(encoding="utf-8")
        self.assertIn("docs/ARX_LIFT2S_CUSTOM_PIPELINE.md", index)
        self.assertIn("docs/ARX_LIFT2S_CUSTOM_PIPELINE.pdf", index)


if __name__ == "__main__":
    unittest.main()

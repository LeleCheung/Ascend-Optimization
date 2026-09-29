"""Legacy source/collection tests stub review; test_catalog_extract_review tests it."""
import pytest


@pytest.fixture(autouse=True)
def stub_extraction_review(monkeypatch):
    from kernelgen.workflows import catalog_extract
    def accepted(operator, catalog, evidence, workspace, factory):
        catalog.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        path=workspace/'review.json'
        path.write_text('{"accepted": true}')
        return True,path,''
    monkeypatch.setattr(catalog_extract,'review_attempt',accepted)

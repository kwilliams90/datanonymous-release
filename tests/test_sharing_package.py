from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from datanonymous.errors import UsageError
from datanonymous.model import FileResult, Message
from datanonymous.privacy.package import finish, inventory
from datanonymous.privacy.sharing_policy import blocking_findings, validate_settings


def _settings(**kw):
    return SimpleNamespace(mode="sharing", min_cell=10, max_row_matches=None, **kw)


@pytest.mark.parametrize("flag", ["force_factor", "keep_name", "keep_titles", "force", "no_strict", "no_default_ids"])
def test_sharing_refuses_bypasses(flag):
    with pytest.raises(UsageError):
        validate_settings(_settings(**{flag: True}))


def test_fidelity_is_explicit_bypass_scope():
    validate_settings(SimpleNamespace(mode="fidelity", force_factor=True, min_cell=1))


def test_inventory_requires_exact_files(tmp_path):
    (tmp_path / "data.csv").write_text("x\n1\n")
    assert not inventory(tmp_path, {"data.csv"})[0]
    assert inventory(tmp_path, {"data.csv", "absent.csv"})[0]
    (tmp_path / "notes.txt").write_text("stale")
    assert inventory(tmp_path, {"data.csv"})[0]


def test_unreadable_workbook_and_external_links_block(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"bad archive")
    assert inventory(tmp_path, {path.name})[0]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("_rels/.rels", '<Relationships><Relationship TargetMode="External" Target="https://example.invalid"/></Relationships>')
    assert inventory(tmp_path, {path.name})[0]


def _run(stage, messages=()):
    file = FileResult("data.csv", "data.csv", "csv", written=True)
    return SimpleNamespace(files=[file], docs={}, messages=list(messages), any_failed=False, skipped=[], out_dir=stage)


def test_unknown_warning_blocks_and_withholds_output(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "data.csv").write_text("x\n1\n")
    run = _run(stage, [Message("warning", "new_warning", "unreviewed")])
    assert blocking_findings(run)
    result = finish(run, tmp_path / "share", stage)
    assert not result["released"]
    assert not (tmp_path / "share").exists()
    assert stage.exists()
    assert not run.files[0].written


def test_only_complete_verified_package_is_promoted(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "data.csv").write_text("x\n1\n")
    run = _run(stage, [Message("warning", "fidelity_drift", "approved utility loss")])
    result = finish(run, tmp_path / "share", stage)
    assert result["released"]
    assert len(result["inventory_sha256"]["data.csv"]) == 64
    assert not stage.exists()
    assert (tmp_path / "share" / "data.csv").exists()


def test_all_or_nothing_withholds_package_with_skipped_input(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "data.csv").write_text("x\n1\n")
    run = _run(stage)
    run.skipped = [SimpleNamespace(rel_path="unsupported")]
    assert not finish(run, tmp_path / "share", stage, all_or_nothing=True)["released"]


def _partial_run(tmp_path, rules):
    from datanonymous.config import Settings
    from datanonymous.pipeline import execute
    from release_corpus import write_corpus

    original = tmp_path / "original"
    write_corpus(original)
    (original / "broken.xlsx").write_bytes(b"invented invalid workbook")  # fails before its columns are read
    return execute(Settings(mode="sharing", seed=573291, rules=rules), original, tmp_path / "out", tmp_path / "rep")


def test_an_unused_rule_scoped_to_a_written_table_still_withholds_everything(tmp_path):
    run = _partial_run(tmp_path, [{"rule": "Nope_A <= Nope_B", "file": "roster.csv"}])
    assert not run.package["released"]
    assert "rule_unused" in run.package["blocking_codes"]


@pytest.mark.parametrize("scope", [None, "broken.xlsx"])
def test_an_unused_rule_that_may_be_meant_for_an_unread_table_does_not_block(tmp_path, scope):
    rule = {"rule": "Nope_A <= Nope_B", **({"file": scope} if scope else {})}
    run = _partial_run(tmp_path, [rule])
    assert run.package["released"], run.package["blocking_codes"]

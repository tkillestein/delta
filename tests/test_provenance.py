from delta._provenance import _is_delta_checkout


def test_is_delta_checkout_true_for_own_repo():
    from pathlib import Path

    import delta

    repo_root = Path(delta.__file__).resolve().parents[2]
    assert _is_delta_checkout(repo_root)


def test_is_delta_checkout_false_for_unrelated_repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "some-other-project"\n')
    assert not _is_delta_checkout(tmp_path)


def test_is_delta_checkout_false_without_pyproject(tmp_path):
    assert not _is_delta_checkout(tmp_path)

from __future__ import annotations

import json
import struct

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def _fixture():
    values = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.bfloat16)
    model = Model({"text": {"config": {}, "resources": {"fixture": "resource"}}})
    model.resources["resource"] = b"identity-test-resource"
    model.add(Parameter("weight", (2, 2), array_source(values, "fixture")))
    return model, Recipe(model)


def _readback(path, expected_identity):
    with Artifact(path) as artifact:
        assert artifact.directory.identity == expected_identity
        assert artifact.read_object("resource") == b"identity-test-resource"
        object_id = artifact.directory.bindings["weight"]["object"]
        assert artifact.read_object(object_id) == struct.pack("<4H", 0x3F80, 0xC000, 0x3F00, 0x4080)
        report = json.loads(path.with_suffix(path.suffix + ".conversion.json").read_text(encoding="utf-8"))
        assert report["identity"] == expected_identity
        assert report["artifact_id"] == artifact.artifact_id.hex()


def test_explicit_identity_roundtrips_without_changing_payload(tmp_path):
    model, recipe = _fixture()
    path = tmp_path / "unrelated-filename.ninfer"
    identity = {"model_id": "组织/model-v1", "weights_id": "canonical-weights/v2"}
    report = convert(model, recipe, path, name="display name", device="cpu", **identity)
    assert report["identity"] == identity
    _readback(path, identity)
    with Artifact(path) as artifact:
        assert artifact.directory.metadata == {"name": "display name"}


@pytest.mark.parametrize("name, expected_model", [(None, "same"), ("named-model", "named-model")])
def test_default_weights_identity_is_unique_per_conversion(tmp_path, name, expected_model):
    identities = []
    for directory in (tmp_path / "a", tmp_path / "b"):
        directory.mkdir()
        model, recipe = _fixture()
        path = directory / "same.ninfer"
        report = convert(model, recipe, path, name=name, device="cpu")
        identity = report["identity"]
        assert identity["model_id"] == expected_model
        assert identity["weights_id"].startswith("conversion-")
        assert len(identity["weights_id"]) == len("conversion-") + 32
        _readback(path, identity)
        identities.append(identity)
    assert identities[0]["weights_id"] != identities[1]["weights_id"]


@pytest.mark.parametrize("field, value", [
    ("model_id", ""), ("weights_id", ""),
    ("model_id", "bad\0model"), ("weights_id", "bad\0weights"),
    ("model_id", 123), ("weights_id", 123),
])
def test_invalid_explicit_identity_fails_before_preparation(tmp_path, monkeypatch, field, value):
    model, recipe = _fixture()

    def unexpected_prepare(**kwargs):
        raise AssertionError("invalid identity must fail before preparing weights")

    monkeypatch.setattr(recipe, "prepare", unexpected_prepare)
    path = tmp_path / "invalid.ninfer"
    with pytest.raises(ArtifactError, match=f"artifact identity {field}"):
        convert(model, recipe, path, device="cpu", **{field: value})
    assert not list(tmp_path.iterdir())

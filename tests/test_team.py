import base64
import json

from src import team

PNG_1PX = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def test_team_file_lists_six_members():
    members, repo, error = team.load_team()
    assert error is None and len(members) == 6
    assert all(m.name and m.role and m.description for m in members)
    assert not any("@" in (m.description + m.linkedin + repo) for m in members)  # no e-mail addresses


def test_initials_ignore_placeholders():
    assert team.initials("Joao [NOM]") == "J"
    assert team.initials("Ruben Silva") == "RS"
    assert team.initials("") == "?"


def test_photo_fallback_and_data_uri(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "a.png").write_bytes(PNG_1PX)
    with_photo = team.Member("A B", photo="assets/a.png")
    assert team.photo_data_uri(with_photo, tmp_path).startswith("data:image/png;base64,")
    assert team.photo_data_uri(team.Member("A", photo="assets/missing.png"), tmp_path) is None
    assert team.photo_data_uri(team.Member("A", photo="assets/a.gif"), tmp_path) is None
    assert team.photo_data_uri(team.Member("A", photo="../outside.png"), tmp_path) is None
    assert team.photo_data_uri(team.Member("A"), tmp_path) is None


def test_bad_team_file_gives_message(tmp_path):
    bad = tmp_path / "team.json"
    bad.write_text("{not json", encoding="utf-8")
    members, _, error = team.load_team(bad)
    assert members == [] and "unavailable" in error
    members, _, error = team.load_team(tmp_path / "absent.json")
    assert members == [] and error
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"members": [{"name": "X Y", "unknown": 1}, {"role": "no name"}]}), encoding="utf-8")
    members, repo, error = team.load_team(ok)
    assert error is None and [m.name for m in members] == ["X Y"] and repo == ""

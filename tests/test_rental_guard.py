import io
import json
import sys
import traceback
from urllib.error import URLError

import pytest

from training import rental_guard


KEY = "test-only-key-must-not-be-logged"


@pytest.fixture(autouse=True)
def expired_deadline(monkeypatch):
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("CONTAINER_API_KEY", KEY)
    monkeypatch.setattr(sys, "argv", ["rental_guard", "--deadline", "2000-01-01T00:00:00+00:00"])


def test_http_success_with_provider_rejection_is_a_failure(monkeypatch, capsys):
    body = json.dumps({"success": False, "msg": f"Rejected credential {KEY}"}).encode()
    monkeypatch.setattr(rental_guard, "urlopen", lambda *args, **kwargs: io.BytesIO(body))

    with pytest.raises(RuntimeError) as caught:
        rental_guard.main()

    output = capsys.readouterr()
    assert KEY not in "".join(traceback.format_exception(caught.value)) + output.out + output.err


def test_transport_failure_does_not_expose_credentials_in_traceback(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise URLError(f"Failed request with Authorization: Bearer {KEY}")

    monkeypatch.setattr(rental_guard, "urlopen", fail)
    with pytest.raises(RuntimeError) as caught:
        rental_guard.main()

    output = capsys.readouterr()
    assert KEY not in "".join(traceback.format_exception(caught.value)) + output.out + output.err

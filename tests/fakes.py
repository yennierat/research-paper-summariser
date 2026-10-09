import requests


class FakeResponse:
    def __init__(self, ok=True, status_code=200, content=b"", data=None, text="", headers=None):
        self.ok, self.status_code, self.content, self._data, self.text = ok, status_code, content, data, text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._data


class FakeConn:
    """Records SQL; answers selects with canned rows. fetchone() returns None (no active model)."""

    def __init__(self, rows=None):
        self.rows, self.sql, self.rowcount = rows or [], [], 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def execute(self, sql, params=None):
        self.sql.append((sql, params))
        return self

    def cursor(self):
        return self

    def executemany(self, sql, params):
        self.sql.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return None

    def commit(self):
        pass

    def statements(self, keyword):
        return [(s, p) for s, p in self.sql if keyword in s]

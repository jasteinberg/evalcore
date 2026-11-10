"""Shared fakes for the test suite: scripted backends, mocked HTTP, toy
tokenizers and models, and frame builders.  One copy of each, so a fix
reaches every test that uses it.  Modules that need torch or httpx import
them at the top.  A test file imports such a module only after its own
`pytest.importorskip`."""

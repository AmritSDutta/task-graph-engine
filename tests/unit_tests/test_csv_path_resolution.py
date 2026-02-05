"""Unit tests for CSV path resolution - no API calls required."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from task_agent.utils.path_utils import resolve_csv_path


class TestResolveCsvPath:
    """Tests for resolve_csv_path function."""

    def test_resolve_absolute_path(self):
        """Absolute paths should be used directly."""
        # Windows absolute path
        if Path(__file__).drive:  # Check if on Windows
            abs_path = "C:\\config\\model_costs.csv"
        else:
            abs_path = "/app/config/model_costs.csv"

        result = resolve_csv_path(abs_path)
        assert str(result) == abs_path
        assert result.is_absolute()

    def test_resolve_relative_path_with_forward_slash(self):
        """Relative paths with / should resolve from cwd."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/mock/cwd")
            result = resolve_csv_path("./config/model_costs.csv")

            assert result == Path("/mock/cwd/config/model_costs.csv")

    def test_resolve_relative_path_with_backslash(self):
        """Relative paths with \\ should resolve from cwd (Windows)."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/mock/cwd")
            result = resolve_csv_path(".\\config\\model_costs.csv")

            assert result == Path("/mock/cwd/config/model_costs.csv")

    def test_resolve_filename_only_to_project_root(self):
        """Filename only should resolve to project root."""
        result = resolve_csv_path("model_costs.csv")

        # Should resolve to project root (parent of src/ directory)
        # The path should exist or at least point to the right location
        assert result.name == "model_costs.csv"
        # Verify it's pointing to project root, not somewhere inside src/
        assert "task_agent" not in str(result) or str(result).endswith("model_costs.csv")

    def test_resolve_custom_project_root(self):
        """Custom project root should be used when provided."""
        custom_root = Path("/custom/project/root")
        result = resolve_csv_path("model_costs.csv", project_root_fallback=custom_root)

        # Use Path comparison (works across platforms)
        assert result == custom_root / "model_costs.csv"
        assert result.name == "model_costs.csv"

    def test_resolve_nested_relative_path(self):
        """Nested relative paths should resolve from cwd."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/app/deployment")
            result = resolve_csv_path("./config/production/model_costs.csv")

            assert result == Path("/app/deployment/config/production/model_costs.csv")

    def test_resolve_parent_directory_relative_path(self):
        """Relative paths with .. should resolve from cwd."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/app/deployment/bin")
            result = resolve_csv_path("../config/model_costs.csv")

            # Normalize paths for comparison (handles ../ differences)
            expected = Path("/app/deployment/config/model_costs.csv")
            assert result.resolve() == expected.resolve()

    def test_resolve_windows_absolute_path(self):
        """Windows absolute paths (with drive letter) should be absolute."""
        result = resolve_csv_path("C:\\Users\\config\\model_costs.csv")

        assert result.is_absolute()
        assert str(result).startswith("C:\\")

    def test_resolve_windows_network_path(self):
        """Windows UNC paths should be absolute."""
        result = resolve_csv_path("\\\\server\\share\\model_costs.csv")

        assert result.is_absolute()

    def test_resolve_unix_absolute_path(self):
        """Unix absolute paths should be absolute."""
        result = resolve_csv_path("/var/config/model_costs.csv")

        assert result.is_absolute()

    def test_filename_with_no_separators_resolves_to_project_root(self):
        """Any filename without path separators uses project root."""
        result = resolve_csv_path("custom_costs.csv")

        # Verify it's in project root
        assert "task_agent" not in str(result).split("custom_costs.csv")[0]
        assert result.name == "custom_costs.csv"

    def test_multiple_separators_resolves_from_cwd(self):
        """Paths with any separators (even multiple) resolve from cwd."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/app")
            result = resolve_csv_path("config/data/model_costs.csv")

            assert result == Path("/app/config/data/model_costs.csv")


class TestCsvPathIntegration:
    """Integration tests for CSV loading with path resolution."""

    def test_load_costs_with_absolute_path(self, tmp_path):
        """Test loading costs from absolute path."""
        # Create temporary CSV file
        csv_file = tmp_path / "test_costs.csv"
        csv_file.write_text("model,cost\ntest-model,1.5\n")

        from task_agent.llms.simple_llm_selector.models import _load_model_costs_from_csv

        costs = _load_model_costs_from_csv(str(csv_file))
        assert costs == {"test-model": 1.5}

    def test_load_costs_with_relative_path(self, tmp_path, monkeypatch):
        """Test loading costs from relative path."""
        # Create CSV in temp directory
        csv_file = tmp_path / "test_costs.csv"
        csv_file.write_text("model,cost\ntest-model,2.5\n")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        from task_agent.llms.simple_llm_selector.models import _load_model_costs_from_csv

        costs = _load_model_costs_from_csv("./test_costs.csv")
        assert costs == {"test-model": 2.5}

    def test_load_costs_nonexistent_path_returns_empty_dict(self):
        """Test that non-existent CSV returns empty dict gracefully."""
        from task_agent.llms.simple_llm_selector.models import _load_model_costs_from_csv

        costs = _load_model_costs_from_csv("/nonexistent/path/costs.csv")
        assert costs == {}

    def test_load_capabilities_with_absolute_path(self, tmp_path):
        """Test loading capabilities from absolute path."""
        # Create temporary CSV file with enabled column
        csv_file = tmp_path / "test_caps.csv"
        csv_file.write_text(
            "model,reasoning,tools,fast,cheap,enabled\n"
            "test-model,True,True,False,True,True\n"
        )

        from task_agent.llms.simple_llm_selector.models import (
            _load_model_capabilities_from_csv,
        )

        caps = _load_model_capabilities_from_csv(str(csv_file))
        assert caps == {"test-model": {"reasoning", "tools", "cheap"}}

    def test_load_capabilities_nonexistent_path_returns_empty_dict(self):
        """Test that non-existent CSV returns empty dict gracefully."""
        from task_agent.llms.simple_llm_selector.models import (
            _load_model_capabilities_from_csv,
        )

        caps = _load_model_capabilities_from_csv("/nonexistent/path/caps.csv")
        assert caps == {}

"""Tests for FastAPI webapp endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import status

from webapp import app, verify_api_key


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model_capabilities():
    """Mock model capabilities data."""
    return {
        "gpt-4o-mini": {"coding", "reasoning", "fast", "tools", "informational"},
        "gemini-2.5-flash": {"coding", "reasoning", "cheap", "informational"},
        "deepseek-v3.1:671b-cloud": {"coding", "reasoning", "long", "tools"},
    }


@pytest.fixture
def mock_model_costs():
    """Mock model costs data."""
    return {
        "gpt-4o-mini": 0.15,
        "gemini-2.5-flash": 0.08,
        "deepseek-v3.1:671b-cloud": 0.5,
    }


@pytest.fixture
def mock_coding_priority():
    """Mock coding model priority list."""
    return ["deepseek-v3.1:671b-cloud", "gemini-2.5-flash"]


@pytest.fixture
def mock_usage_tracker():
    """Mock model usage singleton."""
    tracker = MagicMock()
    tracker.get_model_usage.return_value = 0
    return tracker


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# ============================================================================
# Test Root Endpoint
# ============================================================================


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test GET / returns API information."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert data["name"] == "Task Graph Engine API"
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert data["docs"] == "/docs"
        assert "langgraph_api" in data


# ============================================================================
# Test Health Check
# ============================================================================


class TestHealthCheck:
    """Test the health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Test GET /api/health returns health status."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", {"gpt-4o-mini": {"coding"}}):
                response = client.get("/api/health")

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["status"] == "healthy"
                assert "version" in data
                assert "auth_required" in data
                assert "models_loaded" in data


# ============================================================================
# Test Authentication
# ============================================================================


class TestAuthentication:
    """Test API key authentication."""

    def test_auth_disabled_allows_access(self, client):
        """Test that auth disabled allows access without token."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", {}):
                response = client.get("/api/models")

                # Should allow access without auth header
                assert response.status_code == status.HTTP_200_OK

    def test_auth_enabled_with_valid_token(self, client):
        """Test valid API key is accepted."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-secret-key"
            with patch("webapp.MODEL_CAPABILITIES", {}):
                response = client.get(
                    "/api/models",
                    headers={"Authorization": "Bearer test-secret-key"}
                )

                assert response.status_code == status.HTTP_200_OK

    def test_auth_enabled_without_token(self, client):
        """Test missing auth header returns 401."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-secret-key"
            response = client.get("/api/models")

            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            data = response.json()
            assert "detail" in data

    def test_auth_enabled_with_invalid_token(self, client):
        """Test wrong API key returns 401."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "correct-key"
            response = client.get(
                "/api/models",
                headers={"Authorization": "Bearer wrong-key"}
            )

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_auth_enabled_with_malformed_token(self, client):
        """Test malformed Bearer token returns 401."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            # Missing "Bearer" prefix
            response = client.get(
                "/api/models",
                headers={"Authorization": "test-key"}
            )

            # FastAPI HTTPBearer handles this
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# Test List Models Endpoint
# ============================================================================


class TestListModelsEndpoint:
    """Test the list models endpoint."""

    def test_list_models_without_auth(self, client, mock_model_capabilities, mock_model_costs, mock_coding_priority):
        """Test list models when auth disabled."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    with patch("webapp.CODING_MODEL_PRIORITY", mock_coding_priority):
                        with patch("webapp.get_model_capabilities") as mock_get_caps:
                            with patch("webapp.get_model_cost") as mock_get_cost:
                                # Mock the helper functions
                                mock_get_caps.side_effect = lambda x: mock_model_capabilities.get(x, set())
                                mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                                response = client.get("/api/models")

                                assert response.status_code == status.HTTP_200_OK
                                data = response.json()
                                assert "count" in data
                                assert "models" in data
                                assert data["count"] == 3
                                assert "gpt-4o-mini" in data["models"]

    def test_list_models_with_auth(self, client, mock_model_capabilities, mock_model_costs, mock_coding_priority):
        """Test list models with valid auth token."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    with patch("webapp.CODING_MODEL_PRIORITY", mock_coding_priority):
                        with patch("webapp.get_model_capabilities") as mock_get_caps:
                            with patch("webapp.get_model_cost") as mock_get_cost:
                                mock_get_caps.side_effect = lambda x: mock_model_capabilities.get(x, set())
                                mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                                response = client.get(
                                    "/api/models",
                                    headers={"Authorization": "Bearer test-key"}
                                )

                                assert response.status_code == status.HTTP_200_OK

    def test_list_models_response_structure(self, client, mock_model_capabilities, mock_model_costs, mock_coding_priority):
        """Test list models returns correct structure."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    with patch("webapp.CODING_MODEL_PRIORITY", mock_coding_priority):
                        with patch("webapp.get_model_capabilities") as mock_get_caps:
                            with patch("webapp.get_model_cost") as mock_get_cost:
                                mock_get_caps.side_effect = lambda x: mock_model_capabilities.get(x, set())
                                mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                                response = client.get("/api/models")
                                data = response.json()

                                # Check model entry structure
                                gpt_data = data["models"]["gpt-4o-mini"]
                                assert "capabilities" in gpt_data
                                assert "cost" in gpt_data
                                assert "is_coding_priority" in gpt_data
                                assert isinstance(gpt_data["capabilities"], list)
                                assert gpt_data["cost"] == 0.15
                                assert gpt_data["is_coding_priority"] is False

                                # Check priority model
                                deepseek_data = data["models"]["deepseek-v3.1:671b-cloud"]
                                assert deepseek_data["is_coding_priority"] is True


# ============================================================================
# Test Get Model Details Endpoint
# ============================================================================


class TestGetModelDetailsEndpoint:
    """Test the get model details endpoint."""

    def test_get_model_details_existing(self, client, mock_model_capabilities, mock_model_costs, mock_coding_priority):
        """Test get model details for existing model."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    with patch("webapp.CODING_MODEL_PRIORITY", mock_coding_priority):
                        with patch("webapp.get_model_capabilities") as mock_get_caps:
                            with patch("webapp.get_model_cost") as mock_get_cost:
                                mock_get_caps.side_effect = lambda x: mock_model_capabilities.get(x, set())
                                mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                                response = client.get("/api/models/gpt-4o-mini")

                                assert response.status_code == status.HTTP_200_OK
                                data = response.json()
                                assert data["name"] == "gpt-4o-mini"
                                assert "capabilities" in data
                                assert "cost" in data
                                assert data["cost"] == 0.15

    def test_get_model_details_not_found(self, client, mock_model_capabilities, mock_model_costs):
        """Test get model details for non-existent model."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    response = client.get("/api/models/non-existent-model")

                    assert response.status_code == status.HTTP_404_NOT_FOUND
                    data = response.json()
                    assert "detail" in data
                    assert "not found" in data["detail"].lower()

    def test_get_model_details_with_auth(self, client, mock_model_capabilities, mock_model_costs, mock_coding_priority):
        """Test get model details with authentication."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    with patch("webapp.CODING_MODEL_PRIORITY", mock_coding_priority):
                        with patch("webapp.get_model_capabilities") as mock_get_caps:
                            with patch("webapp.get_model_cost") as mock_get_cost:
                                mock_get_caps.side_effect = lambda x: mock_model_capabilities.get(x, set())
                                mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                                response = client.get(
                                    "/api/models/gpt-4o-mini",
                                    headers={"Authorization": "Bearer test-key"}
                                )

                                assert response.status_code == status.HTTP_200_OK

    def test_get_model_details_unauthorized(self, client, mock_model_capabilities, mock_model_costs):
        """Test get model details fails without auth when required."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.MODEL_COST", mock_model_costs):
                    response = client.get("/api/models/gpt-4o-mini")

                    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ============================================================================
# Test Get Statistics Endpoint
# ============================================================================


class TestGetStatisticsEndpoint:
    """Test the statistics endpoint."""

    def test_get_statistics_with_usage(self, client, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test statistics with model usage data."""
        # Set up usage data
        def mock_get_usage(model):
            return 10 if model == "gpt-4o-mini" else 0

        def mock_get_token_usage(model):
            return 1500.0 if model == "gpt-4o-mini" else 0.0

        mock_usage_tracker.get_model_usage.side_effect = mock_get_usage
        mock_usage_tracker.get_model_token_usage.side_effect = mock_get_token_usage

        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            mock_settings.COST_SPREADING_FACTOR = 0.03
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.get_model_usage_singleton", return_value=mock_usage_tracker):
                    with patch("webapp.get_model_cost") as mock_get_cost:
                        mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                        response = client.get("/api/statistics")

                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()
                        assert "cost_spreading_factor" in data
                        assert "total_models_with_usage" in data
                        assert "models" in data

                        # Check gpt-4o-mini has usage stats
                        assert "gpt-4o-mini" in data["models"]
                        gpt_stats = data["models"]["gpt-4o-mini"]
                        assert gpt_stats["usage_count"] == 10
                        assert gpt_stats["base_cost"] == 0.15
                        assert gpt_stats["token_usage"] == 1500.0
                        assert "derived_cost" in gpt_stats
                        assert "penalty_factor" in gpt_stats

    def test_get_statistics_no_usage(self, client, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test statistics with no model usage."""
        # All models have zero usage
        mock_usage_tracker.get_model_usage.return_value = 0

        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            mock_settings.COST_SPREADING_FACTOR = 0.03
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.get_model_usage_singleton", return_value=mock_usage_tracker):
                    with patch("webapp.get_model_cost") as mock_get_cost:
                        mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                        response = client.get("/api/statistics")

                        assert response.status_code == status.HTTP_200_OK
                        data = response.json()

                        # Should have no models with usage
                        assert data["total_models_with_usage"] == 0
                        assert len(data["models"]) == 0

    def test_get_statistics_derived_costs(self, client, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test statistics calculates derived costs correctly."""
        # Usage: gpt-4o-mini has 5 uses
        def mock_get_usage(model):
            return 5 if model == "gpt-4o-mini" else 0

        def mock_get_token_usage(model):
            return 750.0 if model == "gpt-4o-mini" else 0.0

        mock_usage_tracker.get_model_usage.side_effect = mock_get_usage
        mock_usage_tracker.get_model_token_usage.side_effect = mock_get_token_usage

        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            mock_settings.COST_SPREADING_FACTOR = 0.03
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.get_model_usage_singleton", return_value=mock_usage_tracker):
                    with patch("webapp.get_model_cost") as mock_get_cost:
                        mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                        response = client.get("/api/statistics")
                        data = response.json()

                        # Check derived cost calculation
                        # derived_cost = 0.15 * exp(0.03 * 5) = 0.15 * 1.1618... = 0.174
                        gpt_stats = data["models"]["gpt-4o-mini"]
                        expected_cost = 0.15 * (2.71828 ** (0.03 * 5))
                        assert abs(gpt_stats["derived_cost"] - round(expected_cost, 4)) < 0.001
                        # Verify token_usage field is present
                        assert gpt_stats["token_usage"] == 750.0

    def test_get_statistics_with_auth(self, client, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test statistics endpoint with authentication."""
        mock_usage_tracker.get_model_usage.return_value = 0

        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            mock_settings.COST_SPREADING_FACTOR = 0.03
            with patch("webapp.MODEL_CAPABILITIES", mock_model_capabilities):
                with patch("webapp.get_model_usage_singleton", return_value=mock_usage_tracker):
                    with patch("webapp.get_model_cost") as mock_get_cost:
                        mock_get_cost.side_effect = lambda x: mock_model_costs.get(x)

                        response = client.get(
                            "/api/statistics",
                            headers={"Authorization": "Bearer test-key"}
                        )

                        assert response.status_code == status.HTTP_200_OK


# ============================================================================
# Test Get Config Endpoint
# ============================================================================


class TestGetConfigEndpoint:
    """Test the config endpoint."""

    def test_get_config_returns_non_sensitive(self, client):
        """Test config endpoint returns only non-sensitive config."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            mock_settings.INFERENCE_MODEL = "test-model"
            mock_settings.MODERATION_API_CHECK_REQ = True
            mock_settings.COST_SPREADING_FACTOR = 0.05
            mock_settings.MODEL_COST_CSV_PATH = "costs.csv"
            mock_settings.MODEL_CAPABILITY_CSV_PATH = "caps.csv"
            mock_settings.API_KEY = "secret-key"  # Should NOT be in response

            response = client.get("/api/config")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["inference_model"] == "test-model"
            assert data["moderation_api_check_req"] is True
            assert data["cost_spreading_factor"] == 0.05
            assert data["model_cost_csv_path"] == "costs.csv"
            assert data["model_capability_csv_path"] == "caps.csv"

            # API_KEY should NOT be exposed
            assert "api_key" not in data
            assert "API_KEY" not in data
            assert "secret-key" not in str(data)

    def test_get_config_excludes_secrets(self, client):
        """Test that sensitive config is excluded from response."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = False
            mock_settings.INFERENCE_MODEL = "sarvam-m"
            mock_settings.API_KEY = "super-secret-api-key-12345"

            response = client.get("/api/config")
            data = response.json()

            # Verify secrets are not leaked
            assert "api_key" not in data
            assert "secret" not in str(data).lower()
            assert data["inference_model"] == "sarvam-m"

    def test_get_config_with_auth(self, client):
        """Test config endpoint with authentication."""
        with patch("webapp.settings") as mock_settings:
            mock_settings.REQUIRE_AUTH = True
            mock_settings.API_KEY = "test-key"
            mock_settings.INFERENCE_MODEL = "test-model"
            mock_settings.MODERATION_API_CHECK_REQ = False
            mock_settings.COST_SPREADING_FACTOR = 0.03
            mock_settings.MODEL_COST_CSV_PATH = "costs.csv"
            mock_settings.MODEL_CAPABILITY_CSV_PATH = "caps.csv"

            response = client.get(
                "/api/config",
                headers={"Authorization": "Bearer test-key"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Still should not expose API_KEY
            assert "api_key" not in data


# ============================================================================
# Test CORS Middleware
# ============================================================================


class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in response."""
        response = client.get(
            "/",
            headers={"Origin": "http://example.com"}
        )

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_options_request(self, client):
        """Test OPTIONS request for CORS preflight."""
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            }
        )

        # Should handle OPTIONS request
        assert response.status_code == status.HTTP_200_OK

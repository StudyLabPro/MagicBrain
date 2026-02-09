"""
Tests for KnowledgeBase API integration.

Verifies:
- Loading twins from API
- Graceful degradation on errors
- 404 handling
- Timeout handling
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from magicbrain.integration.knowledgebase_client import KnowledgeBaseClient
from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin


class TestKnowledgeBaseAPI:

    def test_load_existing_twin(self):
        """Test loading existing twin from API."""
        client = KnowledgeBaseClient()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "student_id": "student123",
            "learning_style": "visual",
            "mastery_scores": {"topic1": 0.8, "topic2": 0.6},
            "topic_neurons": {"topic1": 10, "topic2": 20},
        }

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            twin = client._load_twin_from_kb("student123")

            # Verify twin loaded
            assert twin is not None
            assert twin.student_id == "student123"
            assert twin.learning_style == "visual"
            assert "topic1" in twin.mastery_scores
            assert twin.mastery_scores["topic1"] == 0.8

    def test_load_nonexistent_twin(self):
        """Test that 404 returns None (normal case)."""
        client = KnowledgeBaseClient()

        mock_response = Mock()
        mock_response.status_code = 404

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            twin = client._load_twin_from_kb("nonexistent")

            # Should return None without crashing
            assert twin is None

    def test_load_timeout_graceful(self):
        """Test timeout doesn't crash (graceful degradation)."""
        client = KnowledgeBaseClient()

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("timeout")

            twin = client._load_twin_from_kb("student123")

            # Should return None (graceful degradation)
            assert twin is None

    def test_load_http_error_graceful(self):
        """Test HTTP error doesn't crash."""
        client = KnowledgeBaseClient()

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.HTTPError("server error")

            twin = client._load_twin_from_kb("student123")

            # Graceful degradation
            assert twin is None

    def test_load_unexpected_error_graceful(self):
        """Test unexpected error doesn't crash."""
        client = KnowledgeBaseClient()

        with patch('httpx.Client') as mock_client_class:
            mock_client_class.side_effect = ValueError("unexpected error")

            twin = client._load_twin_from_kb("student123")

            # Graceful degradation
            assert twin is None

    def test_api_call_with_auth(self):
        """Test that API key is sent in headers."""
        client = KnowledgeBaseClient(api_key="test_key_123")

        mock_response = Mock()
        mock_response.status_code = 404

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            client._load_twin_from_kb("student123")

            # Verify headers were passed
            call_args = mock_client.get.call_args
            headers = call_args.kwargs.get('headers', {})
            assert 'Authorization' in headers
            assert headers['Authorization'] == 'Bearer test_key_123'

    def test_api_call_correct_url(self):
        """Test that correct URL is called."""
        client = KnowledgeBaseClient(base_url="http://kb-service:8000")

        mock_response = Mock()
        mock_response.status_code = 404

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            client._load_twin_from_kb("student123")

            # Verify URL
            call_args = mock_client.get.call_args
            url = call_args.args[0]
            assert url == "http://kb-service:8000/api/v1/neural-twins/student123"

    def test_timeout_value(self):
        """Test that timeout is set to 5 seconds."""
        client = KnowledgeBaseClient()

        mock_response = Mock()
        mock_response.status_code = 404

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            client._load_twin_from_kb("student123")

            # Verify Client instantiation had timeout
            call_args = mock_client_class.call_args
            assert call_args.kwargs.get('timeout') == 5.0

    def test_restore_last_practice_times(self):
        """Test that last_practice datetimes are restored."""
        client = KnowledgeBaseClient()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "student_id": "student123",
            "learning_style": "adaptive",
            "mastery_scores": {"topic1": 0.8},
            "topic_neurons": {"topic1": 10},
            "last_practice": {
                "topic1": "2026-01-15T10:30:00",
            }
        }

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = mock_response

            twin = client._load_twin_from_kb("student123")

            # Verify datetime restored
            assert twin is not None
            assert "topic1" in twin.last_practice
            from datetime import datetime
            assert isinstance(twin.last_practice["topic1"], datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

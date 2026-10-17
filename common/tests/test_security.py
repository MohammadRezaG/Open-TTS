import unittest
from unittest.mock import patch

from open_tts_common import AppConfig, OwnerSessions, is_loopback, is_network_allowed


class SecurityTests(unittest.TestCase):
    def test_owner_code_is_one_time_and_session_expires(self):
        sessions = OwnerSessions(timeout_minutes=60)
        with patch("open_tts_common.security.time.monotonic", side_effect=[10, 10, 10, 3611, 3611]):
            code = sessions.issue_code()
            token = sessions.exchange(code)
            self.assertIsNotNone(token)
            self.assertIsNone(sessions.exchange(code))
            self.assertFalse(sessions.is_valid(token))

    def test_new_manager_invalidates_previous_session(self):
        sessions = OwnerSessions()
        token = sessions.exchange(sessions.issue_code())
        self.assertTrue(sessions.is_valid(token))
        self.assertFalse(OwnerSessions().is_valid(token))

    def test_network_allowlist_always_allows_loopback(self):
        self.assertTrue(is_loopback("::1"))
        self.assertTrue(is_network_allowed("127.0.0.1", ()))
        self.assertTrue(is_network_allowed("192.0.2.9", ("192.0.2.0/24",)))
        self.assertFalse(is_network_allowed("198.51.100.2", ("192.0.2.0/24",)))

    def test_config_rejects_invalid_cidr(self):
        with self.assertRaisesRegex(ValueError, "Invalid allowed network CIDR"):
            AppConfig(allowed_networks=("not-a-cidr",))


if __name__ == "__main__":
    unittest.main()

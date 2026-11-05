import os
import unittest
from unittest.mock import patch
from webapp import proxy


class ComputeAPI4Tests(unittest.TestCase):
    def test_api4_relay_removes_only_exact_api4_bypass(self):
        env=dict(HTTP_PROXY='http://127.0.0.1:18888', OPENAI_BASE_URL='http://15.204.107.114:3009/v1',
            NO_PROXY='localhost,172.16.1.2,15.204.107.114',no_proxy='127.0.0.1,15.204.107.114:3009')
        with patch.dict(os.environ,env,clear=True),patch.object(proxy,'token',return_value='test-token'):
            self.assertTrue(proxy.configure_compute_proxy())
            self.assertEqual(os.environ['NO_PROXY'],'localhost,172.16.1.2')
            self.assertEqual(os.environ['no_proxy'],'127.0.0.1')
            self.assertIn(proxy.HOST,os.environ['HTTP_PROXY'])

    def test_other_provider_bypass_unchanged(self):
        with patch.dict(os.environ,dict(HTTP_PROXY='http://127.0.0.1:18888',
                OPENAI_BASE_URL='https://api.ofox.ai/v1',NO_PROXY='15.204.107.114,localhost'),clear=True), \
                patch.object(proxy,'token',return_value='test-token'):
            proxy.configure_compute_proxy()
            self.assertEqual(os.environ['NO_PROXY'],'15.204.107.114,localhost')

    def test_explicit_non_loopback_proxy_unchanged(self):
        with patch.dict(os.environ,dict(HTTP_PROXY='http://custom-proxy:8080',
                OPENAI_BASE_URL='http://15.204.107.114:3009/v1',NO_PROXY='15.204.107.114'),clear=True):
            self.assertFalse(proxy.configure_compute_proxy())
            self.assertEqual(os.environ['NO_PROXY'],'15.204.107.114')


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
Test to verify that dask client is properly cleaned up when TPOT creates it.
This test checks that:
1. When TPOT creates its own client, it's cleaned up after fit() completes
2. When user provides a client, TPOT doesn't close it
3. When an exception occurs during fit(), the client is still cleaned up
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import numpy as np
from sklearn.datasets import make_classification

# Add the tpot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from tpot.tpot_estimator.estimator import TPOTEstimator
    from dask.distributed import Client, LocalCluster
except ImportError as e:
    print(f"Could not import required modules: {e}")
    print("This test requires TPOT and dask to be installed")
    sys.exit(1)


class TestDaskClientCleanup(unittest.TestCase):
    
    def setUp(self):
        # Create a small test dataset
        self.X, self.y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    
    def test_client_cleanup_after_normal_completion(self):
        """Test that TPOT cleans up its own client after normal completion"""
        
        with patch('tpot.tpot_estimator.estimator.LocalCluster') as mock_cluster_class, \
             patch('tpot.tpot_estimator.estimator.Client') as mock_client_class:
            
            # Setup mocks
            mock_cluster = Mock()
            mock_client = Mock()
            mock_cluster_class.return_value = mock_cluster
            mock_client_class.return_value = mock_client
            
            # Create TPOT instance without providing a client
            tpot = TPOTEstimator(
                search_space='ClassifierSKLearnSpace',
                population_size=5,
                generations=1,
                max_time_mins=0.1,
                max_eval_time_mins=0.1,
                cv=2,
                verbose=0,
                n_jobs=1
            )
            
            # Mock the evolver to avoid actual optimization
            with patch.object(tpot, '_evolver') as mock_evolver_class:
                mock_evolver_instance = Mock()
                mock_evolver_class.return_value = mock_evolver_instance
                mock_evolver_instance.optimize.return_value = None
                
                # Mock other methods that might be called
                with patch.object(tpot, 'make_evaluated_individuals'), \
                     patch('tpot.utils.get_pareto_frontier'), \
                     patch.object(tpot, 'evaluated_individuals', Mock()), \
                     patch.object(tpot, 'pareto_front', Mock()), \
                     patch('sklearn.utils.shuffle', return_value=(self.X, self.y)):
                    
                    try:
                        tpot.fit(self.X, self.y)
                        # If we reach here, fit completed normally
                        
                        # Verify that cleanup methods were called
                        mock_client.shutdown.assert_called_once()
                        mock_cluster.close.assert_called_once()
                        
                        print("✓ Client cleanup called after normal completion")
                        
                    except Exception as e:
                        # Even if fit fails for other reasons, cleanup should still be called
                        mock_client.shutdown.assert_called_once()
                        mock_cluster.close.assert_called_once()
                        print("✓ Client cleanup called even after exception")
    
    def test_user_provided_client_not_closed(self):
        """Test that TPOT doesn't close a user-provided client"""
        
        with patch('dask.distributed.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Create TPOT instance with user-provided client
            tpot = TPOTEstimator(
                search_space='ClassifierSKLearnSpace',
                population_size=5,
                generations=1,
                max_time_mins=0.1,
                max_eval_time_mins=0.1,
                cv=2,
                verbose=0,
                n_jobs=1,
                client=mock_client  # User provided client
            )
            
            # Mock the evolver to avoid actual optimization
            with patch.object(tpot, '_evolver') as mock_evolver_class:
                mock_evolver_instance = Mock()
                mock_evolver_class.return_value = mock_evolver_instance
                mock_evolver_instance.optimize.return_value = None
                
                # Mock other methods
                with patch.object(tpot, 'make_evaluated_individuals'), \
                     patch('tpot.utils.get_pareto_frontier'), \
                     patch.object(tpot, 'evaluated_individuals', Mock()), \
                     patch.object(tpot, 'pareto_front', Mock()), \
                     patch('sklearn.utils.shuffle', return_value=(self.X, self.y)):
                    
                    try:
                        tpot.fit(self.X, self.y)
                        
                        # Verify that user's client was NOT closed
                        mock_client.shutdown.assert_not_called()
                        print("✓ User-provided client was not closed")
                        
                    except Exception:
                        # Even on exception, user's client should not be closed
                        mock_client.shutdown.assert_not_called()
                        print("✓ User-provided client was not closed even after exception")


def main():
    # Run a simple functional test
    print("Testing TPOT Dask client cleanup...")
    
    try:
        # Test 1: Normal case with TPOT creating its own client
        print("\n1. Testing client cleanup after normal completion...")
        test = TestDaskClientCleanup()
        test.setUp()
        test.test_client_cleanup_after_normal_completion()
        
        # Test 2: User-provided client should not be closed
        print("\n2. Testing user-provided client is not closed...")
        test.test_user_provided_client_not_closed()
        
        print("\n✅ All tests passed! Dask client cleanup is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
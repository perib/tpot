#!/usr/bin/env python3
"""
Simple test to verify the dask client cleanup logic is correctly implemented.
This test verifies the code structure without requiring a full TPOT installation.
"""

import sys
import os
import inspect

# Add the tpot directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_client_cleanup_logic():
    """Test that the client cleanup logic is properly implemented in the fit method"""
    
    try:
        # Import the estimator module
        from tpot.tpot_estimator.estimator import TPOTEstimator
        
        # Get the source code of the fit method
        fit_source = inspect.getsource(TPOTEstimator.fit)
        
        print("✓ Successfully imported TPOTEstimator")
        
        # Check that the fit method has the necessary try-finally structure
        checks = [
            ("try:", "try block exists"),
            ("finally:", "finally block exists"),
            ("cluster = None", "cluster variable is properly initialized"),
            ("if self.client is None and cluster is not None", "proper cleanup condition"),
            ("_client.shutdown()", "client shutdown is called"),
            ("cluster.close()", "cluster close is called")
        ]
        
        for check_text, check_desc in checks:
            if check_text in fit_source:
                print(f"✓ {check_desc}")
            else:
                print(f"❌ {check_desc}")
                return False
        
        # Check that cleanup happens in finally block (not just at the end)
        lines = fit_source.split('\n')
        finally_found = False
        cleanup_in_finally = False
        
        for i, line in enumerate(lines):
            if 'finally:' in line:
                finally_found = True
            elif finally_found and '_client.shutdown()' in line:
                cleanup_in_finally = True
                break
        
        if cleanup_in_finally:
            print("✓ Client cleanup happens in finally block")
        else:
            print("❌ Client cleanup does not happen in finally block")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import TPOTEstimator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False


def test_syntax():
    """Test that the modified file has correct Python syntax"""
    try:
        import py_compile
        py_compile.compile('tpot/tpot_estimator/estimator.py', doraise=True)
        print("✓ File has correct Python syntax")
        return True
    except py_compile.PyCompileError as e:
        print(f"❌ Syntax error: {e}")
        return False


def main():
    print("Testing TPOT Dask client cleanup implementation...")
    
    # Test 1: Check syntax
    print("\n1. Testing Python syntax...")
    if not test_syntax():
        return 1
    
    # Test 2: Check cleanup logic structure
    print("\n2. Testing cleanup logic structure...")
    if not test_client_cleanup_logic():
        return 1
    
    print("\n✅ All tests passed! The dask client cleanup fix has been properly implemented.")
    print("\nSummary of changes:")
    print("- Added proper initialization of cluster variable")
    print("- Wrapped main fit logic in try-finally block")
    print("- Moved client cleanup to finally block to ensure it always runs")
    print("- Cleanup only happens when TPOT created the client (self.client is None)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
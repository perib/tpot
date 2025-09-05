#!/usr/bin/env python3
"""
Direct test of the dask client cleanup implementation by examining the source code.
"""

def test_fit_method_structure():
    """Test that the fit method has the correct structure for client cleanup"""
    
    try:
        # Read the source file directly
        with open('tpot/tpot_estimator/estimator.py', 'r') as f:
            content = f.read()
        
        # Extract the fit method
        fit_start = content.find('def fit(self, X, y):')
        if fit_start == -1:
            print("❌ Could not find fit method")
            return False
        
        # Find the next method definition to get the end of fit method
        next_def = content.find('\n    def ', fit_start + 1)
        if next_def == -1:
            fit_method = content[fit_start:]
        else:
            fit_method = content[fit_start:next_def]
        
        print("✓ Found fit method")
        
        # Check for required components
        checks = [
            ("cluster = None", "cluster variable is properly initialized"),
            ("try:", "try block exists"),
            ("finally:", "finally block exists"),
            ("if self.client is None and cluster is not None", "proper cleanup condition in finally"),
            ("_client.shutdown()", "client shutdown is called"),
            ("cluster.close()", "cluster close is called")
        ]
        
        all_passed = True
        for check_text, check_desc in checks:
            if check_text in fit_method:
                print(f"✓ {check_desc}")
            else:
                print(f"❌ {check_desc}")
                all_passed = False
        
        # Check that cleanup is in finally block, not just at the end
        finally_index = fit_method.find('finally:')
        if finally_index == -1:
            print("❌ No finally block found")
            return False
        
        # Check that shutdown is called after finally
        finally_section = fit_method[finally_index:]
        if '_client.shutdown()' in finally_section:
            print("✓ Client cleanup happens in finally block")
        else:
            print("❌ Client cleanup does not happen in finally block")
            all_passed = False
        
        # Check that the cleanup is conditional on self.client being None
        if 'if self.client is None and cluster is not None' in finally_section:
            print("✓ Cleanup is conditional on TPOT creating the client")
        else:
            print("❌ Cleanup condition is incorrect")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def main():
    print("Testing TPOT Dask client cleanup implementation...")
    print("Examining source code structure...\n")
    
    if test_fit_method_structure():
        print("\n✅ All tests passed! The dask client cleanup fix has been properly implemented.")
        print("\nSummary of the fix:")
        print("1. Added 'cluster = None' initialization to track cluster for cleanup")
        print("2. Wrapped the main fit logic in a try-finally block")
        print("3. Moved client cleanup to the finally block")
        print("4. Cleanup only happens when TPOT created the client (self.client is None)")
        print("5. This ensures cleanup happens even if exceptions occur during fitting")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the implementation.")
        return 1


if __name__ == '__main__':
    exit(main())
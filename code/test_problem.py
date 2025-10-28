#!/usr/bin/env python3
"""
Comprehensive test script for the Problem class.

This script tests all functionality of the problem module including:
- Problem creation and initialization
- Parameter management
- Function setting and validation
- Problem validation
- Error handling
- Integration with different problem types

Usage:
    python test_problem.py
"""

import sys
import os
import numpy as np
import time
from typing import Dict, Any

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from ooc1d.core.problem import Problem

def run_basic_functionality_tests() -> bool:
    """Test basic problem operations."""
    print("Testing Basic Functionality")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Default problem creation
    print("Test 1: Default problem creation")
    try:
        default_problem = Problem()
        assert default_problem.neq == 2
        assert default_problem.domain_start == 0.0
        assert default_problem.domain_length == 1.0
        assert default_problem.domain_end == 1.0
        assert default_problem.name == "unnamed_problem"
        assert default_problem.type == "keller_segel"
        print("  ✓ Default problem creation")
    except Exception as e:
        print(f"  ✗ Default problem creation failed: {e}")
        all_passed = False
    
    # Test 2: Custom problem creation
    print("Test 2: Custom problem creation")
    try:
        custom_problem = Problem(
            neq=3,
            domain_start=1.0,
            domain_length=2.0,
            parameters=np.array([1.0, 0.5, 2.0, 0.0]),
            problem_type="organ_on_chip",
            name="custom_problem"
        )
        
        assert custom_problem.neq == 3
        assert custom_problem.domain_start == 1.0
        assert custom_problem.domain_length == 2.0
        assert custom_problem.domain_end == 3.0
        assert custom_problem.name == "custom_problem"
        assert custom_problem.type == "organ_on_chip"
        print("  ✓ Custom problem creation")
    except Exception as e:
        print(f"  ✗ Custom problem creation failed: {e}")
        all_passed = False
    
    return all_passed

def run_parameter_management_tests() -> bool:
    """Test parameter management functions."""
    print("\nTesting Parameter Management")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Parameter setting and getting
    print("Test 1: Parameter setting and getting")
    try:
        problem = Problem(parameters=np.array([1.0, 2.0, 3.0]))
        
        # Test getting parameters
        assert problem.get_parameter(0) == 1.0
        assert problem.get_parameter(1) == 2.0
        assert problem.get_parameter(2) == 3.0
        
        # Test setting parameters
        problem.set_parameter(0, 1.5)
        problem.set_parameter(1, 2.5)
        problem.set_parameter(2, 3.5)
        
        assert problem.get_parameter(0) == 1.5
        assert problem.get_parameter(1) == 2.5
        assert problem.get_parameter(2) == 3.5
        
        print("  ✓ Parameter setting and getting")
    except Exception as e:
        print(f"  ✗ Parameter setting and getting failed: {e}")
        all_passed = False
    
    # Test 2: Parameter array setting
    print("Test 2: Parameter array setting")
    try:
        problem = Problem(parameters=np.array([1.0, 2.0]))
        problem.set_parameters(np.array([3.0, 4.0, 5.0]))
        
        assert problem.get_parameter(0) == 3.0
        assert problem.get_parameter(1) == 4.0
        assert problem.get_parameter(2) == 5.0
        
        print("  ✓ Parameter array setting")
    except Exception as e:
        print(f"  ✗ Parameter array setting failed: {e}")
        all_passed = False
    
    return all_passed

def run_function_setting_tests() -> bool:
    """Test function setting and management."""
    print("\nTesting Function Setting")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Initial condition setting
    print("Test 1: Initial condition setting")
    try:
        problem = Problem(neq=3)
        
        # Set initial conditions
        problem.set_initial_condition(0, lambda s: np.sin(s))
        problem.set_initial_condition(1, lambda s: np.cos(s))
        problem.set_initial_condition(2, lambda s: np.exp(-s))
        
        # Test functions
        test_s = np.array([0.0, 1.0, 2.0])
        result0 = problem.u0[0](test_s)
        result1 = problem.u0[1](test_s)
        result2 = problem.u0[2](test_s)
        
        assert np.allclose(result0, np.sin(test_s))
        assert np.allclose(result1, np.cos(test_s))
        assert np.allclose(result2, np.exp(-test_s))
        
        print("  ✓ Initial condition setting")
    except Exception as e:
        print(f"  ✗ Initial condition setting failed: {e}")
        all_passed = False
    
    # Test 2: Force function setting
    print("Test 2: Force function setting")
    try:
        problem = Problem(neq=2)
        
        # Set force functions
        problem.set_force(0, lambda s, t: s + t)
        problem.set_force(1, lambda s, t: s * t)
        
        # Test evaluation
        test_s = np.array([1.0, 2.0, 3.0])
        test_t = 2.0
        result0 = problem.force[0](test_s, test_t)
        result1 = problem.force[1](test_s, test_t)
        
        assert np.allclose(result0, test_s + test_t)
        assert np.allclose(result1, test_s * test_t)
        
        print("  ✓ Force function setting")
    except Exception as e:
        print(f"  ✗ Force function setting failed: {e}")
        all_passed = False
    
    # Test 3: Solution function setting
    print("Test 3: Solution function setting")
    try:
        problem = Problem(neq=2)
        
        # Set solution functions
        problem.set_solution(0, lambda s, t: np.sin(s) * np.exp(-t))
        problem.set_solution(1, lambda s, t: np.cos(s) * np.exp(-t))
        
        # Test evaluation
        test_s = np.array([0.0, np.pi/2, np.pi])
        test_t = 1.0
        result0 = problem.solution[0](test_s, test_t)
        result1 = problem.solution[1](test_s, test_t)
        
        assert np.allclose(result0, np.sin(test_s) * np.exp(-test_t))
        assert np.allclose(result1, np.cos(test_s) * np.exp(-test_t))
        
        print("  ✓ Solution function setting")
    except Exception as e:
        print(f"  ✗ Solution function setting failed: {e}")
        all_passed = False
    
    return all_passed

def run_validation_tests() -> bool:
    """Test problem validation functionality."""
    print("\nTesting Problem Validation")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Valid problem validation
    print("Test 1: Valid problem validation")
    try:
        valid_problem = Problem(
            neq=2,
            domain_start=0.0,
            domain_length=1.0,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="valid_test"
        )
        
        is_valid = valid_problem.validate_problem(verbose=False)
        assert is_valid == True
        print("  ✓ Valid problem validation")
    except Exception as e:
        print(f"  ✗ Valid problem validation failed: {e}")
        all_passed = False
    
    # Test 2: Invalid problem (e.g., negative domain length)
    print("Test 2: Invalid problem")
    try:
        invalid_problem = Problem(
            neq=2,
            domain_start=0.0,
            domain_length=-1.0,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="invalid_test"
        )
        
        is_valid = invalid_problem.validate_problem(verbose=False)
        assert is_valid == False
        print("  ✓ Invalid problem correctly identified")
    except Exception as e:
        print(f"  ✗ Invalid problem test failed: {e}")
        all_passed = False
    
    return all_passed

def run_problem_type_tests() -> bool:
    """Test different problem types."""
    print("\nTesting Problem Types")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Keller-Segel problem
    print("Test 1: Keller-Segel problem")
    try:
        ks_problem = Problem(
            neq=2,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="KS_test"
        )
        
        # Set Keller-Segel specific functions
        ks_problem.set_chemotaxis(
            lambda x: np.ones_like(x),
            lambda x: np.zeros_like(x)
        )
        
        ks_problem.set_initial_condition(0, lambda s: np.exp(-s**2))
        ks_problem.set_initial_condition(1, lambda s: np.sin(np.pi * s))
        
        is_valid = ks_problem.validate_problem(verbose=False)
        functions_ok = ks_problem.test_functions(verbose=False)
        
        assert is_valid and functions_ok
        print("  ✓ Keller-Segel problem")
    except Exception as e:
        print(f"  ✗ Keller-Segel problem failed: {e}")
        all_passed = False
    
    # Test 2: Organ-on-Chip problem
    print("Test 2: Organ-on-Chip problem")
    try:
        ooc_problem = Problem(
            neq=4,
            parameters=np.array([1e-9, 0.001, 1e-4, 1e-5, 1e-3]),
            problem_type="organ_on_chip",
            name="OoC_test"
        )
        
        # Set initial conditions
        ooc_problem.set_initial_condition(0, lambda s: 0.5 * np.ones_like(s))
        ooc_problem.set_initial_condition(1, lambda s: np.zeros_like(s))
        ooc_problem.set_initial_condition(2, lambda s: 0.1 * np.ones_like(s))
        ooc_problem.set_initial_condition(3, lambda s: 0.05 * np.ones_like(s))
        
        # Set source terms
        for i in range(4):
            ooc_problem.set_force(i, lambda s, t: np.zeros_like(s))
        
        is_valid = ooc_problem.validate_problem(verbose=False)
        functions_ok = ooc_problem.test_functions(verbose=False)
        
        assert is_valid and functions_ok
        print("  ✓ Organ-on-Chip problem")
    except Exception as e:
        print(f"  ✗ Organ-on-Chip problem failed: {e}")
        all_passed = False
    
    return all_passed

def run_error_handling_tests() -> bool:
    """Test error handling and edge cases."""
    print("\nTesting Error Handling")
    print("-" * 40)
    
    all_passed = True
    
    # Test 1: Invalid parameter access
    print("Test 1: Invalid parameter access")
    try:
        problem = Problem(parameters=np.array([1.0, 2.0]))
        
        # Should raise IndexError
        try:
            invalid_param = problem.get_parameter(5)
            print("  ✗ Should have raised IndexError")
            all_passed = False
        except IndexError:
            print("  ✓ IndexError properly raised")
        except Exception as e:
            print(f"  ✗ Wrong exception type: {e}")
            all_passed = False
            
    except Exception as e:
        print(f"  ✗ Error handling test setup failed: {e}")
        all_passed = False
    
    # Test 2: Invalid function setting
    print("Test 2: Invalid function setting")
    try:
        problem = Problem()
        
        # Invalid function (not callable)
        try:
            problem.set_function("invalid_func", 12345)
            print("  ✗ Should have raised TypeError")
            all_passed = False
        except TypeError:
            print("  ✓ TypeError properly raised")
        except Exception as e:
            print(f"  ✗ Wrong exception type: {e}")
            all_passed = False
            
    except Exception as e:
        print(f"  ✗ Error handling test setup failed: {e}")
        all_passed = False
    
    return all_passed

def run_test_problems() -> bool:
    """Test the predefined test problems."""
    print("\nTesting Predefined Test Problems")
    print("-" * 40)
    
    all_passed = True
    
    try:
        test_problems = Problem.create_test_problems()
        
        expected_problems = ["ks_basic", "ooc_basic", "custom_geometry", "analytical", "invalid"]
        
        for name in expected_problems:
            if name not in test_problems:
                print(f"  ✗ Missing test problem: {name}")
                all_passed = False
                continue
            
            problem = test_problems[name]
            
            # Run self-test on each problem
            if name != "invalid":  # Invalid problem is expected to fail validation
                test_result = problem.run_self_test(verbose=False)
                if test_result:
                    print(f"  ✓ {name} problem self-test passed")
                else:
                    print(f"  ✗ {name} problem self-test failed")
                    all_passed = False
            else:
                # Invalid case should fail validation
                is_valid = problem.validate_problem(verbose=False)
                if not is_valid:
                    print(f"  ✓ {name} problem correctly identified as invalid")
                else:
                    print(f"  ✗ {name} problem should be invalid")
                    all_passed = False
        
    except Exception as e:
        print(f"  ✗ Test problems creation failed: {e}")
        all_passed = False
    
    return all_passed

def run_performance_tests() -> bool:
    """Test performance with function evaluations."""
    print("\nTesting Performance")
    print("-" * 40)
    
    all_passed = True
    
    # Test function evaluation performance
    print("Test 1: Function evaluation performance")
    try:
        problem = Problem(neq=2)
        
        # Set computationally simple functions
        problem.set_initial_condition(0, lambda s: np.sin(s))
        problem.set_initial_condition(1, lambda s: np.cos(s))
        problem.set_force(0, lambda s, t: s * t)
        problem.set_force(1, lambda s, t: s**2 + t**2)
        
        # Time function evaluations
        large_s = np.linspace(0, 10, 10000)
        t_val = 1.0
        
        start_time = time.time()
        
        for _ in range(100):  # Multiple evaluations
            result0 = problem.u0[0](large_s)
            result1 = problem.u0[1](large_s)
            result2 = problem.force[0](large_s, t_val)
            result3 = problem.force[1](large_s, t_val)
        
        eval_time = time.time() - start_time
        
        print(f"  ✓ Function evaluations completed in {eval_time:.3f} seconds")
        
        if eval_time > 5.0:  # Should be much faster
            print(f"  ⚠ Warning: Function evaluations seem slow ({eval_time:.3f}s)")
    
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        all_passed = False
    
    return all_passed

def main():
    """Run all problem tests."""
    print("="*60)
    print("PROBLEM MODULE COMPREHENSIVE TEST")
    print("="*60)
    
    test_functions = [
        run_basic_functionality_tests,
        run_parameter_management_tests,
        run_function_setting_tests,
        run_validation_tests,
        run_problem_type_tests,
        run_error_handling_tests,
        run_test_problems,
        run_performance_tests
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test function {test_func.__name__} crashed: {e}")
            results.append(False)
    
    total_time = time.time() - total_start_time
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    print(f"Total time: {total_time:.3f} seconds")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return_code = 0
    else:
        print("✗ SOME TESTS FAILED!")
        return_code = 1
    
    # Demonstrate usage example
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    
    try:
        # Create and test a Keller-Segel problem
        example_problem = Problem(
            neq=2,
            domain_start=0.0,
            domain_length=1.0,
            parameters=np.array([1.0, 2.0, 0.1, 0.0]),
            problem_type="keller_segel",
            name="example_ks"
        )
        
        # Set up chemotaxis
        example_problem.set_chemotaxis(
            lambda x: 1.0 / (1.0 + x),
            lambda x: -1.0 / (1.0 + x)**2
        )
        
        # Set initial conditions
        example_problem.set_initial_condition(0, lambda s: np.exp(-s**2))
        example_problem.set_initial_condition(1, lambda s: np.sin(np.pi * s))
        
        # Set extrema for visualization
        example_problem.set_extrema((0.0, 0.0), (1.0, 0.0))
        
        print("Created example Keller-Segel problem:")
        print(f"  Name: {example_problem.name}")
        print(f"  Type: {example_problem.type}")
        print(f"  Equations: {example_problem.neq}")
        print(f"  Domain: [{example_problem.domain_start}, {example_problem.domain_end}]")
        print(f"  Parameters: {example_problem.parameters}")
        
        print("\nRunning self-test:")
        example_problem.run_self_test(verbose=True)
        
    except Exception as e:
        print(f"Usage example failed: {e}")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
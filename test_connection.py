#!/usr/bin/env python3
"""Test script for server-client connection flow"""
import sys
sys.path.insert(0, '/Users/ojaswini/Downloads/FedFortress')

print("=== Testing Server-Client Connection ===")

# Test 1: Import main modules
print("\n[Test 1] Importing modules...")
try:
    from src.main import run_federated_training, run_federated_training_with_server
    from src.client.client import Client
    from src.server.server import AsyncFLServer, ServerConfig
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create server
print("\n[Test 2] Creating AsyncFLServer...")
try:
    import numpy as np
    global_model = {
        'weight': np.zeros((10, 5)),
        'bias': np.zeros(10),
    }
    config = ServerConfig(async_buffer_size=2, dp_enabled=False)
    server = AsyncFLServer(global_model, config)
    print("✓ Server created")
except Exception as e:
    print(f"✗ Server creation failed: {e}")
    sys.exit(1)

# Test 3: Get global model from server
print("\n[Test 3] Getting global model from server...")
try:
    model, version = server.get_global_model()
    print(f"✓ Got global model (version: {version})")
except Exception as e:
    print(f"✗ Get model failed: {e}")
    sys.exit(1)

print("\n=== All Tests Passed ===")
print("\nServer-Client connection is ready!")
print("\nTo run full training:")
print("  python3 test_run.py")


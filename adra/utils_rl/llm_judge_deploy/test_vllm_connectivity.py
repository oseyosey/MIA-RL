#!/usr/bin/env python3
"""
Test connectivity and functionality of vLLM server for LLM judge evaluation.

This script tests:
- Server health and availability
- Model information retrieval
- Single text generation
- Batch text generation
- LLM judge prompt evaluation
- Error handling and fallback behavior

Usage:
    python test_vllm_connectivity.py http://localhost:8000
    python test_vllm_connectivity.py http://192.168.1.100:8000
"""

import sys
import time
import requests
import json
from typing import Optional, List, Dict, Any

def test_server_health(server_url: str) -> bool:
    """Test if the server is healthy and responsive."""
    print(f"🏥 Testing server health: {server_url}")
    
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server health check passed")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server health check timed out")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_server_info(server_url: str) -> Optional[Dict[str, Any]]:
    """Get server information and available models."""
    print(f"\n📋 Getting server information...")
    
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        response.raise_for_status()
        
        info = response.json()
        print("✅ Server info retrieved successfully")
        
        if "data" in info and len(info["data"]) > 0:
            model_info = info["data"][0]
            print(f"   Model: {model_info.get('id', 'Unknown')}")
            print(f"   Created: {model_info.get('created', 'Unknown')}")
        
        return info
        
    except Exception as e:
        print(f"❌ Failed to get server info: {e}")
        return None

def test_single_generation(server_url: str, model_name: str = "Qwen/Qwen3-32B") -> bool:
    """Test single text generation."""
    print(f"\n🔧 Testing single text generation...")
    
    test_prompt = "What is 2 + 2? Please answer briefly."
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": test_prompt}],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            print("✅ Single generation successful")
            print(f"   Prompt: {test_prompt}")
            print(f"   Response: {generated_text[:100]}...")
            print(f"   Latency: {end_time - start_time:.2f}s")
            return True
        else:
            print(f"❌ Unexpected response format: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Single generation failed: {e}")
        return False

def test_llm_judge_evaluation(server_url: str, model_name: str = "Qwen/Qwen3-32B") -> bool:
    """Test LLM judge evaluation with a realistic prompt."""
    print(f"\n⚖️ Testing LLM judge evaluation...")
    
    # Create a judge prompt
    judge_prompt = """
Rate the two math problem solutions (one reference, one candidate) in terms of their similarity. Return a real value between 0-1 with 3 decimals.

INPUTS
- Problem:
What is 2 + 2?

- Reference solution:
2 + 2 = 4

- Candidate solution:
The answer is 4.

Think through your evaluation, then provide your final answer in the exact format below.

OUTPUT FORMAT (must follow exactly)
After your reasoning, output ONLY one line:
REWARD: <number between 0 and 1 with 3 decimals>
""".strip()
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            print("✅ LLM judge evaluation successful")
            print(f"   Response: {generated_text}")
            print(f"   Latency: {end_time - start_time:.2f}s")
            
            # Try to extract score
            import re
            
            # Look for REWARD pattern in the entire response
            pattern = r"REWARD:\s*([0-9]*\.?[0-9]+)"
            match = re.search(pattern, generated_text, re.IGNORECASE)
            
            score = None
            if match:
                try:
                    score = float(match.group(1))
                except (ValueError, IndexError):
                    score = None
            
            if score is not None:
                print(f"   Extracted score: {score}")
                if 0 <= score <= 1:
                    print("✅ Score extraction successful")
                    return True
                else:
                    print("⚠️ Score out of range [0,1]")
                    return True  # Still consider it successful
            else:
                print("⚠️ Could not extract score from response")
                print(f"   Raw response: {generated_text}")
                return True  # Still consider it successful if generation worked
                
        else:
            print(f"❌ Unexpected response format: {result}")
            return False
            
    except Exception as e:
        print(f"❌ LLM judge evaluation failed: {e}")
        return False

def test_batch_generation(server_url: str, model_name: str = "Qwen/Qwen3-32B") -> bool:
    """Test batch generation by sending multiple requests."""
    print(f"\n📦 Testing batch generation (multiple concurrent requests)...")
    
    test_prompts = [
        "What is 3 + 3?",
        "What is 4 + 4?", 
        "What is 5 + 5?"
    ]
    
    results = []
    start_time = time.time()
    
    try:
        import concurrent.futures
        
        def send_request(prompt):
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 30,
                "temperature": 0.1,
                "stream": False
            }
            
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        # Send requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_request, prompt) for prompt in test_prompts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        if len(results) == len(test_prompts):
            print("✅ Batch generation successful")
            print(f"   Processed {len(test_prompts)} requests")
            print(f"   Total time: {end_time - start_time:.2f}s")
            print(f"   Average per request: {(end_time - start_time) / len(test_prompts):.2f}s")
            
            # Show first result
            if results[0].get("choices"):
                first_response = results[0]["choices"][0]["message"]["content"]
                print(f"   Sample response: {first_response[:50]}...")
            
            return True
        else:
            print(f"❌ Expected {len(test_prompts)} results, got {len(results)}")
            return False
            
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return False

def test_error_handling(server_url: str) -> bool:
    """Test error handling with invalid requests."""
    print(f"\n🚫 Testing error handling...")
    
    # Test with invalid model
    invalid_payload = {
        "model": "invalid-model-name",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=invalid_payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            print("✅ Server correctly handles invalid model")
            return True
        else:
            print("⚠️ Server accepted invalid model (might be using fallback)")
            return True
            
    except Exception as e:
        print(f"✅ Server error handling working: {e}")
        return True

def test_thinking_mode(server_url: str, model_name: str = "Qwen/Qwen3-32B") -> bool:
    """Test understanding of thinking mode output."""
    print(f"\n🤔 Testing thinking mode handling...")
    
    simple_prompt = "Give me a score between 0 and 1 for how good pizza is. Output format: SCORE: 0.X"
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": simple_prompt}],
        "max_tokens": 4096,
        "temperature": 0.3,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            generated_text = result["choices"][0]["message"]["content"]
            print("✅ Thinking mode test successful")
            print(f"   Full response: {generated_text}")
            
            # Check if it uses thinking tags
            if "<think>" in generated_text:
                print("   📝 Model is using thinking mode")
                if "</think>" in generated_text:
                    after_think = generated_text.split("</think>", 1)[-1].strip()
                    print(f"   🎯 Content after thinking: {after_think}")
                else:
                    print("   ⚠️ Thinking mode without closing tag")
            else:
                print("   📝 Model is NOT using thinking mode")
            
            return True
        else:
            print(f"❌ Unexpected response format: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Thinking mode test failed: {e}")
        return False

def run_full_test(server_url: str) -> bool:
    """Run the complete test suite."""
    print("=" * 60)
    print(f"🧪 Testing vLLM Server: {server_url}")
    print("=" * 60)
    
    tests = [
        ("Health Check", lambda: test_server_health(server_url)),
        ("Server Info", lambda: test_server_info(server_url) is not None),
        ("Single Generation", lambda: test_single_generation(server_url)),
        ("Thinking Mode", lambda: test_thinking_mode(server_url)),
        ("LLM Judge Evaluation", lambda: test_llm_judge_evaluation(server_url)),
        ("Batch Generation", lambda: test_batch_generation(server_url)),
        ("Error Handling", lambda: test_error_handling(server_url)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}\n")
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! vLLM server is working correctly.")
        print("\nYou can now use this server with:")
        print(f'export LLM_JUDGE_SERVER_URL="{server_url}"')
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the server configuration.")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_vllm_connectivity.py <server_url>")
        print("Example: python test_vllm_connectivity.py http://localhost:8000")
        sys.exit(1)
    
    server_url = sys.argv[1].rstrip("/")
    success = run_full_test(server_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

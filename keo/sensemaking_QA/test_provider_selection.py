#!/usr/bin/env python3
"""
Test script to verify the provider selection functionality
"""

import sys
import os
sys.path.append('.')

def test_provider_selection():
    """Test that the provider selection works correctly"""
    
    from answer_generator import SensemakingAnswerGenerator
    
    print("Testing provider selection...")
    
    # Test OpenAI provider (will need actual API key to work)
    print("1. Testing OpenAI provider initialization...")
    try:
        if os.getenv("OPENAI_API_KEY"):
            openai_generator = SensemakingAnswerGenerator(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                provider="openai"
            )
            print("✓ OpenAI provider initialized successfully")
        else:
            print("⚠ Skipping OpenAI test - no OPENAI_API_KEY")
    except Exception as e:
        print(f"✗ OpenAI provider test failed: {e}")
    
    # Test HuggingFace provider (will need actual token to work)
    print("2. Testing HuggingFace provider initialization...")
    try:
        if os.getenv("HF_TOKEN"):
            try:
                from huggingface_hub import InferenceClient
                hf_generator = SensemakingAnswerGenerator(
                    api_key=os.getenv("HF_TOKEN"),
                    model="google/gemma-3-4b-it",
                    provider="huggingface",
                    API_provider="featherless-ai"
                )
                print("✓ HuggingFace provider initialized successfully")
                
                # Test different API providers
                print("2a. Testing different HuggingFace API providers...")
                try:
                    hf_generator_openai = SensemakingAnswerGenerator(
                        api_key=os.getenv("HF_TOKEN"),
                        model="google/gemma-3-4b-it",
                        provider="huggingface",
                        API_provider="openai"
                    )
                    print("✓ HuggingFace with OpenAI API provider initialized successfully")
                except Exception as e:
                    print(f"⚠ HuggingFace with OpenAI API provider test failed: {e}")
                    
            except ImportError:
                print("⚠ Skipping HuggingFace test - huggingface_hub not installed")
        else:
            print("⚠ Skipping HuggingFace test - no HF_TOKEN")
    except Exception as e:
        print(f"✗ HuggingFace provider test failed: {e}")
    
    # Test invalid provider
    print("3. Testing invalid provider...")
    try:
        invalid_generator = SensemakingAnswerGenerator(
            api_key="dummy",
            model="dummy",
            provider="invalid"
        )
        print("✗ Invalid provider should have raised an error")
    except ValueError as e:
        print(f"✓ Invalid provider correctly raised error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\nProvider selection tests completed!")

if __name__ == "__main__":
    test_provider_selection()

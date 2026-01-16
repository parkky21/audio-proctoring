"""
Test script to demonstrate the Audio Proctoring POC.

This script simulates the proctoring workflow:
1. Create a session
2. Register main speaker with first audio
3. Verify same speaker (should pass)
4. Detect different speaker (should flag anomaly)

Usage:
    python test_poc.py

Make sure the server is running:
    uvicorn app.main:app --reload --port 8000
"""
import requests
import os
import sys

# Configuration
BASE_URL = "http://localhost:8000/api/v1"


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_result(result: dict, success: bool = True):
    """Print analysis result in a formatted way."""
    if success:
        print(f"  ✅ Status: Success")
    else:
        print(f"  ❌ Status: Failed")
    
    for key, value in result.items():
        print(f"  {key}: {value}")


def create_session() -> str:
    """Create a new proctoring session."""
    print_header("Creating New Session")
    
    response = requests.post(f"{BASE_URL}/session")
    
    if response.status_code == 201:
        data = response.json()
        print(f"  Session ID: {data['session_id']}")
        print(f"  Message: {data['message']}")
        return data['session_id']
    else:
        print(f"  Error: {response.text}")
        sys.exit(1)


def analyze_audio(session_id: str, audio_path: str, description: str) -> dict:
    """Analyze an audio file."""
    print_header(f"Analyzing: {description}")
    print(f"  File: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"  ⚠️ File not found: {audio_path}")
        return None
    
    with open(audio_path, 'rb') as f:
        files = {'audio': (os.path.basename(audio_path), f)}
        response = requests.post(
            f"{BASE_URL}/session/{session_id}/analyze",
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        print_result(result, not result['is_anomaly'])
        return result
    else:
        print(f"  Error: {response.text}")
        return None


def get_session_status(session_id: str):
    """Get session status."""
    print_header("Session Status")
    
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    
    if response.status_code == 200:
        data = response.json()
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Error: {response.text}")


def delete_session(session_id: str):
    """Delete the session."""
    print_header("Deleting Session")
    
    response = requests.delete(f"{BASE_URL}/session/{session_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Message: {data['message']}")
    else:
        print(f"  Error: {response.text}")


def check_health():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def main():
    print("\n" + "=" * 60)
    print("  AUDIO PROCTORING POC - TEST SCRIPT")
    print("=" * 60)
    
    # Check if server is running
    if not check_health():
        print("\n❌ Error: Server is not running!")
        print("Please start the server first:")
        print("  uvicorn app.main:app --reload --port 8000")
        sys.exit(1)
    
    print("\n✅ Server is running!")
    
    # Check for test audio files
    test_dir = "test_audio"
    if not os.path.exists(test_dir):
        print(f"\n⚠️ Test audio directory not found: {test_dir}")
        print("\nTo test the POC, please:")
        print("1. Create a 'test_audio' directory")
        print("2. Add audio files:")
        print("   - speaker1_sample1.wav (main speaker, first sample)")
        print("   - speaker1_sample2.wav (main speaker, second sample)")
        print("   - speaker2_sample1.wav (different speaker)")
        
        # Create directory and show demo without files
        os.makedirs(test_dir, exist_ok=True)
        
        # Demo mode without actual files
        print("\n" + "-" * 60)
        print("Running in DEMO mode (no audio files)...")
        print("-" * 60)
        
        # Create and delete a session to show the API works
        session_id = create_session()
        get_session_status(session_id)
        delete_session(session_id)
        
        print("\n" + "=" * 60)
        print("  DEMO COMPLETE")
        print("  Add audio files to test_audio/ for full testing")
        print("=" * 60)
        return
    
    # Expected test files
    speaker1_file1 = os.path.join(test_dir, "speaker1_sample1.wav")
    speaker1_file2 = os.path.join(test_dir, "speaker1_sample2.wav")
    speaker2_file1 = os.path.join(test_dir, "speaker2_sample1.wav")
    
    # Create session
    session_id = create_session()
    
    # Test 1: Register main speaker
    if os.path.exists(speaker1_file1):
        result = analyze_audio(session_id, speaker1_file1, "Main Speaker Registration")
        if result and result['is_first_audio']:
            print("\n  → Main speaker registered successfully!")
    
    # Test 2: Verify same speaker
    if os.path.exists(speaker1_file2):
        result = analyze_audio(session_id, speaker1_file2, "Same Speaker Verification")
        if result and result['is_main_speaker']:
            print("\n  → Same speaker verified! ✓")
    
    # Test 3: Detect different speaker
    if os.path.exists(speaker2_file1):
        result = analyze_audio(session_id, speaker2_file1, "Different Speaker Detection")
        if result and result['is_anomaly']:
            print("\n  → Foreign speaker detected! ⚠️")
    
    # Get final session status
    get_session_status(session_id)
    
    # Cleanup
    delete_session(session_id)
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

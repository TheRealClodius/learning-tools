#!/usr/bin/env python3
"""
API Key Generation Utility

Generates secure API keys for your Signal Agent API.
Can be used as a standalone script or imported as a module.
"""

import secrets
import string
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import argparse

class APIKeyManager:
    """Manage API key generation and validation"""
    
    def __init__(self, keys_file: str = "api_keys.json"):
        self.keys_file = keys_file
        self.keys_data = self._load_keys()
    
    def _load_keys(self) -> Dict:
        """Load existing API keys from file"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"keys": {}, "metadata": {"created": datetime.now().isoformat()}}
        return {"keys": {}, "metadata": {"created": datetime.now().isoformat()}}
    
    def _save_keys(self):
        """Save API keys to file"""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys_data, f, indent=2)
    
    def generate_api_key(self, 
                        name: str = None, 
                        prefix: str = "sk-signal", 
                        length: int = 32) -> Dict[str, str]:
        """
        Generate a new API key
        
        Args:
            name: Human-readable name for the key
            prefix: Prefix for the key (default: "sk-signal")
            length: Length of the random part (default: 32)
            
        Returns:
            Dict containing the key and metadata
        """
        # Generate secure random string
        alphabet = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        # Create the full API key
        api_key = f"{prefix}-{random_part}"
        
        # Hash for internal tracking
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Create metadata
        key_info = {
            "name": name or f"Generated key {len(self.keys_data['keys']) + 1}",
            "created": datetime.now().isoformat(),
            "hash": api_key_hash,
            "active": True,
            "usage_count": 0,
            "last_used": None
        }
        
        # Store in keys data
        self.keys_data["keys"][api_key] = key_info
        self.keys_data["metadata"]["last_generated"] = datetime.now().isoformat()
        
        # Save to file
        self._save_keys()
        
        return {
            "api_key": api_key,
            "hash": api_key_hash,
            "name": key_info["name"],
            "created": key_info["created"]
        }
    
    def list_keys(self) -> List[Dict]:
        """List all generated API keys (without exposing the actual keys)"""
        keys_list = []
        for api_key, info in self.keys_data["keys"].items():
            keys_list.append({
                "name": info["name"],
                "hash": info["hash"],
                "created": info["created"],
                "active": info["active"],
                "usage_count": info["usage_count"],
                "last_used": info["last_used"],
                "key_preview": f"{api_key[:15]}...{api_key[-4:]}"  # Show first 15 and last 4 chars
            })
        return keys_list
    
    def revoke_key(self, api_key_or_hash: str) -> bool:
        """Revoke an API key"""
        # Try to find by full key first
        if api_key_or_hash in self.keys_data["keys"]:
            self.keys_data["keys"][api_key_or_hash]["active"] = False
            self._save_keys()
            return True
        
        # Try to find by hash
        for api_key, info in self.keys_data["keys"].items():
            if info["hash"] == api_key_or_hash:
                self.keys_data["keys"][api_key]["active"] = False
                self._save_keys()
                return True
        
        return False
    
    def validate_key(self, api_key: str) -> bool:
        """Check if an API key is valid and active"""
        key_info = self.keys_data["keys"].get(api_key)
        if key_info and key_info.get("active", False):
            # Update usage
            key_info["usage_count"] += 1
            key_info["last_used"] = datetime.now().isoformat()
            self._save_keys()
            return True
        return False
    
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """Get information about an API key"""
        return self.keys_data["keys"].get(api_key)

def generate_simple_key(prefix: str = "sk-signal", length: int = 32) -> str:
    """Simple function to generate a single API key"""
    alphabet = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}-{random_part}"

def main():
    """Command-line interface for API key generation"""
    parser = argparse.ArgumentParser(description="Generate API keys for Signal Agent API")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a new API key")
    gen_parser.add_argument("--name", type=str, help="Name for the API key")
    gen_parser.add_argument("--prefix", type=str, default="sk-signal", help="Key prefix (default: sk-signal)")
    gen_parser.add_argument("--length", type=int, default=32, help="Length of random part (default: 32)")
    gen_parser.add_argument("--simple", action="store_true", help="Generate simple key without storage")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all generated API keys")
    
    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("key_or_hash", help="API key or hash to revoke")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate an API key")
    validate_parser.add_argument("api_key", help="API key to validate")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        if args.simple:
            # Generate simple key without storage
            key = generate_simple_key(args.prefix, args.length)
            print(f"Generated API key: {key}")
            print(f"Key hash: {hashlib.sha256(key.encode()).hexdigest()[:16]}")
        else:
            # Generate and store key
            manager = APIKeyManager()
            result = manager.generate_api_key(args.name, args.prefix, args.length)
            
            print("✅ API Key Generated Successfully!")
            print(f"API Key: {result['api_key']}")
            print(f"Name: {result['name']}")
            print(f"Hash: {result['hash']}")
            print(f"Created: {result['created']}")
            print(f"\n⚠️  Store this key securely - it won't be shown again!")
    
    elif args.command == "list":
        manager = APIKeyManager()
        keys = manager.list_keys()
        
        if not keys:
            print("No API keys found.")
        else:
            print(f"Found {len(keys)} API key(s):")
            print("-" * 80)
            for key in keys:
                status = "🟢 Active" if key["active"] else "🔴 Revoked"
                print(f"Name: {key['name']}")
                print(f"Hash: {key['hash']}")
                print(f"Preview: {key['key_preview']}")
                print(f"Status: {status}")
                print(f"Created: {key['created']}")
                print(f"Usage: {key['usage_count']} requests")
                if key["last_used"]:
                    print(f"Last used: {key['last_used']}")
                print("-" * 80)
    
    elif args.command == "revoke":
        manager = APIKeyManager()
        if manager.revoke_key(args.key_or_hash):
            print(f"✅ API key revoked successfully: {args.key_or_hash}")
        else:
            print(f"❌ API key not found: {args.key_or_hash}")
    
    elif args.command == "validate":
        manager = APIKeyManager()
        if manager.validate_key(args.api_key):
            print(f"✅ API key is valid and active")
        else:
            print(f"❌ API key is invalid or revoked")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

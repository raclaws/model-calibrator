"""
Community registry for pre-calibrated model profiles.

Pre-populated with top 30 local models. Supports trust levels
(verified, community, unverified) and staleness tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from model_calibrator.schema import CapabilityManifest


class Registry:
    """
    Community registry of pre-calibrated model profiles.
    
    Profiles are fetched from:
    1. Local cache (~/.model-calibrator/profiles/)
    2. Remote registry (GitHub raw or CDN)
    """
    
    # Remote registry URL (GitHub raw for now, CDN later)
    REGISTRY_URL = "https://raw.githubusercontent.com/model-calibrator/registry/main"
    
    # Local cache directory
    CACHE_DIR = Path.home() / ".model-calibrator" / "profiles"
    
    # Trust levels in order of preference
    TRUST_LEVELS = ["verified", "community", "unverified"]
    
    @classmethod
    def get(
        cls,
        model_id: str,
        min_trust: str = "community",
        use_cache: bool = True,
        refresh: bool = False,
    ) -> CapabilityManifest | None:
        """
        Get manifest for a model from the registry.
        
        Args:
            model_id: Canonical model ID (e.g., "ollama/llama3.2:8b-q4_K_M")
            min_trust: Minimum trust level ("verified", "community", "unverified")
            use_cache: Whether to use local cache
            refresh: Force refresh from remote
        
        Returns:
            CapabilityManifest or None if not found
        """
        # Normalize model_id to filesystem-safe path
        safe_id = cls._normalize_id(model_id)
        
        # Try local cache first (unless refresh requested)
        if use_cache and not refresh:
            manifest = cls._load_from_cache(safe_id)
            if manifest and cls._meets_trust(manifest, min_trust):
                return manifest
        
        # Try remote registry
        manifest = cls._fetch_from_remote(safe_id)
        if manifest:
            if cls._meets_trust(manifest, min_trust):
                # Cache for future use
                cls._save_to_cache(safe_id, manifest)
                return manifest
        
        return None
    
    @classmethod
    def list_models(cls, provider: str | None = None) -> list[str]:
        """
        List all models in the registry.
        
        Args:
            provider: Optional filter by provider (e.g., "ollama", "openai")
        
        Returns:
            List of model IDs
        """
        # Try to fetch index from remote
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{cls.REGISTRY_URL}/index.json")
                if resp.status_code == 200:
                    index = resp.json()
                    models = list(index.get("models", {}).keys())
                    if provider:
                        models = [m for m in models if m.startswith(f"{provider}/")]
                    return sorted(models)
        except Exception:
            pass
        
        # Fall back to local cache
        if cls.CACHE_DIR.exists():
            models = []
            for path in cls.CACHE_DIR.rglob("*.yaml"):
                model_id = str(path.relative_to(cls.CACHE_DIR)).replace("/", "/").replace(".yaml", "")
                if provider is None or model_id.startswith(f"{provider}/"):
                    models.append(model_id)
            return sorted(models)
        
        return []
    
    @classmethod
    def save(
        cls,
        manifest: CapabilityManifest,
        trust_level: str = "unverified",
    ) -> Path:
        """
        Save a manifest to local cache.
        
        Args:
            manifest: The manifest to save
            trust_level: Trust level for this profile
        
        Returns:
            Path to saved file
        """
        safe_id = cls._normalize_id(manifest.model_id)
        return cls._save_to_cache(safe_id, manifest, trust_level)
    
    @classmethod
    def _normalize_id(cls, model_id: str) -> str:
        """Normalize model ID to filesystem-safe format."""
        # Replace colons and other unsafe chars
        return model_id.replace(":", "-").replace(" ", "_")
    
    @classmethod
    def _load_from_cache(cls, safe_id: str) -> CapabilityManifest | None:
        """Load manifest from local cache."""
        cache_path = cls.CACHE_DIR / f"{safe_id}.yaml"
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path) as f:
                data = yaml.safe_load(f)
            return CapabilityManifest.model_validate(data)
        except Exception:
            return None
    
    @classmethod
    def _save_to_cache(
        cls,
        safe_id: str,
        manifest: CapabilityManifest,
        trust_level: str = "unverified",
    ) -> Path:
        """Save manifest to local cache."""
        cache_path = cls.CACHE_DIR / f"{safe_id}.yaml"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and add trust metadata
        data = manifest.model_dump(mode="json")
        data["_trust_level"] = trust_level
        
        with open(cache_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        return cache_path
    
    @classmethod
    def _fetch_from_remote(cls, safe_id: str) -> CapabilityManifest | None:
        """Fetch manifest from remote registry."""
        url = f"{cls.REGISTRY_URL}/profiles/{safe_id}.yaml"
        
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = yaml.safe_load(resp.text)
                    return CapabilityManifest.model_validate(data)
        except Exception:
            pass
        
        return None
    
    @classmethod
    def _meets_trust(cls, manifest: CapabilityManifest, min_trust: str) -> bool:
        """Check if manifest meets minimum trust level."""
        # Get trust level from manifest (stored in _trust_level during cache)
        # For now, assume all remote profiles are at least "community"
        manifest_trust = getattr(manifest, "_trust_level", "community")
        
        try:
            min_idx = cls.TRUST_LEVELS.index(min_trust)
            manifest_idx = cls.TRUST_LEVELS.index(manifest_trust)
            return manifest_idx <= min_idx  # Lower index = higher trust
        except ValueError:
            return True  # Unknown trust level, allow
    
    @classmethod
    def clear_cache(cls) -> int:
        """Clear local cache. Returns number of files removed."""
        if not cls.CACHE_DIR.exists():
            return 0
        
        count = 0
        for path in cls.CACHE_DIR.rglob("*.yaml"):
            path.unlink()
            count += 1
        
        return count

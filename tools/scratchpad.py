"""
Scratchpad Tool

This module provides tool functions for the Scratchpad service,
enabling agents to maintain context, track findings, and manage evidence during
progressive reduction and discovery tasks.

The module supports three main operations:
- add_finding: Record discoveries with supporting evidence
- get_findings: Retrieve and summarize findings
- get_evidence: Access cached evidence

Configuration via environment variables:
- SCRATCHPAD_MAX_FINDINGS: Maximum findings to store (default: 100)
- SCRATCHPAD_MAX_EVIDENCE: Maximum evidence items to cache (default: 1000)
- SCRATCHPAD_EVIDENCE_TTL: Evidence cache TTL in seconds (default: 3600)
"""

import os
import logging
from typing import Dict, Any, Optional, List

from interfaces.slack.services.scratchpad import Scratchpad

logger = logging.getLogger(__name__)

# Global scratchpad instance
_scratchpad: Optional[Scratchpad] = None

def get_scratchpad() -> Scratchpad:
    """Get or create the global scratchpad instance"""
    global _scratchpad
    if _scratchpad is None:
        max_findings = int(os.getenv('SCRATCHPAD_MAX_FINDINGS', '100'))
        max_evidence = int(os.getenv('SCRATCHPAD_MAX_EVIDENCE', '1000'))
        _scratchpad = Scratchpad(
            max_findings=max_findings,
            max_evidence=max_evidence
        )
    return _scratchpad

class ScratchpadClient:
    """
    Client for Scratchpad operations
    """
    
    def __init__(self):
        self.scratchpad = get_scratchpad()
        
    async def add_finding(self,
                         summary: str,
                         evidence_ids: List[str],
                         source: str = "slack") -> Dict[str, Any]:
        """
        Add a finding to the scratchpad
        
        Args:
            summary: Finding summary
            evidence_ids: Supporting evidence IDs
            source: Source system (default: slack)
            
        Returns:
            Dictionary with finding ID
        """
        try:
            finding_id = await self.scratchpad.add_finding(
                summary=summary,
                evidence_ids=evidence_ids,
                source=source
            )
            
            return {
                "success": True,
                "finding_id": finding_id
            }
            
        except Exception as e:
            logger.error(f"Add finding error: {e}")
            return {
                "success": False,
                "message": f"Failed to add finding: {str(e)}"
            }
            
    async def get_findings(self,
                          query: Optional[str] = None,
                          source: Optional[str] = None,
                          max_findings: int = 5) -> Dict[str, Any]:
        """
        Get findings from the scratchpad
        
        Args:
            query: Optional search query
            source: Optional source system filter
            max_findings: Maximum findings to return
            
        Returns:
            Dictionary with findings and evidence
        """
        try:
            summary = await self.scratchpad.summarize_findings(
                max_findings=max_findings,
                query=query
            )
            
            return {
                "success": True,
                "findings": summary
            }
            
        except Exception as e:
            logger.error(f"Get findings error: {e}")
            return {
                "success": False,
                "message": f"Failed to get findings: {str(e)}"
            }
            
    async def get_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """
        Get evidence from the scratchpad
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            Dictionary with evidence content and metadata
        """
        try:
            evidence = await self.scratchpad.get_evidence(evidence_id)
            
            if evidence:
                return {
                    "success": True,
                    "evidence": {
                        "content": evidence.content,
                        "metadata": evidence.metadata,
                        "fetch_time": evidence.fetch_time
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Evidence not found or expired"
                }
                
        except Exception as e:
            logger.error(f"Get evidence error: {e}")
            return {
                "success": False,
                "message": f"Failed to get evidence: {str(e)}"
            }

# Global client instance
_scratchpad_client: Optional[ScratchpadClient] = None

def get_scratchpad_client() -> ScratchpadClient:
    """Get or create the global Scratchpad client instance"""
    global _scratchpad_client
    if _scratchpad_client is None:
        _scratchpad_client = ScratchpadClient()
    return _scratchpad_client

# Tool function wrappers for integration with the execute system
async def add_finding(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for adding a finding to the scratchpad
    """
    try:
        client = get_scratchpad_client()
        
        summary = input_data.get("summary")
        if not summary:
            return {
                "success": False,
                "message": "Finding summary is required"
            }
            
        evidence_ids = input_data.get("evidence_ids", [])
        if not evidence_ids:
            return {
                "success": False,
                "message": "At least one evidence ID is required"
            }
            
        return await client.add_finding(
            summary=summary,
            evidence_ids=evidence_ids,
            source=input_data.get("source", "slack")
        )
        
    except Exception as e:
        logger.error(f"Add finding error: {e}")
        return {
            "success": False,
            "message": f"Failed to add finding: {str(e)}"
        }

async def get_findings(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting findings from the scratchpad
    """
    try:
        client = get_scratchpad_client()
        
        return await client.get_findings(
            query=input_data.get("query"),
            source=input_data.get("source"),
            max_findings=input_data.get("max_findings", 5)
        )
        
    except Exception as e:
        logger.error(f"Get findings error: {e}")
        return {
            "success": False,
            "message": f"Failed to get findings: {str(e)}"
        }

async def get_evidence(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool function for getting evidence from the scratchpad
    """
    try:
        client = get_scratchpad_client()
        
        evidence_id = input_data.get("evidence_id")
        if not evidence_id:
            return {
                "success": False,
                "message": "Evidence ID is required"
            }
            
        return await client.get_evidence(evidence_id)
        
    except Exception as e:
        logger.error(f"Get evidence error: {e}")
        return {
            "success": False,
            "message": f"Failed to get evidence: {str(e)}"
        }

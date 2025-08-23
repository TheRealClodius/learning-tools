"""
Scratchpad for Progressive Reduction

Maintains context and findings throughout the reduction process.
- Tracks search context
- Stores findings
- Caches evidence
- Manages citations
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SearchContext:
    """Search context information"""
    query: str
    filters: Dict[str, Any]
    strategy: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class Finding:
    """A single finding with evidence"""
    id: str
    summary: str
    evidence_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "slack"  # For future multi-source support

@dataclass
class Evidence:
    """Cached evidence item"""
    content: str
    metadata: Dict[str, Any]
    fetch_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl: int = 3600  # 1 hour default TTL

class Scratchpad:
    """
    Maintains context and findings during progressive reduction
    """
    
    def __init__(self, max_findings: int = 100, max_evidence: int = 1000):
        self.max_findings = max_findings
        self.max_evidence = max_evidence
        
        self.contexts: List[SearchContext] = []
        self.findings: List[Finding] = []
        self.evidence_cache: Dict[str, Evidence] = {}
        
    async def add_context(self,
                         query: str,
                         filters: Optional[Dict[str, Any]] = None,
                         strategy: str = "auto") -> None:
        """
        Add new search context
        
        Args:
            query: Search query
            filters: Applied filters
            strategy: Search strategy used
        """
        context = SearchContext(
            query=query,
            filters=filters or {},
            strategy=strategy
        )
        self.contexts.append(context)
        
        # Keep only recent contexts
        if len(self.contexts) > 10:  # Keep last 10 contexts
            self.contexts = self.contexts[-10:]
            
    async def add_finding(self,
                         summary: str,
                         evidence_ids: List[str],
                         source: str = "slack") -> str:
        """
        Add a new finding
        
        Args:
            summary: Finding summary
            evidence_ids: List of evidence IDs supporting this finding
            source: Source system (e.g., "slack")
            
        Returns:
            Finding ID
        """
        finding_id = f"f_{int(time.time())}_{len(self.findings)}"
        
        finding = Finding(
            id=finding_id,
            summary=summary,
            evidence_ids=evidence_ids,
            source=source
        )
        
        self.findings.append(finding)
        
        # Maintain size limit
        if len(self.findings) > self.max_findings:
            self.findings = self.findings[-self.max_findings:]
            
        return finding_id
        
    async def cache_evidence(self,
                           evidence_id: str,
                           content: str,
                           metadata: Dict[str, Any],
                           ttl: Optional[int] = None) -> None:
        """
        Cache evidence for future reference
        
        Args:
            evidence_id: Unique evidence identifier
            content: Evidence content
            metadata: Evidence metadata
            ttl: Time-to-live in seconds (optional)
        """
        evidence = Evidence(
            content=content,
            metadata=metadata,
            ttl=ttl or 3600
        )
        
        self.evidence_cache[evidence_id] = evidence
        
        # Maintain cache size
        if len(self.evidence_cache) > self.max_evidence:
            # Remove oldest items
            sorted_items = sorted(
                self.evidence_cache.items(),
                key=lambda x: x[1].fetch_time
            )
            self.evidence_cache = dict(sorted_items[-self.max_evidence:])
            
    async def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """
        Get cached evidence
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            Evidence if found and not expired, None otherwise
        """
        evidence = self.evidence_cache.get(evidence_id)
        if not evidence:
            return None
            
        # Check TTL
        fetch_time = datetime.fromisoformat(evidence.fetch_time)
        now = datetime.now(timezone.utc)
        
        if (now - fetch_time).total_seconds() > evidence.ttl:
            # Expired
            del self.evidence_cache[evidence_id]
            return None
            
        return evidence
        
    async def get_findings_for_evidence(self, evidence_id: str) -> List[Finding]:
        """
        Get all findings that reference a piece of evidence
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            List of findings that cite this evidence
        """
        return [
            finding for finding in self.findings
            if evidence_id in finding.evidence_ids
        ]
        
    async def get_context_findings(self,
                                 query: Optional[str] = None,
                                 source: Optional[str] = None) -> List[Finding]:
        """
        Get findings relevant to current context
        
        Args:
            query: Optional query to filter findings
            source: Optional source system filter
            
        Returns:
            List of relevant findings
        """
        findings = self.findings
        
        if source:
            findings = [f for f in findings if f.source == source]
            
        if query:
            # Simple term matching for now
            query_terms = query.lower().split()
            findings = [
                f for f in findings
                if any(term in f.summary.lower() for term in query_terms)
            ]
            
        return findings
        
    def get_current_context(self) -> Optional[SearchContext]:
        """Get most recent search context"""
        return self.contexts[-1] if self.contexts else None
        
    async def summarize_findings(self,
                               max_findings: int = 5,
                               query: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize recent/relevant findings
        
        Args:
            max_findings: Maximum findings to include
            query: Optional query to filter findings
            
        Returns:
            Summary of findings with evidence
        """
        findings = await self.get_context_findings(query)
        findings = findings[-max_findings:]  # Get most recent
        
        # Collect unique evidence
        evidence_ids = {
            eid for f in findings
            for eid in f.evidence_ids
        }
        
        evidence = {}
        for eid in evidence_ids:
            if cached := await self.get_evidence(eid):
                evidence[eid] = {
                    "content": cached.content,
                    "metadata": cached.metadata
                }
                
        return {
            "findings": [
                {
                    "id": f.id,
                    "summary": f.summary,
                    "timestamp": f.timestamp,
                    "evidence_count": len(f.evidence_ids)
                }
                for f in findings
            ],
            "evidence": evidence,
            "context": self.get_current_context().__dict__ if self.get_current_context() else None
        }

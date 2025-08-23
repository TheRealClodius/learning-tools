"""
Progress Tracker for Progressive Research

Tracks:
- Research goal and progress
- Finding history
- Evidence saturation
- Facet-based decisions
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Finding:
    """A research finding with evidence"""
    summary: str
    evidence_ids: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    iteration: int = 0

@dataclass
class IterationState:
    """State for a single research iteration"""
    findings: List[Finding]
    facets: Dict[str, Any]
    suggested_areas: List[Dict[str, str]]
    token_usage: Dict[str, int]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ProgressTracker:
    """
    Tracks research progress and guides exploration
    """
    
    def __init__(self, goal: Optional[str] = None):
        self.goal = goal or "Explore and understand"
        self.iterations: List[IterationState] = []
        self.findings_history: List[Finding] = []
        self.current_iteration = 0
        
    def start_research(self, goal: str) -> None:
        """Start a new research task"""
        self.goal = goal
        self.iterations = []
        self.findings_history = []
        self.current_iteration = 0
        
    def start_iteration(self) -> int:
        """Start a new research iteration"""
        self.current_iteration += 1
        return self.current_iteration
        
    def add_finding(self, summary: str, evidence_ids: List[str]) -> Finding:
        """Add a new finding"""
        finding = Finding(
            summary=summary,
            evidence_ids=evidence_ids,
            iteration=self.current_iteration
        )
        self.findings_history.append(finding)
        return finding
        
    def record_iteration_state(self,
                             findings: List[Finding],
                             facets: Dict[str, Any],
                             suggested_areas: List[Dict[str, str]],
                             token_usage: Dict[str, int]) -> None:
        """Record state for current iteration"""
        state = IterationState(
            findings=findings,
            facets=facets,
            suggested_areas=suggested_areas,
            token_usage=token_usage
        )
        self.iterations.append(state)
        
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of current progress"""
        if not self.iterations:
            return {
                "iteration": 0,
                "findings_count": 0,
                "status": "Not started"
            }
            
        latest = self.iterations[-1]
        recent_findings = len([
            f for f in self.findings_history
            if f.iteration > max(0, self.current_iteration - 3)
        ])
        
        return {
            "iteration": self.current_iteration,
            "total_findings": len(self.findings_history),
            "recent_findings": recent_findings,
            "latest_areas": latest.suggested_areas,
            "token_usage": latest.token_usage
        }
        
    def check_saturation(self) -> Dict[str, Any]:
        """Check for evidence saturation"""
        if len(self.iterations) < 3:
            return {
                "is_saturated": False,
                "reason": "Insufficient iterations"
            }
            
        # Check finding rate
        recent_findings = [
            len([f for f in self.findings_history if f.iteration == i])
            for i in range(max(1, self.current_iteration - 2),
                         self.current_iteration + 1)
        ]
        
        # Check if findings are decreasing
        if all(recent_findings[i] > recent_findings[i+1] 
               for i in range(len(recent_findings)-1)):
            return {
                "is_saturated": True,
                "reason": "Diminishing new findings",
                "finding_rates": recent_findings
            }
            
        # Check evidence overlap
        recent_evidence = set()
        overlap_count = 0
        
        for finding in reversed(self.findings_history[-5:]):
            current_evidence = set(finding.evidence_ids)
            overlap = len(current_evidence & recent_evidence)
            if overlap > len(current_evidence) * 0.7:  # 70% overlap threshold
                overlap_count += 1
            recent_evidence.update(current_evidence)
            
        if overlap_count >= 3:  # High evidence reuse
            return {
                "is_saturated": True,
                "reason": "High evidence reuse",
                "overlap_rate": overlap_count / 5
            }
            
        return {
            "is_saturated": False,
            "reason": "Still finding new evidence"
        }
        
    def should_continue(self) -> Dict[str, Any]:
        """Check if research should continue"""
        # Check iteration limit
        if self.current_iteration >= 10:
            return {
                "continue": False,
                "reason": "Max iterations reached"
            }
            
        # Check saturation
        saturation = self.check_saturation()
        if saturation["is_saturated"]:
            return {
                "continue": False,
                "reason": f"Evidence saturated: {saturation['reason']}"
            }
            
        # Check progress
        progress = self.get_progress_summary()
        if progress["recent_findings"] == 0:
            return {
                "continue": False,
                "reason": "No recent findings"
            }
            
        return {
            "continue": True,
            "reason": "Active progress"
        }
        
    def get_facet_guidance(self) -> List[Dict[str, str]]:
        """Get guidance for facet exploration"""
        if not self.iterations:
            return []
            
        latest = self.iterations[-1]
        previous = self.iterations[-2] if len(self.iterations) > 1 else None
        
        guidance = []
        
        # Check channel activity changes
        if previous:
            for channel, count in latest.facets.get("channels", {}).items():
                prev_count = previous.facets.get("channels", {}).get(channel, 0)
                if count > prev_count * 1.5:  # 50% increase
                    guidance.append({
                        "area": channel,
                        "reason": "Increased activity",
                        "priority": "high"
                    })
                    
        # Check date patterns
        dates = latest.facets.get("dates", {})
        if dates:
            recent_count = sum(
                count for date, count in dates.items()
                if (datetime.now() - datetime.fromisoformat(date)).days <= 7
            )
            if recent_count > sum(dates.values()) * 0.4:  # 40% recent
                guidance.append({
                    "area": "recent_messages",
                    "reason": "High recent activity",
                    "priority": "medium"
                })
                
        # Check author patterns
        authors = latest.facets.get("authors", {})
        if authors:
            top_authors = sorted(
                authors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            guidance.append({
                "area": "key_participants",
                "reason": f"Focus on top contributors: {', '.join(a[0] for a in top_authors)}",
                "priority": "medium"
            })
            
        return guidance
        
    def get_iteration_context(self) -> Dict[str, Any]:
        """Get full context for current iteration"""
        return {
            "goal": self.goal,
            "iteration": self.current_iteration,
            "progress": self.get_progress_summary(),
            "findings_history": [
                {
                    "summary": f.summary,
                    "evidence_ids": f.evidence_ids,
                    "iteration": f.iteration
                }
                for f in self.findings_history
            ],
            "saturation": self.check_saturation(),
            "facet_guidance": self.get_facet_guidance(),
            "should_continue": self.should_continue()
        }
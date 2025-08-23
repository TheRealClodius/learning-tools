"""
Monitoring System for Guardrail Activations
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class GuardRailEvent:
    """Represents a guardrail activation event"""
    rail_type: str
    timestamp: float
    details: Dict[str, Any]
    run_id: str
    result_set_id: str

@dataclass
class GuardRailMetrics:
    """Tracks guardrail activation metrics"""
    explosion_count: int = 0
    dedup_count: int = 0
    compression_count: int = 0
    collision_count: int = 0
    token_limit_count: int = 0
    
    # Detailed metrics
    explosion_details: List[GuardRailEvent] = field(default_factory=list)
    dedup_details: List[GuardRailEvent] = field(default_factory=list)
    compression_details: List[GuardRailEvent] = field(default_factory=list)
    collision_details: List[GuardRailEvent] = field(default_factory=list)
    token_limit_details: List[GuardRailEvent] = field(default_factory=list)

class MonitoringSystem:
    """
    Monitoring system for tracking and analyzing guardrail activations
    """
    
    def __init__(self):
        self.metrics = GuardRailMetrics()
        self._time_window = 3600  # 1 hour window for rate tracking
        self._activation_times: Dict[str, List[float]] = defaultdict(list)
        
    def record_activation(self,
                        rail_type: str,
                        details: Dict[str, Any],
                        run_id: str,
                        result_set_id: str) -> None:
        """
        Record a guardrail activation
        
        Args:
            rail_type: Type of guardrail activated
            details: Additional context about the activation
            run_id: ID of the run that triggered the activation
            result_set_id: ID of the result set being processed
        """
        now = time.time()
        event = GuardRailEvent(
            rail_type=rail_type,
            timestamp=now,
            details=details,
            run_id=run_id,
            result_set_id=result_set_id
        )
        
        # Update counts
        if rail_type == "explosion":
            self.metrics.explosion_count += 1
            self.metrics.explosion_details.append(event)
        elif rail_type == "dedup":
            self.metrics.dedup_count += 1
            self.metrics.dedup_details.append(event)
        elif rail_type == "compression":
            self.metrics.compression_count += 1
            self.metrics.compression_details.append(event)
        elif rail_type == "collision":
            self.metrics.collision_count += 1
            self.metrics.collision_details.append(event)
        elif rail_type == "token_limit":
            self.metrics.token_limit_count += 1
            self.metrics.token_limit_details.append(event)
            
        # Record timestamp for rate tracking
        self._activation_times[rail_type].append(now)
        self._cleanup_old_activations()
        
        # Log the event
        logger.info(
            f"Guardrail activation: {rail_type} "
            f"(run_id={run_id}, result_set_id={result_set_id})"
        )
        
    def get_activation_rate(self, rail_type: Optional[str] = None) -> Dict[str, float]:
        """
        Get activation rate (events/hour) for specified rail type or all types
        
        Args:
            rail_type: Optional specific rail type to get rate for
            
        Returns:
            Dict mapping rail types to their activation rates
        """
        now = time.time()
        rates = {}
        
        def calc_rate(times: List[float]) -> float:
            # Count events in last time window
            recent = sum(1 for t in times if now - t <= self._time_window)
            return recent / (self._time_window / 3600)  # Convert to per hour
        
        if rail_type:
            rates[rail_type] = calc_rate(self._activation_times[rail_type])
        else:
            for rtype, times in self._activation_times.items():
                rates[rtype] = calc_rate(times)
                
        return rates
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "counts": {
                "explosion": self.metrics.explosion_count,
                "dedup": self.metrics.dedup_count,
                "compression": self.metrics.compression_count,
                "collision": self.metrics.collision_count,
                "token_limit": self.metrics.token_limit_count
            },
            "rates": self.get_activation_rate(),
            "recent_events": self._get_recent_events()
        }
        
    def _get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent activation events"""
        # Combine all events
        all_events = (
            [("explosion", e) for e in self.metrics.explosion_details] +
            [("dedup", e) for e in self.metrics.dedup_details] +
            [("compression", e) for e in self.metrics.compression_details] +
            [("collision", e) for e in self.metrics.collision_details] +
            [("token_limit", e) for e in self.metrics.token_limit_details]
        )
        
        # Sort by timestamp (newest first) and take latest
        recent = sorted(
            all_events,
            key=lambda x: x[1].timestamp,
            reverse=True
        )[:limit]
        
        # Format for output
        return [
            {
                "type": type_,
                "timestamp": datetime.fromtimestamp(
                    event.timestamp,
                    tz=timezone.utc
                ).isoformat(),
                "run_id": event.run_id,
                "result_set_id": event.result_set_id,
                "details": event.details
            }
            for type_, event in recent
        ]
        
    def _cleanup_old_activations(self) -> None:
        """Remove activation records older than the time window"""
        now = time.time()
        cutoff = now - self._time_window
        
        for rail_type in self._activation_times:
            self._activation_times[rail_type] = [
                t for t in self._activation_times[rail_type]
                if t > cutoff
            ]
            
    def export_metrics(self, format: str = "json") -> str:
        """
        Export current metrics in specified format
        
        Args:
            format: Output format ("json" or "text")
            
        Returns:
            Formatted metrics string
        """
        metrics = self.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
            
        # Text format
        lines = [
            "Guardrail Activation Metrics:",
            "=========================="
        ]
        
        # Counts
        lines.append("\nCounts:")
        for rail_type, count in metrics["counts"].items():
            lines.append(f"  {rail_type}: {count}")
            
        # Rates
        lines.append("\nRates (events/hour):")
        for rail_type, rate in metrics["rates"].items():
            lines.append(f"  {rail_type}: {rate:.2f}")
            
        # Recent events
        lines.append("\nRecent Events:")
        for event in metrics["recent_events"]:
            lines.append(
                f"  {event['timestamp']} - {event['type']} "
                f"(run_id={event['run_id']})"
            )
            
        return "\n".join(lines)
        
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics = GuardRailMetrics()
        self._activation_times.clear()

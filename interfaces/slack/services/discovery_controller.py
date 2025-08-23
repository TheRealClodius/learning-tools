"""
Discovery Controller

Manages the progressive discovery process:
- Loop control and iteration management
- Progress assessment and tracking
- Evidence saturation detection
- Task completion monitoring
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .progress_tracker import ProgressTracker
from .progressive_reducer import ProgressiveReducer
from .summarizer import ProgressiveSummarizer
from .token_manager import TokenManager
from .guard_rails import GuardRails

logger = logging.getLogger(__name__)

@dataclass
class DiscoveryTask:
    """A research task to complete"""
    description: str
    criteria: List[str]
    priority: str = "medium"
    completed: bool = False
    evidence_ids: List[str] = field(default_factory=list)

@dataclass
class DiscoveryRisk:
    """A potential research risk"""
    description: str
    mitigation: str
    status: str = "open"
    priority: str = "medium"

@dataclass
class DiscoveryMetrics:
    """Metrics for discovery progress"""
    new_findings_per_iteration: List[int] = field(default_factory=list)
    information_gain: float = 0.0
    coverage_estimate: float = 0.0
    task_completion: float = 0.0
    risk_mitigation: float = 0.0

class DiscoveryController:
    """
    Controls the progressive discovery process
    """
    
    def __init__(self,
                 goal: str,
                 tasks: Optional[List[Dict[str, Any]]] = None,
                 risks: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize controller
        
        Args:
            goal: Research goal
            tasks: List of tasks to complete
            risks: List of risks to mitigate
        """
        # Core components
        self.progress_tracker = ProgressTracker(goal)
        self.reducer = ProgressiveReducer()
        self.summarizer = ProgressiveSummarizer(progress_tracker=self.progress_tracker)
        self.token_manager = TokenManager()
        self.guard_rails = GuardRails()
        
        # Discovery state
        self.goal = goal
        self.tasks = [DiscoveryTask(**t) for t in (tasks or [])]
        self.risks = [DiscoveryRisk(**r) for r in (risks or [])]
        self.metrics = DiscoveryMetrics()
        
        # Loop control
        self.max_iterations = 10
        self.min_information_gain = 0.1
        self.min_task_completion = 0.8
        
    async def run_discovery(self, initial_query: str) -> Dict[str, Any]:
        """
        Run the progressive discovery process
        
        Args:
            initial_query: Starting search query
            
        Returns:
            Final discovery results
        """
        current_query = initial_query
        iteration = 0
        
        while await self._should_continue(iteration):
            iteration += 1
            logger.info(f"Starting discovery iteration {iteration}")
            
            # Run search and reduction
            results = await self._run_iteration(current_query, iteration)
            
            # Update metrics
            self._update_metrics(results)
            
            # Check completion and saturation
            if self._is_complete() or self._is_saturated(results):
                break
                
            # Update query based on guidance
            current_query = self._refine_query(results)
            
        return await self._create_final_results()
        
    async def _run_iteration(self,
                           query: str,
                           iteration: int) -> Dict[str, Any]:
        """Run a single discovery iteration"""
        # Start iteration in progress tracker
        self.progress_tracker.start_iteration()
        
        # Get search results and reduce
        results = await self.reducer.reduce_results(
            query=query,
            goal=self.goal,
            iteration=iteration
        )
        
        # Update tasks and risks based on findings
        self._process_findings(results)
        
        # Record iteration state
        self.progress_tracker.record_iteration_state(
            findings=results.get("findings", []),
            facets=results.get("facets", {}),
            suggested_areas=results.get("suggested_areas", []),
            token_usage=self.token_manager.get_budget_status()
        )
        
        return results
        
    def _process_findings(self, results: Dict[str, Any]) -> None:
        """Process findings to update tasks and risks"""
        findings = results.get("findings", [])
        
        # Update task completion
        for task in self.tasks:
            if not task.completed:
                # Check if finding satisfies task criteria
                for finding in findings:
                    if all(
                        criterion.lower() in finding["summary"].lower()
                        for criterion in task.criteria
                    ):
                        task.completed = True
                        task.evidence_ids.extend(finding["evidence_ids"])
                        
        # Update risk status
        for risk in self.risks:
            if risk.status == "open":
                # Check if finding addresses risk
                for finding in findings:
                    if risk.mitigation.lower() in finding["summary"].lower():
                        risk.status = "mitigated"
                        
    def _update_metrics(self, results: Dict[str, Any]) -> None:
        """Update discovery metrics"""
        # Track new findings
        new_findings = len(results.get("findings", []))
        self.metrics.new_findings_per_iteration.append(new_findings)
        
        # Calculate information gain
        if len(self.metrics.new_findings_per_iteration) > 1:
            prev_findings = self.metrics.new_findings_per_iteration[-2]
            self.metrics.information_gain = new_findings / (prev_findings + 1)
            
        # Update completion metrics
        completed_tasks = len([t for t in self.tasks if t.completed])
        self.metrics.task_completion = completed_tasks / len(self.tasks) if self.tasks else 1.0
        
        mitigated_risks = len([r for r in self.risks if r.status == "mitigated"])
        self.metrics.risk_mitigation = mitigated_risks / len(self.risks) if self.risks else 1.0
        
        # Estimate coverage
        self.metrics.coverage_estimate = (
            self.metrics.task_completion * 0.6 +  # Weight task completion higher
            self.metrics.risk_mitigation * 0.4
        )
        
    async def _should_continue(self, iteration: int) -> bool:
        """Check if discovery should continue"""
        if iteration >= self.max_iterations:
            logger.info("Max iterations reached")
            return False
            
        if iteration > 0:  # Skip checks for first iteration
            if self.metrics.information_gain < self.min_information_gain:
                logger.info("Information gain below threshold")
                return False
                
            if self.metrics.task_completion > self.min_task_completion:
                logger.info("Sufficient task completion achieved")
                return False
                
            # Check progress tracker's saturation assessment
            saturation = self.progress_tracker.check_saturation()
            if saturation["is_saturated"]:
                logger.info(f"Evidence saturated: {saturation['reason']}")
                return False
                
        return True
        
    def _is_complete(self) -> bool:
        """Check if discovery is complete"""
        return (
            all(task.completed for task in self.tasks) and
            all(risk.status == "mitigated" for risk in self.risks)
        )
        
    def _is_saturated(self, results: Dict[str, Any]) -> bool:
        """Check if evidence is saturated"""
        # Get saturation assessment
        saturation = self.progress_tracker.check_saturation()
        
        # Check token usage
        token_status = self.token_manager.get_budget_status()
        budget_saturated = all(
            usage["current"] > usage["limit"] * 0.9  # 90% of limits
            for usage in token_status.values()
        )
        
        return saturation["is_saturated"] or budget_saturated
        
    def _refine_query(self, results: Dict[str, Any]) -> str:
        """Refine query based on results"""
        # Get facet guidance
        guidance = self.progress_tracker.get_facet_guidance()
        
        # Find highest priority area
        priority_area = None
        for area in guidance:
            if area["priority"] == "high":
                priority_area = area
                break
                
        if priority_area:
            return f"{self.goal} {priority_area['area']}"
            
        return self.goal  # Fallback to original goal
        
    async def _create_final_results(self) -> Dict[str, Any]:
        """Create final discovery results"""
        return {
            "goal": self.goal,
            "metrics": self.metrics.__dict__,
            "tasks": [
                {
                    "description": task.description,
                    "completed": task.completed,
                    "evidence_ids": task.evidence_ids
                }
                for task in self.tasks
            ],
            "risks": [
                {
                    "description": risk.description,
                    "status": risk.status,
                    "mitigation": risk.mitigation
                }
                for risk in self.risks
            ],
            "findings": self.progress_tracker.findings_history,
            "token_usage": self.token_manager.get_budget_status(),
            "saturation": self.progress_tracker.check_saturation()
        }

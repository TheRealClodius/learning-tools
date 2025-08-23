"""
Token Manager for Progressive Reduction

Implements token policy enforcement for research tools:
- Cluster summaries: ≤200 tokens, max 10 clusters → ≤2,000 tokens total
- Global brief: ≤300-400 tokens
- Scratchpad: rolling ≤500 tokens
- Total evidence per step: ~3,000 tokens
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class TokenLimits:
    """Token limits for different components"""
    CLUSTER_SUMMARY: int = 200
    MAX_CLUSTERS: int = 10
    TOTAL_CLUSTERS: int = 2000
    GLOBAL_BRIEF: int = 400
    SCRATCHPAD: int = 500
    TOTAL_EVIDENCE: int = 3000

class TokenBudget:
    """Tracks token usage for a reduction operation"""
    def __init__(self):
        self.cluster_tokens: Dict[str, int] = {}  # cluster_id -> tokens
        self.global_brief_tokens: int = 0
        self.scratchpad_tokens: int = 0
        self.evidence_tokens: int = 0
        
    def reset(self):
        """Reset all token counts"""
        self.cluster_tokens.clear()
        self.global_brief_tokens = 0
        self.scratchpad_tokens = 0
        self.evidence_tokens = 0
        
    @property
    def total_cluster_tokens(self) -> int:
        return sum(self.cluster_tokens.values())

class TokenManager:
    """
    Manages token budgets and enforces limits for progressive reduction
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        """Initialize token manager with model-specific settings
        
        Args:
            model: Model name ('gemini-2.5-flash', 'gemini-pro', or 'claude-3-5-sonnet-20241022')
        """
        self.limits = TokenLimits()
        self.budget = TokenBudget()
        self.model = model
        
        # Set encoder based on model
        if model.startswith("claude"):
            # Claude uses cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
            self.encoder = encoding
            # Configure special tokens
            self.special_tokens = {
                "<|endoftext|>",
                "<|fim_prefix|>",
                "<|fim_middle|>",
                "<|fim_suffix|>",
                "<|endofprompt|>"
            }
            # Claude has larger context window
            self.limits.CLUSTER_SUMMARY = 400   # Allow larger summaries
            self.limits.GLOBAL_BRIEF = 800     # Allow larger briefs
            self.limits.TOTAL_EVIDENCE = 6000  # Allow more evidence
            self.limits.MAX_CLUSTERS = 15      # Allow more clusters
            
        elif model.startswith("gemini"):
            # Gemini also uses cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")
            self.encoder = encoding
            # Configure special tokens
            self.special_tokens = {
                "<|endoftext|>",
                "<|fim_prefix|>",
                "<|fim_middle|>",
                "<|fim_suffix|>",
                "<|endofprompt|>"
            }
            
            if model == "gemini-pro":
                self.limits.CLUSTER_SUMMARY = 300  # Allow larger summaries
                self.limits.GLOBAL_BRIEF = 600    # Allow larger briefs
                self.limits.TOTAL_EVIDENCE = 4000 # Allow more evidence
                
        else:
            raise ValueError(f"Unsupported model: {model}")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        if not text:
            return 0
        # Handle special tokens
        for token in self.special_tokens:
            if token in text:
                # Replace special tokens with regular text
                text = text.replace(token, " " * len(token))
        return len(self.encoder.encode(text))
        
    def enforce_cluster_limit(self, summary: str, cluster_id: str) -> str:
        """
        Enforce token limit for a cluster summary
        Returns: Truncated summary if needed
        """
        tokens = self.count_tokens(summary)
        
        if tokens <= self.limits.CLUSTER_SUMMARY:
            self.budget.cluster_tokens[cluster_id] = tokens
            return summary
            
        # Truncate to limit while trying to keep complete sentences
        words = summary.split()
        truncated = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens > self.limits.CLUSTER_SUMMARY:
                break
            truncated.append(word)
            current_tokens += word_tokens
            
        truncated_summary = " ".join(truncated).strip() + "..."
        self.budget.cluster_tokens[cluster_id] = self.count_tokens(truncated_summary)
        
        logger.warning(f"Truncated cluster {cluster_id} summary from {tokens} to {self.budget.cluster_tokens[cluster_id]} tokens")
        return truncated_summary
        
    def enforce_global_brief_limit(self, brief: str) -> str:
        """
        Enforce token limit for global brief
        Returns: Truncated brief if needed
        """
        tokens = self.count_tokens(brief)
        
        if tokens <= self.limits.GLOBAL_BRIEF:
            self.budget.global_brief_tokens = tokens
            return brief
            
        # Truncate to limit while keeping complete sentences
        sentences = brief.split(". ")
        truncated = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence + ". ")
            if current_tokens + sentence_tokens > self.limits.GLOBAL_BRIEF:
                break
            truncated.append(sentence)
            current_tokens += sentence_tokens
            
        truncated_brief = ". ".join(truncated).strip() + "..."
        self.budget.global_brief_tokens = self.count_tokens(truncated_brief)
        
        logger.warning(f"Truncated global brief from {tokens} to {self.budget.global_brief_tokens} tokens")
        return truncated_brief
        
    def enforce_scratchpad_limit(self, content: str) -> str:
        """
        Enforce rolling token limit for scratchpad
        Returns: Truncated content if needed, removing oldest content first
        """
        tokens = self.count_tokens(content)
        
        if tokens <= self.limits.SCRATCHPAD:
            self.budget.scratchpad_tokens = tokens
            return content
            
        # Remove oldest content (assumes content is chronological)
        paragraphs = content.split("\n\n")
        truncated = []
        current_tokens = 0
        
        for paragraph in reversed(paragraphs):  # Start from newest
            paragraph_tokens = self.count_tokens(paragraph + "\n\n")
            if current_tokens + paragraph_tokens > self.limits.SCRATCHPAD:
                break
            truncated.insert(0, paragraph)  # Keep chronological order
            current_tokens += paragraph_tokens
            
        truncated_content = "\n\n".join(truncated).strip()
        self.budget.scratchpad_tokens = self.count_tokens(truncated_content)
        
        logger.warning(f"Truncated scratchpad from {tokens} to {self.budget.scratchpad_tokens} tokens")
        return truncated_content
        
    def select_evidence(self, 
                       evidence_items: List[Dict[str, Any]], 
                       prioritize_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Select evidence items within token budget
        Args:
            evidence_items: List of evidence items with 'id' and 'content'
            prioritize_ids: Optional list of evidence IDs to prioritize
            
        Returns: List of evidence items that fit within budget
        """
        if prioritize_ids:
            # Move prioritized items to front
            evidence_items.sort(
                key=lambda x: prioritize_ids.index(x["id"]) if x["id"] in prioritize_ids else len(prioritize_ids)
            )
            
        selected = []
        current_tokens = 0
        
        for item in evidence_items:
            content = item.get("content", "")
            tokens = self.count_tokens(content)
            
            if current_tokens + tokens > self.limits.TOTAL_EVIDENCE:
                break
                
            selected.append(item)
            current_tokens += tokens
            
        self.budget.evidence_tokens = current_tokens
        
        if len(selected) < len(evidence_items):
            logger.warning(
                f"Selected {len(selected)}/{len(evidence_items)} evidence items "
                f"({current_tokens}/{self.limits.TOTAL_EVIDENCE} tokens)"
            )
            
        return selected
        
    def check_limits(self) -> List[str]:
        """Check for limit violations and return warnings"""
        warnings = []
        
        if self.budget.total_cluster_tokens > self.limits.TOTAL_CLUSTERS:
            warnings.append(
                f"Total cluster tokens ({self.budget.total_cluster_tokens}) "
                f"exceeds limit ({self.limits.TOTAL_CLUSTERS})"
            )
            
        if len(self.budget.cluster_tokens) > self.limits.MAX_CLUSTERS:
            warnings.append(
                f"Number of clusters ({len(self.budget.cluster_tokens)}) "
                f"exceeds limit ({self.limits.MAX_CLUSTERS})"
            )
            
        if self.budget.global_brief_tokens > self.limits.GLOBAL_BRIEF:
            warnings.append(
                f"Global brief tokens ({self.budget.global_brief_tokens}) "
                f"exceeds limit ({self.limits.GLOBAL_BRIEF})"
            )
            
        if self.budget.scratchpad_tokens > self.limits.SCRATCHPAD:
            warnings.append(
                f"Scratchpad tokens ({self.budget.scratchpad_tokens}) "
                f"exceeds limit ({self.limits.SCRATCHPAD})"
            )
            
        if self.budget.evidence_tokens > self.limits.TOTAL_EVIDENCE:
            warnings.append(
                f"Evidence tokens ({self.budget.evidence_tokens}) "
                f"exceeds limit ({self.limits.TOTAL_EVIDENCE})"
            )
            
        return warnings
        
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current token budget status"""
        return {
            "clusters": {
                "current": self.budget.total_cluster_tokens,
                "limit": self.limits.TOTAL_CLUSTERS,
                "individual_clusters": self.budget.cluster_tokens
            },
            "global_brief": {
                "current": self.budget.global_brief_tokens,
                "limit": self.limits.GLOBAL_BRIEF
            },
            "scratchpad": {
                "current": self.budget.scratchpad_tokens,
                "limit": self.limits.SCRATCHPAD
            },
            "evidence": {
                "current": self.budget.evidence_tokens,
                "limit": self.limits.TOTAL_EVIDENCE
            }
        }

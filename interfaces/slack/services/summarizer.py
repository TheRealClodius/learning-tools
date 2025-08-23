"""
Gemini-powered summarizer for progressive reduction
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional
import yaml

from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class ProgressiveSummarizer:
    """
    Implements Gemini-powered summarization for progressive reduction
    """
    
    def __init__(self,
                 config_path: str = "agents/system prompts/progressive_summarizer.yaml",
                 model: Optional[str] = None,
                 progress_tracker: Optional['ProgressTracker'] = None):
        """Initialize the summarizer
        
        Args:
            config_path: Path to YAML configuration file
            model: Optional model override ('gemini-2.5-flash', 'gemini-pro', or 'claude-3-5-sonnet-20241022')
            progress_tracker: Optional ProgressTracker instance for context
        """
        self.config = self._load_config(config_path)
        
        # Extract configuration
        self.model_config = self.config.get('model_config', {})
        self.summarization_config = self.config.get('summarization_config', {})
        
        # Configure model settings
        self.model = model or self.model_config.get('model', 'gemini-2.5-flash')
        self.max_tokens = self.model_config.get('max_tokens', 500)
        self.temperature = self.model_config.get('temperature', 0.1)
        
        # Progress tracking
        self.progress_tracker = progress_tracker
        
        # Load prompts
        self.cluster_prompt = self.config.get('cluster_prompt', '')
        self.global_prompt = self.config.get('global_prompt', '')
        
        # Initialize Gemini client
        self._init_gemini_client()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"PROG-SUMMARIZER: Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"PROG-SUMMARIZER: Error loading config: {e}. Using defaults.")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model_config': {
                'model': 'gemini-2.5-flash',
                'max_tokens': 500,
                'temperature': 0.1
            },
            'summarization_config': {
                'cluster_summary_tokens': 200,
                'global_summary_tokens': 400
            },
            'cluster_prompt': """
            Create a concise summary of this message cluster from Slack.
            Focus on key points and maintain evidence links.
            
            CRITICAL:
            - Keep summary under 200 tokens
            - Preserve evidence IDs in parentheses
            - Format: Summary text (evidence_id1; evidence_id2)
            
            Cluster data:
            {cluster_data}
            """,
            'global_prompt': """
            Create a brief overview of these Slack discussion clusters.
            Focus on key findings and trends.
            
            CRITICAL:
            - Keep summary under 400 tokens
            - Include 2-3 key findings with evidence
            - Format each finding: • Finding text (evidence_id1; evidence_id2)
            
            Discussion data:
            {discussion_data}
            """
        }
        
    def _init_gemini_client(self):
        """Initialize Gemini client"""
        try:
            from google import genai
            from google.genai import types
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("PROG-SUMMARIZER: GEMINI_API_KEY not found")
                self.client = None
                self.types = None
                return
                
            self.client = genai.Client(api_key=api_key)
            self.types = types
            logger.info(f"PROG-SUMMARIZER: Initialized {self.model}")
            
        except Exception as e:
            logger.error(f"PROG-SUMMARIZER: Error initializing Gemini: {e}")
            self.client = None
            self.types = None
            
    async def summarize_cluster(self, cluster: Dict[str, Any]) -> str:
        """
        Summarize a message cluster
        
        Args:
            cluster: Cluster data including messages and features
            
        Returns:
            Summarized text with evidence IDs
        """
        if not self.client or not self.types:
            return self._fallback_cluster_summary(cluster)
            
        try:
            # Format cluster data
            cluster_data = {
                "messages": [
                    {
                        "text": item.get("text", ""),
                        "evidence_id": f"slack:ch/{item.get('channel')}/{item.get('ts')}"
                    }
                    for item in cluster["items"]
                ],
                "features": cluster["features"].__dict__  # Convert dataclass to dict
            }
            
            # Fill prompt template
            prompt = self.cluster_prompt.format(
                cluster_data=json.dumps(cluster_data, indent=2),
                goal="Analyze and summarize Slack discussions"  # Default goal for cluster summaries
            )
            
            # Generate summary
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=self.types.GenerateContentConfig(
                    max_output_tokens=self.summarization_config["cluster_summary_tokens"],
                    temperature=self.temperature
                )
            )
            
            if response is None or response.text is None:
                return self._fallback_cluster_summary(cluster)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"PROG-SUMMARIZER: Error summarizing cluster: {e}")
            return self._fallback_cluster_summary(cluster)
            
    async def create_global_brief(self,
                               clusters: Dict[str, Any],
                               query: str,
                               goal: Optional[str] = None) -> str:
        """
        Create global summary of all clusters
        
        Args:
            clusters: All cluster data
            query: Original search query
            goal: Research goal (if not using progress tracker)
            
        Returns:
            Global summary with key findings
        """
        if not self.client or not self.types:
            return self._fallback_global_summary(clusters, query)
            
        try:
            # Get progress context if available
            context = {}
            if self.progress_tracker:
                context = self.progress_tracker.get_iteration_context()
            else:
                context = {
                    "goal": goal,
                    "iteration": 1,
                    "progress": {"findings_count": 0},
                    "facet_guidance": []
                }
                
            # Format discussion data
            discussion_data = {
                "query": query,
                "goal": context["goal"],
                "iteration": context["iteration"],
                "progress_summary": context.get("progress", {}),
                "facet_guidance": context.get("facet_guidance", []),
                "clusters": [
                    {
                        "summary": cluster.get("summary_text", ""),
                        "features": cluster.get("features", {}),
                        "evidence_ids": cluster.get("evidence_ids", [])
                    }
                    for cluster in clusters.values()
                ]
            }
            
            # Fill prompt template with all required parameters
            prompt = self.global_prompt.format(
                goal=context["goal"],
                progress_summary=json.dumps(context.get("progress", {})),
                iteration_number=context["iteration"],
                facet_summary=json.dumps(context.get("facet_guidance", [])),
                previous_findings="No previous findings available" if not self.progress_tracker else "See progress tracker",
                discussion_data=json.dumps(discussion_data, indent=2)
            )
            
            # Generate summary
            response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=self.types.GenerateContentConfig(
                    max_output_tokens=self.summarization_config["global_summary_tokens"],
                    temperature=self.temperature
                )
            )
            
            if response is None or response.text is None:
                return self._fallback_global_summary(clusters, query)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"PROG-SUMMARIZER: Error creating global brief: {e}")
            return self._fallback_global_summary(clusters, query)
            
    def _fallback_cluster_summary(self, cluster: Dict[str, Any]) -> str:
        """Generate basic cluster summary when LLM is unavailable"""
        if not cluster["items"]:
            return "Empty cluster"
            
        # Use first message as preview
        main_message = cluster["items"][0]
        preview = main_message.get("text", "")[:100] + "..."
        
        # Get evidence IDs
        evidence_ids = [
            f"slack:ch/{item['channel']}/{item['ts']}"
            for item in cluster["items"][:2]
        ]
        
        return f"{preview} ({'; '.join(evidence_ids)})"
        
    def _fallback_global_summary(self, clusters: Dict[str, Any], query: str) -> str:
        """Generate basic global summary when LLM is unavailable"""
        cluster_count = len(clusters)
        total_messages = cluster_count  # Each cluster represents one message in fallback
        
        # Get top 2 clusters by engagement
        top_clusters = sorted(
            clusters.items(),
            key=lambda x: (
                x[1].get("engagement", {}).get("reactions", 0) +
                x[1].get("engagement", {}).get("replies", 0)
            ),
            reverse=True
        )[:2]
        
        findings = []
        for _, cluster in top_clusters:
            if cluster.get("summary_text"):
                findings.append(f"• {cluster['summary_text']}")
        
        summary = [
            f"Found {cluster_count} discussion clusters with {total_messages} total messages"
        ]
        if findings:
            summary.extend(["", "Key findings:"] + findings)
            
        return "\n".join(summary)

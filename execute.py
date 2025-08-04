"""
Pydantic Models for All Tool Input/Output Validation

This file contains all the data model classes generated from JSON schemas.
These provide type-safe interfaces for tool function calls.

Naming Convention:
- JSON files: {action}_input.json, {action}_output.json
- Pydantic classes: execute_{service}_{action}_input, execute_{service}_{action}_output  
- Commands: {service}.{action}

Generated from schemas in: schemas/services/{service}/{action}_{input|output}.json
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

# =============================================================================
# REGISTRY TOOL MODELS
# =============================================================================

class SearchType(str, Enum):
    """Search type enumeration for registry search"""
    capability = "capability"
    tag = "tag"
    category = "category"
    description = "description"
    name = "name"

class ComplexityLevel(str, Enum):
    """Tool complexity level enumeration"""
    simple = "simple"
    moderate = "moderate"
    complex = "complex"

class ImplementationType(str, Enum):
    """Tool implementation type enumeration"""
    rest_api = "rest_api"
    mcp_server = "mcp_server"
    internal = "internal"

# Registry Search Models
class execute_reg_search_input(BaseModel):
    """Input model for registry search operation"""
    explanation: str = Field(..., description="One sentence explanation of why this search is being performed")
    query: str = Field(..., description="Search query string")
    search_type: SearchType = Field(default=SearchType.description, description="Type of search to perform")
    categories: Optional[List[str]] = Field(default=None, max_length=10, description="Filter by specific categories")
    complexity: Optional[ComplexityLevel] = Field(default=None, description="Filter by tool complexity level")
    implementation_type: Optional[ImplementationType] = Field(default=None, description="Filter by implementation type")
    capabilities: Optional[List[str]] = Field(default=None, max_length=10, description="Filter by specific capabilities")
    include_schemas: bool = Field(default=False, description="Include full input/output schemas in results")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")

class ToolInfo(BaseModel):
    """Tool information model"""
    name: str = Field(..., description="Tool name in format service.action")
    display_name: str = Field(..., description="Human-readable tool name")
    description: List[str] = Field(..., description="Detailed tool description array")
    category: str = Field(..., description="Tool category")
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    capabilities: List[str] = Field(..., description="Tool capabilities")
    use_cases: List[str] = Field(default_factory=list, description="Common use cases")
    complexity: ComplexityLevel = Field(..., description="Tool complexity level")
    output_type: str = Field(..., description="Type of output the tool provides")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for using the tool")
    related_tools: List[str] = Field(default_factory=list, description="Names of related tools")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Usage examples")
    implementation_type: ImplementationType = Field(..., description="How the tool is implemented")
    schemas: Optional[Dict[str, str]] = Field(default=None, description="Schema file paths")

class execute_reg_search_output(BaseModel):
    """Output model for registry search operation"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Search result data")

# Registry Describe Models
class execute_reg_describe_input(BaseModel):
    """Input model for registry describe operation"""
    explanation: str = Field(..., description="One sentence explanation of why this tool description is needed")
    tool_name: str = Field(..., description="Name of the tool to describe (format: service.action)")
    include_schema: bool = Field(default=True, description="Include input/output schema information")

class execute_reg_describe_output(BaseModel):
    """Output model for registry describe operation"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Tool description data")

# Registry List Models
class execute_reg_list_input(BaseModel):
    """Input model for registry list operation"""
    explanation: str = Field(..., description="One sentence explanation of why the tool list is needed")
    category: Optional[str] = Field(default=None, description="Filter by specific category")
    complexity: Optional[ComplexityLevel] = Field(default=None, description="Filter by tool complexity level")
    implementation_type: Optional[ImplementationType] = Field(default=None, description="Filter by implementation type")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of tools to return")

class execute_reg_list_output(BaseModel):
    """Output model for registry list operation"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Tool list data")

# Registry Categories Models
class execute_reg_categories_input(BaseModel):
    """Input model for registry categories operation"""
    explanation: str = Field(..., description="One sentence explanation of why category information is needed")

class CategoryInfo(BaseModel):
    """Category information model"""
    name: str = Field(..., description="Human-readable category name")
    description: str = Field(..., description="Category description")
    icon: str = Field(..., description="Category icon emoji")
    tool_count: int = Field(default=0, description="Number of tools in this category")

class execute_reg_categories_output(BaseModel):
    """Output model for registry categories output"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Categories data")

# =============================================================================
# PERPLEXITY TOOL MODELS
# =============================================================================

class PerplexityModel(str, Enum):
    """Perplexity model enumeration for search operations"""
    sonar = "sonar"
    sonar_pro = "sonar-pro"
    sonar_reasoning = "sonar-reasoning"
    sonar_reasoning_pro = "sonar-reasoning-pro"

class PerplexityResearchModel(str, Enum):
    """Perplexity model enumeration for research operations"""
    sonar_deep_research = "sonar-deep-research"

class RecencyFilter(str, Enum):
    """Time filter enumeration for search results"""
    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    year = "year"

class SearchContextSize(str, Enum):
    """Search context size enumeration"""
    low = "low"
    medium = "medium"
    high = "high"

# Perplexity Search Models
class execute_perplexity_search_input(BaseModel):
    """Input model for Perplexity search operation"""
    explanation: str = Field(..., description="One sentence explanation of why this search is being performed")
    query: str = Field(..., min_length=1, max_length=2000, description="The search query or question to ask Perplexity")
    model: PerplexityModel = Field(default=PerplexityModel.sonar, description="Perplexity model to use for search")
    api_key: str = Field(..., min_length=1, description="Perplexity API key for authentication")
    temperature: float = Field(default=0.2, ge=0, le=2, description="Controls randomness in the response")
    max_tokens: int = Field(default=1000, ge=1, le=8000, description="Maximum number of tokens to generate")
    search_domain_filter: Optional[List[str]] = Field(default=None, max_length=20, description="List of domains to include/exclude (prefix with '-' to exclude)")
    search_recency_filter: Optional[RecencyFilter] = Field(default=None, description="Filter results by recency")
    return_images: bool = Field(default=False, description="Whether to include images in the response")
    return_related_questions: bool = Field(default=False, description="Whether to return related follow-up questions")
    search_context_size: SearchContextSize = Field(default=SearchContextSize.medium, description="Amount of search context to include")

class CitationData(BaseModel):
    """Citation information model"""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    snippet: str = Field(..., description="Relevant snippet from the source")

class ImageData(BaseModel):
    """Image information model"""
    url: str = Field(..., description="Image URL")
    description: str = Field(..., description="Image description")

class SearchUsageData(BaseModel):
    """Search usage information model"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")

class execute_perplexity_search_output(BaseModel):
    """Output model for Perplexity search operation"""
    success: bool = Field(..., description="Whether the search operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Search result data")

# Perplexity Research Models
class execute_perplexity_research_input(BaseModel):
    """Input model for Perplexity research operation"""
    explanation: str = Field(..., description="One sentence explanation of why this research is being performed")
    directive: str = Field(..., min_length=1, max_length=5000, description="The research directive or comprehensive question to investigate")
    model: PerplexityResearchModel = Field(default=PerplexityResearchModel.sonar_deep_research, description="Perplexity model to use for deep research")
    api_key: str = Field(..., min_length=1, description="Perplexity API key for authentication")
    temperature: float = Field(default=0.1, ge=0, le=2, description="Controls randomness in the response (lower for research)")
    max_tokens: int = Field(default=4000, ge=1, le=8000, description="Maximum number of tokens to generate")
    search_domain_filter: Optional[List[str]] = Field(default=None, max_length=30, description="List of domains to include/exclude for research (prefix with '-' to exclude)")
    search_recency_filter: Optional[RecencyFilter] = Field(default=None, description="Filter research sources by recency")
    search_after_date_filter: Optional[str] = Field(default=None, pattern=r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{4})$", description="Only include sources published after this date (MM/DD/YYYY format)")
    search_before_date_filter: Optional[str] = Field(default=None, pattern=r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{4})$", description="Only include sources published before this date (MM/DD/YYYY format)")
    return_related_questions: bool = Field(default=True, description="Whether to return related follow-up questions for further research")
    search_context_size: SearchContextSize = Field(default=SearchContextSize.high, description="Amount of search context to include for comprehensive research")
    focus_areas: Optional[List[str]] = Field(default=None, max_length=10, description="Specific areas or topics to focus the research on")

class ResearchCitationData(BaseModel):
    """Enhanced citation information model for research"""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    snippet: str = Field(..., description="Relevant snippet from the source")
    published_date: str = Field(default="", description="Publication date of the source")
    domain: str = Field(default="", description="Source domain")

class ResearchUsageData(BaseModel):
    """Research usage information model"""
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")
    search_count: Optional[int] = Field(default=None, description="Number of searches performed during research")
    reasoning_tokens: Optional[int] = Field(default=None, description="Number of reasoning tokens used")

class ResearchMethodologyData(BaseModel):
    """Research methodology information model"""
    search_strategy: str = Field(..., description="Description of the search strategy used")
    sources_evaluated: int = Field(default=0, description="Number of sources evaluated during research")
    research_duration: str = Field(default="", description="Time taken to complete the research")

class execute_perplexity_research_output(BaseModel):
    """Output model for Perplexity research operation"""
    success: bool = Field(..., description="Whether the research operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Research result data")

# =============================================================================
# WEATHER TOOL MODELS
# =============================================================================

class WeatherUnits(str, Enum):
    """Weather units enumeration"""
    standard = "standard"  # Kelvin, m/s, hPa
    metric = "metric"      # Celsius, m/s, hPa
    imperial = "imperial"  # Fahrenheit, mph, hPa

class WeatherMode(str, Enum):
    """Weather response mode enumeration"""
    json = "json"
    xml = "xml"
    html = "html"

# Weather Search (Geocoding) Models
class execute_weather_search_input(BaseModel):
    """Input model for weather geocoding search"""
    explanation: str = Field(..., description="One sentence explanation of why geocoding is needed")
    q: str = Field(..., min_length=1, description="City name, state code and country code divided by comma")
    appid: str = Field(..., min_length=1, description="OpenWeather API key")
    limit: int = Field(default=5, ge=1, le=5, description="Number of locations to return (max 5)")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "explanation": "User wants weather for San Francisco but I need coordinates first",
                    "q": "San Francisco,CA,US",
                    "appid": "your_api_key",
                    "limit": 1
                }
            ]
        }

class LocationData(BaseModel):
    """Geographic location data model"""
    name: str = Field(..., description="Name of the location")
    lat: float = Field(..., description="Latitude coordinate")
    lon: float = Field(..., description="Longitude coordinate")
    country: str = Field(..., pattern="^[A-Z]{2}$", description="Country code (ISO 3166-1 alpha-2)")
    state: Optional[str] = Field(default=None, description="State or region name")
    local_names: Optional[Dict[str, str]] = Field(default=None, description="City names in different languages")

class execute_weather_search_output(BaseModel):
    """Output model for weather geocoding search"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: List[LocationData] = Field(..., max_length=5, description="Array of location matches")

# Weather Current Models  
class execute_weather_current_input(BaseModel):
    """Input model for current weather data"""
    explanation: str = Field(..., description="One sentence explanation of why current weather is needed")
    
    # Location parameters (one required via model validation)
    lat: Optional[float] = Field(default=None, description="Latitude coordinate")
    lon: Optional[float] = Field(default=None, description="Longitude coordinate")
    q: Optional[str] = Field(default=None, description="City name, state code, country code")
    id: Optional[int] = Field(default=None, description="City ID")
    zip: Optional[str] = Field(default=None, description="Zip/postal code with country code")
    
    # API parameters
    appid: str = Field(..., min_length=1, description="OpenWeather API key")
    mode: Optional[WeatherMode] = Field(default=WeatherMode.json, description="Response format")
    units: Optional[WeatherUnits] = Field(default=WeatherUnits.standard, description="Temperature units")
    lang: Optional[str] = Field(default=None, description="Language code (ISO 639-1)")

    def model_validate(cls, values):
        """Validate that at least one location parameter is provided"""
        lat, lon, q, city_id, zip_code = values.get('lat'), values.get('lon'), values.get('q'), values.get('id'), values.get('zip')
        
        if lat is not None and lon is not None:
            return values
        elif any([q, city_id, zip_code]):
            return values
        else:
            raise ValueError("Location required: provide lat/lon, q (city name), id (city ID), or zip code")

class execute_weather_current_output(BaseModel):
    """Output model for current weather data"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Current weather data in OpenWeather format")

# Weather Forecast Models
class execute_weather_forecast_input(BaseModel):
    """Input model for weather forecast data"""
    explanation: str = Field(..., description="One sentence explanation of why forecast data is needed")
    lat: float = Field(..., description="Latitude coordinate (required for One Call API)")
    lon: float = Field(..., description="Longitude coordinate (required for One Call API)")
    appid: str = Field(..., min_length=1, description="OpenWeather API key")
    exclude: Optional[str] = Field(default=None, description="Exclude forecast parts (comma-separated: current,minutely,hourly,daily,alerts)")
    units: Optional[WeatherUnits] = Field(default=WeatherUnits.standard, description="Temperature units")
    lang: Optional[str] = Field(default=None, description="Language code (ISO 639-1)")

class execute_weather_forecast_output(BaseModel):
    """Output model for weather forecast data"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Weather forecast data in OpenWeather One Call API format")

# =============================================================================
# MEMORY TOOL MODELS (MemoryOS)
# =============================================================================

class ExecutionDetails(BaseModel):
    """Execution details for conversation memories"""
    tools_used: Optional[List[str]] = Field(default=None, description="List of tools that were executed, in chronological order")
    errors: Optional[List[Dict[str, str]]] = Field(default=None, description="Any errors that occurred during execution")
    duration_ms: Optional[int] = Field(default=None, ge=0, description="How long the execution took in milliseconds")
    success: Optional[bool] = Field(default=None, description="Whether the overall execution was successful")

# Conversation Memory Models
class execute_memory_conversation_add_input(BaseModel):
    """Input model for adding conversation memories with optional execution data to MemoryOS"""
    user_input: str = Field(..., description="The user's input or question.")
    agent_response: str = Field(..., description="The agent's response.")
    user_id: str = Field(..., description="User identifier for memory isolation.")
    message_id: Optional[str] = Field(default=None, description="Optional unique message ID for linking memories.")
    timestamp: Optional[str] = Field(default=None, description="Optional ISO format timestamp.")
    meta_data: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata dictionary.")
    execution_details: Optional[ExecutionDetails] = Field(default=None, description="Optional execution details for this conversation")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for filtering (e.g., ['conversation', 'execution'])")

class execute_memory_conversation_add_output(BaseModel):
    """Output model for conversation memory add operation"""
    success: bool = Field(..., description="Whether the memory addition was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Operation result details")

class execute_memory_conversation_retrieve_input(BaseModel):
    """Input model for retrieving conversation memories from MemoryOS"""
    query: str = Field(..., description="The search query for retrieving memories.")
    user_id: str = Field(..., description="User identifier for memory isolation.")
    message_id: Optional[str] = Field(default=None, description="Optional specific message ID to retrieve a single memory.")
    max_results: int = Field(default=10, description="Maximum number of results to return.")
    tags_filter: Optional[List[str]] = Field(default=None, description="Optional list of tags to filter memories (e.g., [\"conversation\", \"execution\"])")

class execute_memory_conversation_retrieve_output(BaseModel):
    """Output model for conversation memory retrieval operation"""
    status: str = Field(..., description="Operation status (success/error)")
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="Retrieved conversation details")


class execute_memory_get_profile_input(BaseModel):
    """Input model for getting agent-generated user profile from MemoryOS embeddings search"""
    explanation: str = Field(..., description="One sentence explanation of why the agent-generated profile is being requested")
    include_knowledge: bool = Field(default=True, description="Whether to include agent-generated knowledge entries from embeddings search")
    include_assistant_knowledge: bool = Field(default=False, description="Whether to include assistant knowledge base entries from embeddings search")

class execute_memory_get_profile_output(BaseModel):
    """Output model for agent-generated user profile retrieval operation"""
    success: bool = Field(..., description="Whether the agent-generated profile retrieval was successful")
    message: str = Field(..., description="Human-readable result message")
    data: Dict[str, Any] = Field(..., description="Agent-generated profile data from embeddings search including inferred personality traits, preferences, and knowledge")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_model_classes() -> Dict[str, BaseModel]:
    """Get all Pydantic model classes defined in this module"""
    import inspect
    
    models = {}
    for name, obj in inspect.getmembers(inspect.getmodule(inspect.currentframe())):
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseModel) and 
            obj != BaseModel):
            models[name] = obj
    
    return models

def get_tool_models(service: str, action: str) -> tuple[BaseModel, BaseModel]:
    """Get input and output models for a specific tool"""
    input_class_name = f"execute_{service}_{action}_input"
    output_class_name = f"execute_{service}_{action}_output"
    
    models = get_all_model_classes()
    
    input_model = models.get(input_class_name)
    output_model = models.get(output_class_name)
    
    if not input_model or not output_model:
        raise ValueError(f"Models not found for {service}.{action}")
    
    return input_model, output_model 
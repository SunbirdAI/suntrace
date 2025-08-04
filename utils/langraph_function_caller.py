import os
import json
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from shapely.wkt import loads as wkt_loads
import unicodedata

# Load system prompt from external file
from configs.system_prompt import SYSTEM_PROMPT


class LangGraphGeospatialAgent:
    """
    A LangGraph-based agent for geospatial analysis using custom tools.
    """
    
    def __init__(self, geospatial_analyzer=None):
        """
        Initialize the LangGraph agent with geospatial analysis capabilities.
        
        Args:
            geospatial_analyzer: Instance of GeospatialAnalyzer for tool execution
        """
        if geospatial_analyzer is None:
            from utils.factory import create_geospatial_analyzer
            geospatial_analyzer = create_geospatial_analyzer()
        
        self.geospatial_analyzer = geospatial_analyzer
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Set up memory for multi-turn conversations
        self.checkpointer = InMemorySaver()
        
        # Create the agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer
        )
    
    def _create_tools(self) -> List:
        """Create LangGraph-compatible tools from the geospatial analyzer methods."""
        
        @tool
        def count_features_within_region(region: str, layer_name: str, filter_expr: Optional[str] = None) -> Dict[str, Any]:
            """
            Counts features in a specified geospatial layer that intersect with a given region.
            
            Args:
                region: The area as a Shapely Polygon in WKT format
                layer_name: The layer to count features in (buildings, tiles, roads, villages, parishes, subcounties, existing_grid, grid_extension, candidate_minigrids, existing_minigrids)
                filter_expr: Optional pandas-style filter (e.g. "type=='residential'")
                
            Returns:
                Dictionary containing the count and analysis results
            """
            try:
                region_geom = wkt_loads(region)
                return self.geospatial_analyzer.count_features_within_region(
                    region_geom, layer_name, filter_expr
                )
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def analyze_region(region: str) -> Dict[str, Any]:
            """
            Performs comprehensive analysis of a geographic region, providing structured insights 
            about settlements, infrastructure, and environmental characteristics.
            
            Args:
                region: The geographic area (as a Shapely Polygon in WKT format) to analyze
                
            Returns:
                Dictionary containing comprehensive regional analysis
            """
            try:
                region_geom = wkt_loads(region)
                return self.geospatial_analyzer.analyze_region(region_geom)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def analyze_environmental_metrics(region: str) -> Dict[str, Any]:
            """
            Performs environmental analysis of a geographic region including NDVI, elevation, slope, 
            rainfall, and vegetation density metrics.
            
            Args:
                region: The geographic area (as a Shapely Polygon in WKT format) to analyze
                
            Returns:
                Dictionary containing environmental metrics like ndvi_mean, elev_mean, slope_mean, rain_total_mm, etc.
            """
            try:
                region_geom = wkt_loads(region)
                return self.geospatial_analyzer._analyze_environmental_metrics(region_geom)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def analyze_settlements_in_region(region: str) -> Dict[str, Any]:
            """
            Analyzes building data and settlement patterns within a specified geographic region.
            Returns building counts, categories, and intersecting villages with electrification status.
            
            Args:
                region: The geographic area (as a Shapely Polygon in WKT format) to analyze
                
            Returns:
                Dictionary containing settlement analysis including building_count, building_categories, 
                intersecting_village_count, and electrification details
            """
            try:
                region_geom = wkt_loads(region)
                return self.geospatial_analyzer._analyze_settlements_in_region(region_geom)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def analyze_infrastructure_in_region(region: str) -> Dict[str, Any]:
            """
            Analyzes infrastructure elements including roads, electricity grid, and energy systems.
            
            Args:
                region: The geographic area (as a Shapely Polygon in WKT format) to analyze
                
            Returns:
                Dictionary containing infrastructure analysis including roads, electricity grid status,
                distance to grid, minigrid information, and capacity distribution
            """
            try:
                region_geom = wkt_loads(region)
                return self.geospatial_analyzer._analyze_infrastructure_in_region(region_geom)
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def compute_distance_to_grid(geometry: str) -> Dict[str, Any]:
            """
            Calculates the distance from a given geometry to the nearest grid infrastructure.
            
            Args:
                geometry: The geometry to measure distance from (as a Shapely geometry in WKT format)
                
            Returns:
                Dictionary containing the distance in meters to the nearest grid infrastructure
            """
            try:
                geom = wkt_loads(geometry)
                return self.geospatial_analyzer.compute_distance_to_grid(geom)
            except Exception as e:
                return {"error": str(e)}
        
        return [
            count_features_within_region,
            analyze_region,
            analyze_environmental_metrics,
            analyze_settlements_in_region,
            analyze_infrastructure_in_region,
            compute_distance_to_grid
        ]
    
    def ask(self, user_prompt: str, thread_id: str = "default") -> str:
        """
        Process a user query using the LangGraph agent.
        
        Args:
            user_prompt: The user's question or request
            thread_id: Unique identifier for the conversation thread (for memory)
            
        Returns:
            String response from the agent
        """
        try:
            # Clean the user prompt
            if isinstance(user_prompt, bytes):
                user_prompt = user_prompt.decode('utf-8', errors='ignore')
            else:
                user_prompt = str(user_prompt)
            
            # Normalize Unicode characters
            user_prompt = unicodedata.normalize('NFKD', user_prompt)
            
            # Configure the conversation thread
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke the agent
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config=config
            )
            
            # Extract the final response
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            
            return "I apologize, but I couldn't generate a response. Please try again."
            
        except Exception as e:
            error_msg = f"ERROR in LangGraph agent: {str(e)}"
            print(error_msg)
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return error_msg
    
    def stream(self, user_prompt: str, thread_id: str = "default"):
        """
        Stream the agent's response for real-time updates.
        
        Args:
            user_prompt: The user's question or request
            thread_id: Unique identifier for the conversation thread
            
        Yields:
            Streaming updates from the agent
        """
        try:
            # Clean the user prompt
            if isinstance(user_prompt, bytes):
                user_prompt = user_prompt.decode('utf-8', errors='ignore')
            else:
                user_prompt = str(user_prompt)
            
            # Normalize Unicode characters
            user_prompt = unicodedata.normalize('NFKD', user_prompt)
            
            # Configure the conversation thread
            config = {"configurable": {"thread_id": thread_id}}
            
            # Stream the agent's response
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config=config
            ):
                yield chunk
                
        except Exception as e:
            error_msg = f"ERROR in LangGraph streaming: {str(e)}"
            print(error_msg)
            yield {"error": error_msg}


# Convenience function to maintain compatibility with existing code
def ask_with_functions(user_prompt: str, analyzer=None, thread_id: str = "default") -> str:
    """
    Backward-compatible function that uses LangGraph agent.
    
    Args:
        user_prompt: The user's question or request
        analyzer: GeospatialAnalyzer instance (optional)
        thread_id: Conversation thread ID for memory
        
    Returns:
        String response from the agent
    """
    # Create a global agent instance to maintain conversation state
    if not hasattr(ask_with_functions, '_agent'):
        ask_with_functions._agent = LangGraphGeospatialAgent(analyzer)
    
    return ask_with_functions._agent.ask(user_prompt, thread_id)


# Example usage (commented out to avoid running on import)
# if __name__ == "__main__":
#     # Create agent
#     agent = LangGraphGeospatialAgent()
#     
#     # Example query
#     response = agent.ask("How many buildings are in the region? Here's the WKT: POLYGON((...))")
#     print(response)
#     
#     # Example with streaming
#     for chunk in agent.stream("Analyze the environmental conditions in this area"):
#         print(chunk)
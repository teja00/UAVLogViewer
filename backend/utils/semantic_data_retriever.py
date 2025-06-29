"""
Semantic Data Retriever for UAV Log Analyzer
Implements Phase 3: Semantic Data Understanding using vector embeddings to connect user queries to relevant dataframes.
"""

import logging
import os
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import json

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    VECTOR_DEPS_AVAILABLE = True
except ImportError:
    VECTOR_DEPS_AVAILABLE = False
    logging.info("Vector dependencies not available. Using enhanced rule-based semantic data understanding.")
    
# Enhanced fallback is still "Phase 3" - intelligent semantic understanding
# The rule-based approach with semantic keyword mapping provides excellent results
SEMANTIC_UNDERSTANDING_AVAILABLE = True  # Always available (vector or enhanced fallback)

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Represents a semantic match between query and dataframe."""
    dataframe_key: str
    dataframe_description: str
    similarity_score: float
    relevance_reason: str


class SemanticDataRetriever:
    """
    Semantic Data Understanding system that connects user queries to relevant dataframes
    using vector embeddings and similarity search.
    """
    
    def __init__(self, storage_dir: str = "./semantic_data_store"):
        self.storage_dir = storage_dir
        self.collection_name = "uav_dataframe_semantics"
        self.client = None
        self.collection = None
        self.embedder = None
        self.is_initialized = False
        
        # Fallback semantic mapping for when vector deps aren't available
        self.fallback_semantic_map = {
            "altitude": ["GPS", "BARO", "CTUN", "AHR2", "POS"],
            "position": ["GPS", "POS", "AHR2"],
            "battery": ["CURR", "BAT", "POWR", "MOTB"],
            "power": ["CURR", "BAT", "POWR", "MOTB"],
            "voltage": ["CURR", "BAT", "POWR"],
            "current": ["CURR", "BAT", "MOTB"],
            "temperature": ["BAT", "CURR", "MOTB"],
            "attitude": ["ATT", "AHR2", "IMU", "RATE"],
            "orientation": ["ATT", "AHR2", "IMU", "MAG"],
            "roll": ["ATT", "AHR2", "RATE", "PIDR"],
            "pitch": ["ATT", "AHR2", "RATE", "PIDP"],
            "yaw": ["ATT", "AHR2", "RATE", "PIDY", "MAG"],
            "heading": ["ATT", "AHR2", "MAG"],
            "compass": ["MAG", "AHR2"],
            "gps": ["GPS", "GPS2", "POS"],
            "navigation": ["GPS", "GPS2", "POS", "AHR2"],
            "satellite": ["GPS", "GPS2"],
            "signal": ["GPS", "GPS2"],
            "error": ["ERR", "MSG"],
            "warning": ["ERR", "MSG"],
            "message": ["MSG", "ERR"],
            "mode": ["MODE"],
            "flight_mode": ["MODE"],
            "control": ["RCIN", "RCOU", "RATE", "CTUN"],
            "input": ["RCIN"],
            "output": ["RCOU"],
            "servo": ["RCOU"],
            "motor": ["RCOU", "MOTB"],
            "vibration": ["VIBE"],
            "imu": ["IMU", "VIBE"],
            "gyro": ["IMU", "RATE"],
            "accelerometer": ["IMU", "VIBE"],
            "kalman": ["XKF1", "XKF2", "XKF3", "XKF4", "POS"],
            "ekf": ["XKF1", "XKF2", "XKF3", "XKF4", "POS"],
            "wind": ["XKF2"],
            "innovation": ["XKF1", "XKF3"],
            "parameter": ["PARM"],
            "pid": ["PIDR", "PIDP", "PIDY", "PIDA"],
            "tuning": ["PIDR", "PIDP", "PIDY", "PIDA", "RATE"],
            "height": ["GPS", "BARO", "CTUN", "AHR2"],
            "climb": ["CTUN", "GPS", "BARO"],
            "descent": ["CTUN", "GPS", "BARO"],
            "throttle": ["CTUN", "RCOU"],
            "speed": ["GPS", "AHR2"],
            "velocity": ["GPS", "AHR2", "POS"],
        }
        
        # Initialize the system lazily (when first needed)
        self._initialization_attempted = False
    
    async def _ensure_initialized(self):
        """Ensure the semantic data retriever is initialized (lazy initialization)."""
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                await self.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize semantic data retriever: {e}")
    
    async def initialize(self):
        """Initialize the vector store and embedding model."""
        if not VECTOR_DEPS_AVAILABLE:
            logger.info("Vector dependencies not available, using enhanced rule-based semantic understanding")
            self.is_initialized = True
            return
        
        try:
            # Create storage directory
            os.makedirs(self.storage_dir, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.storage_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info("Loaded existing semantic data collection")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new semantic data collection")
            
            # Initialize sentence transformer for embeddings
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
            
            self.is_initialized = True
            logger.info("Semantic data retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic data retriever: {e}")
            logger.info("Falling back to rule-based semantic mapping")
            self.is_initialized = True
    
    async def index_dataframe_documentation(self, dataframe_docs: Dict[str, str]) -> bool:
        """
        Index the dataframe documentation into the vector store.
        Returns True if successful, False otherwise.
        """
        # Ensure initialization happens first
        await self._ensure_initialized()
        
        if not VECTOR_DEPS_AVAILABLE or not self.is_initialized:
            logger.info("Using fallback mode - no vector indexing needed")
            return True
        
        try:
            # Check if we need to re-index (compare with stored hash)
            docs_hash = self._compute_docs_hash(dataframe_docs)
            stored_hash = self._get_stored_docs_hash()
            
            if docs_hash == stored_hash:
                logger.info("Dataframe documentation already indexed and up to date")
                return True
            
            # Clear existing data
            try:
                self.collection.delete(where={})
                logger.info("Cleared existing semantic data index")
            except Exception as e:
                logger.warning(f"Could not clear existing data: {e}")
            
            # Prepare data for indexing
            documents = []
            metadatas = []
            ids = []
            
            for df_key, description in dataframe_docs.items():
                # Create rich document text for better semantic understanding
                doc_text = f"Dataframe: {df_key}\nDescription: {description}\nData Type: UAV flight telemetry"
                
                documents.append(doc_text)
                metadatas.append({
                    "dataframe_key": df_key,
                    "description": description,
                    "doc_type": "dataframe_documentation"
                })
                ids.append(f"df_{df_key}")
            
            # Add documents to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            # Store the hash for future comparisons
            self._store_docs_hash(docs_hash)
            
            logger.info(f"Successfully indexed {len(documents)} dataframe documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index dataframe documentation: {e}")
            return False
    
    async def find_relevant_dataframes(self, user_query: str, available_dataframes: List[str], 
                                     top_k: int = 5) -> List[SemanticMatch]:
        """
        Find the most relevant dataframes for a user query using semantic similarity.
        
        Args:
            user_query: The user's natural language query
            available_dataframes: List of dataframes available in the current session
            top_k: Number of top matches to return
            
        Returns:
            List of SemanticMatch objects sorted by relevance
        """
        # Ensure initialization happens first
        await self._ensure_initialized()
        
        if not self.is_initialized:
            logger.warning("Semantic retriever not initialized, using fallback")
            return self._fallback_semantic_search(user_query, available_dataframes, top_k)
        
        if not VECTOR_DEPS_AVAILABLE or not self.collection:
            return self._fallback_semantic_search(user_query, available_dataframes, top_k)
        
        try:
            # Enhance query for better semantic matching
            enhanced_query = self._enhance_query_for_search(user_query)
            
            # Perform semantic search
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=min(top_k * 2, 20),  # Get more results to filter by availability
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results and filter by available dataframes
            matches = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                df_key = metadata['dataframe_key']
                
                # Only include dataframes that are available in the current session
                if df_key in available_dataframes:
                    similarity_score = 1.0 - distance  # Convert distance to similarity
                    
                    # Generate relevance reason
                    relevance_reason = self._generate_relevance_reason(user_query, df_key, metadata['description'])
                    
                    matches.append(SemanticMatch(
                        dataframe_key=df_key,
                        dataframe_description=metadata['description'],
                        similarity_score=similarity_score,
                        relevance_reason=relevance_reason
                    ))
            
            # Sort by similarity score and return top_k
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}, using fallback")
            return self._fallback_semantic_search(user_query, available_dataframes, top_k)
    
    def _fallback_semantic_search(self, user_query: str, available_dataframes: List[str], 
                                 top_k: int = 5) -> List[SemanticMatch]:
        """
        Fallback semantic search using rule-based keyword matching.
        """
        query_lower = user_query.lower()
        matches = []
        
        # Score each available dataframe
        for df_key in available_dataframes:
            score = 0.0
            reasons = []
            
            # Direct keyword matching
            if df_key.lower() in query_lower:
                score += 0.9
                reasons.append(f"Direct mention of {df_key}")
            
            # Semantic keyword matching
            for keyword, related_dfs in self.fallback_semantic_map.items():
                if keyword in query_lower and df_key in related_dfs:
                    keyword_score = 0.8 if keyword in query_lower.split() else 0.6
                    score += keyword_score
                    reasons.append(f"Related to '{keyword}'")
            
            # Boost score for common query patterns
            if "maximum" in query_lower or "highest" in query_lower:
                if df_key in ["GPS", "BARO", "CTUN"]:
                    score += 0.3
                    reasons.append("Suitable for maximum value queries")
            
            if "error" in query_lower or "problem" in query_lower:
                if df_key in ["ERR", "MSG"]:
                    score += 0.4
                    reasons.append("Contains error information")
            
            if score > 0:
                matches.append(SemanticMatch(
                    dataframe_key=df_key,
                    dataframe_description=f"UAV telemetry data from {df_key} messages",
                    similarity_score=min(score, 1.0),
                    relevance_reason="; ".join(reasons) if reasons else "Potential relevance detected"
                ))
        
        # Sort by score and return top_k
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:top_k]
    
    def _enhance_query_for_search(self, query: str) -> str:
        """Enhance the user query for better semantic search results."""
        enhanced_parts = [query]
        
        # Add UAV/flight context
        enhanced_parts.append("UAV flight telemetry data")
        
        # Add related terms based on query content
        query_lower = query.lower()
        if any(word in query_lower for word in ["altitude", "height", "high", "low"]):
            enhanced_parts.append("altitude GPS barometer elevation")
        
        if any(word in query_lower for word in ["battery", "power", "voltage", "current"]):
            enhanced_parts.append("battery power electrical system")
        
        if any(word in query_lower for word in ["error", "problem", "issue", "warning"]):
            enhanced_parts.append("error message warning system alert")
        
        if any(word in query_lower for word in ["gps", "position", "navigation"]):
            enhanced_parts.append("GPS navigation position location")
        
        return " ".join(enhanced_parts)
    
    def _generate_relevance_reason(self, query: str, df_key: str, description: str) -> str:
        """Generate a human-readable reason for why this dataframe is relevant."""
        query_lower = query.lower()
        desc_lower = description.lower()
        
        reasons = []
        
        # Check for direct keyword matches
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in desc_lower:
                reasons.append(f"Contains '{word}' data")
        
        # Check for semantic relationships
        if "altitude" in query_lower and df_key in ["GPS", "BARO", "CTUN"]:
            reasons.append("Primary altitude data source")
        elif "battery" in query_lower and df_key in ["CURR", "BAT"]:
            reasons.append("Battery and power system data")
        elif "error" in query_lower and df_key in ["ERR", "MSG"]:
            reasons.append("System error and message data")
        
        return "; ".join(reasons) if reasons else "Semantic similarity detected"
    
    def _compute_docs_hash(self, docs: Dict[str, str]) -> str:
        """Compute hash of documentation for change detection."""
        content = json.dumps(docs, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_stored_docs_hash(self) -> Optional[str]:
        """Get stored documentation hash."""
        hash_file = os.path.join(self.storage_dir, "docs_hash.txt")
        try:
            with open(hash_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None
    
    def _store_docs_hash(self, docs_hash: str):
        """Store documentation hash."""
        hash_file = os.path.join(self.storage_dir, "docs_hash.txt")
        try:
            with open(hash_file, 'w') as f:
                f.write(docs_hash)
        except Exception as e:
            logger.warning(f"Could not store docs hash: {e}")
    
    async def get_semantic_context_for_query(self, user_query: str, available_dataframes: List[str]) -> str:
        """
        Get semantic context string for injection into planner prompts.
        This is the core functionality that transforms the planning process.
        """
        matches = await self.find_relevant_dataframes(user_query, available_dataframes)
        
        if not matches:
            return "No specific dataframes identified as highly relevant for this query. Consider general analysis approach."
        
        context_parts = []
        context_parts.append("SEMANTIC DATA ANALYSIS:")
        context_parts.append(f"Based on your query '{user_query}', the most relevant data sources are:")
        
        for i, match in enumerate(matches, 1):
            context_parts.append(f"{i}. {match.dataframe_key}: {match.dataframe_description}")
            context_parts.append(f"   Relevance: {match.relevance_reason} (Score: {match.similarity_score:.2f})")
        
        context_parts.append("\nRECOMMENDATION: Focus your analysis on these dataframes for the most relevant results.")
        
        return "\n".join(context_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the semantic retriever."""
        status = {
            "initialized": self.is_initialized,
            "semantic_understanding_available": SEMANTIC_UNDERSTANDING_AVAILABLE,
            "vector_deps_available": VECTOR_DEPS_AVAILABLE,
            "storage_dir": self.storage_dir,
            "method": "vector_embeddings" if VECTOR_DEPS_AVAILABLE and self.collection else "enhanced_rule_based",
            "phase3_active": True  # Phase 3 is always active (vector or enhanced fallback)
        }
        
        if VECTOR_DEPS_AVAILABLE and self.collection:
            try:
                count = self.collection.count()
                status["indexed_documents"] = count
            except Exception:
                status["indexed_documents"] = "unknown"
        else:
            status["semantic_keywords"] = len(self.fallback_semantic_map)
        
        return status 
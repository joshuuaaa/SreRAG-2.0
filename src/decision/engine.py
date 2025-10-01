"""
Decision Tree Engine
Loads and executes YAML-based medical decision trees with safety-first logic.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DecisionTreeEngine:
    """Handles loading and navigation of decision trees."""
    
    def __init__(self, trees_path: str = "decision_trees"):
        self.trees_path = Path(trees_path)
        self.trees: Dict[str, Dict] = {}
        self.current_tree: Optional[str] = None
        self.current_node: Optional[str] = None
        self.conversation_history: List[Dict] = []
        
        self.load_trees()
    
    def load_trees(self):
        """Load all YAML decision trees from the trees directory."""
        if not self.trees_path.exists():
            logger.warning(f"Decision trees path not found: {self.trees_path}")
            return
        
        yaml_files = self.trees_path.glob("*.yaml")
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, 'r') as f:
                    tree_data = yaml.safe_load(f)
                    tree_id = tree_data.get('id', yaml_file.stem)
                    self.trees[tree_id] = tree_data
                    logger.info(f"Loaded decision tree: {tree_id}")
            except Exception as e:
                logger.error(f"Error loading {yaml_file}: {e}")
    
    def match_tree(self, user_input: str) -> Optional[str]:
        """
        Match user input to the most appropriate decision tree.
        Uses keyword matching (can be enhanced with semantic search).
        """
        user_lower = user_input.lower()
        
        # Priority matching - critical scenarios first
        priority_trees = [
            (tid, tree) for tid, tree in self.trees.items()
            if tree.get('priority') == 'critical'
        ]
        
        # Keywords for each tree (can be defined in YAML)
        tree_keywords = {
            'severe_bleeding': ['bleed', 'blood', 'cut', 'wound', 'hemorrhage', 'gash'],
            'cpr': ['breathing', 'breath', 'unconscious', 'collapsed', 'heart', 'pulse'],
            'burns': ['burn', 'fire', 'scald', 'hot', 'flame'],
            'choking': ['choke', 'choking', 'can\'t breathe', 'heimlich'],
        }
        
        # Check priority trees first
        for tree_id, tree in priority_trees:
            keywords = tree_keywords.get(tree_id, [])
            if any(keyword in user_lower for keyword in keywords):
                return tree_id
        
        # Check non-priority trees
        for tree_id, tree in self.trees.items():
            if tree.get('priority') != 'critical':
                keywords = tree_keywords.get(tree_id, [])
                if any(keyword in user_lower for keyword in keywords):
                    return tree_id
        
        return None
    
    def start_tree(self, tree_id: str) -> Dict[str, Any]:
        """Start executing a decision tree from its initial node."""
        if tree_id not in self.trees:
            raise ValueError(f"Tree '{tree_id}' not found")
        
        self.current_tree = tree_id
        tree = self.trees[tree_id]
        
        # Find the first node (usually 'initial_assessment' or similar)
        nodes = tree.get('nodes', [])
        if not nodes:
            raise ValueError(f"Tree '{tree_id}' has no nodes")
        
        first_node = nodes[0]
        self.current_node = first_node['id']
        
        logger.info(f"Started tree '{tree_id}' at node '{self.current_node}'")
        return self._format_node_response(first_node)
    
    def process_response(self, user_response: str) -> Dict[str, Any]:
        """Process user response and navigate to next node."""
        if not self.current_tree or not self.current_node:
            raise RuntimeError("No active tree. Call start_tree() first.")
        
        current_node_data = self._get_current_node()
        if not current_node_data:
            raise RuntimeError(f"Current node '{self.current_node}' not found")
        
        # Record interaction
        self.conversation_history.append({
            'node': self.current_node,
            'user_response': user_response,
        })
        
        # Determine next node based on user response
        next_node_id = self._determine_next_node(current_node_data, user_response)
        
        if next_node_id == 'end' or not next_node_id:
            return {'type': 'end', 'message': 'Scenario complete'}
        
        # Move to next node
        next_node = self._get_node_by_id(next_node_id)
        if not next_node:
            raise RuntimeError(f"Next node '{next_node_id}' not found")
        
        self.current_node = next_node_id
        return self._format_node_response(next_node)
    
    def _get_current_node(self) -> Optional[Dict]:
        """Get the current node data."""
        if not self.current_tree:
            return None
        
        tree = self.trees[self.current_tree]
        for node in tree.get('nodes', []):
            if node['id'] == self.current_node:
                return node
        return None
    
    def _get_node_by_id(self, node_id: str) -> Optional[Dict]:
        """Get a node by its ID from the current tree."""
        if not self.current_tree:
            return None
        
        tree = self.trees[self.current_tree]
        for node in tree.get('nodes', []):
            if node['id'] == node_id:
                return node
        return None
    
    def _determine_next_node(self, node: Dict, user_response: str) -> Optional[str]:
        """Determine the next node based on user response."""
        node_type = node.get('type')
        
        # If it's an instruction node, follow the 'next' field
        if node_type == 'instruction':
            return node.get('next')
        
        # If it's a question node, match the user's response to options
        if node_type == 'question':
            options = node.get('options', [])
            user_lower = user_response.lower().strip()
            
            for option in options:
                answer = option.get('answer', '').lower()
                # Simple matching (can be enhanced)
                if answer in user_lower or user_lower in answer:
                    return option.get('next')
            
            # Default fallback if no match
            if options:
                return options[0].get('next')
        
        return None
    
    def _format_node_response(self, node: Dict) -> Dict[str, Any]:
        """Format a node into a structured response."""
        return {
            'node_id': node['id'],
            'type': node.get('type'),
            'prompt': node.get('prompt', ''),
            'steps': node.get('steps', []),
            'details': node.get('details', []),
            'options': node.get('options', []),
            'rag_tags': node.get('rag_tags', []),
            'rag_filter': node.get('rag_filter', {}),
            'priority': node.get('priority'),
        }
    
    def reset(self):
        """Reset the current tree and node."""
        self.current_tree = None
        self.current_node = None
        self.conversation_history = []
        logger.info("Decision tree engine reset")
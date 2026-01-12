"""Adaptive configuration system with performance monitoring and auto-tuning"""

import os
import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring"""
    query_count: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    rag_retrieval_time: float = 0.0
    llm_generation_time: float = 0.0
    safety_validation_time: float = 0.0
    successful_responses: int = 0
    failed_responses: int = 0
    safety_blocks: int = 0
    safety_warnings: int = 0
    emergency_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.emergency_types is None:
            self.emergency_types = {}

@dataclass
class SystemHealth:
    """System health indicators"""
    overall_health: str  # excellent, good, fair, poor
    response_time_trend: str  # improving, stable, degrading
    error_rate: float
    safety_score: float
    user_satisfaction: float
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

class AdaptiveConfig:
    """Adaptive configuration manager with auto-tuning capabilities"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.metrics_path = Path("logs/performance_metrics.json")
        self.health_path = Path("logs/system_health.json")
        
        # Create logs directory
        self.metrics_path.parent.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.current_session = PerformanceMetrics()
        self.historical_metrics = self._load_historical_metrics()
        
        # Performance thresholds for auto-tuning
        self.thresholds = {
            'max_response_time': 5.0,  # seconds
            'min_safety_score': 0.8,
            'max_error_rate': 0.1,
            'target_retrieval_time': 1.0
        }
        
        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.tune_interval = 50  # queries between tune checks
        self.last_tune_query = 0
    
    def _load_historical_metrics(self) -> List[PerformanceMetrics]:
        """Load historical performance metrics"""
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, 'r') as f:
                    data = json.load(f)
                return [PerformanceMetrics(**metrics) for metrics in data]
            except Exception:
                return []
        return []
    
    def _save_metrics(self):
        """Save current metrics to disk"""
        try:
            # Add current session to historical data
            all_metrics = self.historical_metrics + [self.current_session]
            
            # Keep only last 100 sessions
            if len(all_metrics) > 100:
                all_metrics = all_metrics[-100:]
            
            # Convert to JSON-serializable format
            metrics_data = [asdict(metrics) for metrics in all_metrics]
            
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def start_query_timer(self) -> Dict[str, float]:
        """Start timing a query processing"""
        return {
            'start_time': time.time(),
            'rag_start': None,
            'llm_start': None,
            'validation_start': None
        }
    
    def mark_rag_start(self, timer: Dict[str, float]):
        """Mark start of RAG retrieval"""
        timer['rag_start'] = time.time()
    
    def mark_rag_end(self, timer: Dict[str, float]):
        """Mark end of RAG retrieval"""
        if timer['rag_start']:
            self.current_session.rag_retrieval_time += time.time() - timer['rag_start']
    
    def mark_llm_start(self, timer: Dict[str, float]):
        """Mark start of LLM generation"""
        timer['llm_start'] = time.time()
    
    def mark_llm_end(self, timer: Dict[str, float]):
        """Mark end of LLM generation"""
        if timer['llm_start']:
            self.current_session.llm_generation_time += time.time() - timer['llm_start']
    
    def mark_validation_start(self, timer: Dict[str, float]):
        """Mark start of safety validation"""
        timer['validation_start'] = time.time()
    
    def mark_validation_end(self, timer: Dict[str, float]):
        """Mark end of safety validation"""
        if timer['validation_start']:
            self.current_session.safety_validation_time += time.time() - timer['validation_start']
    
    def end_query_timer(self, timer: Dict[str, float], success: bool = True, 
                       emergency_type: str = 'general', safety_result: Optional[str] = None):
        """End query timing and record metrics"""
        
        total_time = time.time() - timer['start_time']
        
        # Update metrics
        self.current_session.query_count += 1
        self.current_session.total_response_time += total_time
        self.current_session.avg_response_time = (
            self.current_session.total_response_time / self.current_session.query_count
        )
        
        if success:
            self.current_session.successful_responses += 1
        else:
            self.current_session.failed_responses += 1
        
        # Track emergency types
        if emergency_type not in self.current_session.emergency_types:
            self.current_session.emergency_types[emergency_type] = 0
        self.current_session.emergency_types[emergency_type] += 1
        
        # Track safety results
        if safety_result == 'blocked':
            self.current_session.safety_blocks += 1
        elif safety_result == 'warning':
            self.current_session.safety_warnings += 1
        
        # Check if auto-tuning should run
        if (self.auto_tune_enabled and 
            self.current_session.query_count - self.last_tune_query >= self.tune_interval):
            self._auto_tune_system()
            self.last_tune_query = self.current_session.query_count
        
        # Save metrics periodically
        if self.current_session.query_count % 10 == 0:
            self._save_metrics()
    
    def get_system_health(self) -> SystemHealth:
        """Assess current system health"""
        
        if self.current_session.query_count == 0:
            return SystemHealth(
                overall_health="unknown",
                response_time_trend="unknown",
                error_rate=0.0,
                safety_score=1.0,
                user_satisfaction=1.0,
                recommendations=["No queries processed yet"]
            )
        
        # Calculate error rate
        error_rate = (
            self.current_session.failed_responses / 
            max(1, self.current_session.query_count)
        )
        
        # Calculate safety score
        safety_issues = self.current_session.safety_blocks + self.current_session.safety_warnings
        safety_score = 1.0 - (safety_issues / max(1, self.current_session.query_count))
        
        # Assess response time trend
        response_time_trend = self._assess_response_time_trend()
        
        # Calculate overall health
        overall_health = self._calculate_overall_health(error_rate, safety_score)
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(error_rate, safety_score)
        
        return SystemHealth(
            overall_health=overall_health,
            response_time_trend=response_time_trend,
            error_rate=error_rate,
            safety_score=safety_score,
            user_satisfaction=0.9,  # Placeholder - could be enhanced with actual feedback
            recommendations=recommendations
        )
    
    def _assess_response_time_trend(self) -> str:
        """Assess if response times are improving, stable, or degrading"""
        
        if len(self.historical_metrics) < 3:
            return "insufficient_data"
        
        recent_times = [m.avg_response_time for m in self.historical_metrics[-3:]]
        
        if recent_times[-1] < recent_times[0] * 0.9:
            return "improving"
        elif recent_times[-1] > recent_times[0] * 1.1:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_overall_health(self, error_rate: float, safety_score: float) -> str:
        """Calculate overall system health rating"""
        
        avg_response_time = self.current_session.avg_response_time
        
        # Score components (0-1 scale)
        response_time_score = max(0, 1 - (avg_response_time / self.thresholds['max_response_time']))
        error_score = max(0, 1 - (error_rate / self.thresholds['max_error_rate']))
        
        # Weighted overall score
        overall_score = (
            response_time_score * 0.3 +
            error_score * 0.3 +
            safety_score * 0.4
        )
        
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.7:
            return "good"
        elif overall_score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _generate_health_recommendations(self, error_rate: float, safety_score: float) -> List[str]:
        """Generate recommendations for system improvement"""
        
        recommendations = []
        
        # Response time recommendations
        if self.current_session.avg_response_time > self.thresholds['max_response_time']:
            recommendations.append("Consider optimizing LLM parameters to reduce response time")
            
            if self.current_session.rag_retrieval_time > self.thresholds['target_retrieval_time']:
                recommendations.append("RAG retrieval is slow - consider index optimization")
        
        # Error rate recommendations
        if error_rate > self.thresholds['max_error_rate']:
            recommendations.append("High error rate detected - check LLM model and configuration")
        
        # Safety recommendations
        if safety_score < self.thresholds['min_safety_score']:
            recommendations.append("Safety score is low - review prompt engineering and validation")
        
        # Usage pattern recommendations
        most_common_emergency = max(
            self.current_session.emergency_types,
            key=self.current_session.emergency_types.get,
            default=None
        )
        
        if most_common_emergency and most_common_emergency != 'general':
            recommendations.append(
                f"Most queries are about {most_common_emergency} - consider specialized optimization"
            )
        
        return recommendations
    
    def _auto_tune_system(self):
        """Automatically tune system parameters based on performance"""
        
        print(f"\n🔧 Auto-tuning system (after {self.current_session.query_count} queries)...")
        
        health = self.get_system_health()
        current_config = self._load_current_config()
        
        changes_made = []
        
        # Tune based on response time
        if self.current_session.avg_response_time > self.thresholds['max_response_time']:
            # Reduce max_tokens if response time is too high
            current_tokens = current_config.get('llm', {}).get('max_tokens', 300)
            if current_tokens > 200:
                new_tokens = max(200, current_tokens - 50)
                self._update_config_value(['llm', 'max_tokens'], new_tokens)
                changes_made.append(f"Reduced max_tokens from {current_tokens} to {new_tokens}")
        
        # Tune based on error rate
        if health.error_rate > 0.1:
            # Increase temperature for more creative responses
            current_temp = current_config.get('llm', {}).get('temperature', 0.55)
            if current_temp < 0.7:
                new_temp = min(0.7, current_temp + 0.1)
                self._update_config_value(['llm', 'temperature'], new_temp)
                changes_made.append(f"Increased temperature from {current_temp} to {new_temp}")
        
        # Tune RAG parameters
        if self.current_session.rag_retrieval_time > self.thresholds['target_retrieval_time']:
            current_k = current_config.get('rag', {}).get('top_k', 5)
            if current_k > 3:
                new_k = current_k - 1
                self._update_config_value(['rag', 'top_k'], new_k)
                changes_made.append(f"Reduced RAG top_k from {current_k} to {new_k}")
        
        if changes_made:
            print("   Auto-tuning changes:")
            for change in changes_made:
                print(f"   • {change}")
        else:
            print("   No auto-tuning changes needed")
    
    def _load_current_config(self) -> Dict[str, Any]:
        """Load current configuration"""
        try:
            from src.utils import load_config
            return load_config()
        except Exception:
            return {}
    
    def _update_config_value(self, path: List[str], value: Any):
        """Update a configuration value (this is a simplified version)"""
        # In a full implementation, this would update the YAML file
        # For now, just print what would be changed
        path_str = '.'.join(path)
        print(f"   Would update {path_str} = {value}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        health = self.get_system_health()
        
        return {
            'session_metrics': asdict(self.current_session),
            'system_health': asdict(health),
            'performance_breakdown': {
                'avg_response_time': self.current_session.avg_response_time,
                'rag_time_ratio': (
                    self.current_session.rag_retrieval_time / 
                    max(1, self.current_session.total_response_time)
                ),
                'llm_time_ratio': (
                    self.current_session.llm_generation_time / 
                    max(1, self.current_session.total_response_time)
                ),
                'validation_time_ratio': (
                    self.current_session.safety_validation_time / 
                    max(1, self.current_session.total_response_time)
                )
            },
            'emergency_distribution': self.current_session.emergency_types
        }
    
    def reset_session_metrics(self):
        """Reset current session metrics (e.g., at start of new session)"""
        self.current_session = PerformanceMetrics()
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """Export metrics to JSON file"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/metrics_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_session': asdict(self.current_session),
            'system_health': asdict(self.get_system_health()),
            'performance_summary': self.get_performance_summary(),
            'historical_metrics': [asdict(m) for m in self.historical_metrics]
        }
        
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath

# Global instance for use throughout the application
adaptive_config = AdaptiveConfig()
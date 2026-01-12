"""LLM response validation and safety checks for medical emergency guidance"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"

@dataclass
class ValidationResult:
    """Result of response validation"""
    safety_level: SafetyLevel
    score: float  # 0-1, higher is safer
    issues: List[str]
    suggestions: List[str]
    modified_response: Optional[str] = None

class MedicalResponseValidator:
    """Validates LLM responses for medical safety and accuracy"""
    
    def __init__(self):
        # Dangerous advice patterns that should never be suggested
        self.dangerous_patterns = {
            'remove_embedded_objects': [
                r'remove.*(?:knife|glass|metal|object|shard)',
                r'pull.*out.*(?:knife|glass|metal|object)',
                r'extract.*(?:embedded|stuck|impaled)'
            ],
            'move_spinal_injury': [
                r'move.*(?:head|neck|spine|back).*injur',
                r'turn.*head.*injur',
                r'lift.*spinal'
            ],
            'induce_vomiting': [
                r'make.*vomit',
                r'induce.*vomiting',
                r'force.*throw up'
            ],
            'inappropriate_medications': [
                r'give.*(?:aspirin|ibuprofen).*(?:child|infant|baby)',
                r'administer.*medication.*without',
                r'inject.*insulin'
            ],
            'delay_emergency_services': [
                r'wait.*before.*calling',
                r'don\'?t.*call.*911',
                r'avoid.*emergency.*services'
            ]
        }
        
        # Required safety elements for different emergency types
        self.required_safety_elements = {
            'bleeding': [
                'direct pressure',
                'call emergency services',
                'do not remove embedded'
            ],
            'cardiac': [
                'call emergency services',
                'check pulse',
                'begin cpr'
            ],
            'respiratory': [
                'clear airway',
                'call emergency services',
                'monitor breathing'
            ],
            'neurological': [
                'call emergency services',
                'protect spine',
                'monitor consciousness'
            ],
            'poisoning': [
                'call poison control',
                'do not induce vomiting',
                'emergency services'
            ]
        }
        
        # Warning patterns that need clarification
        self.warning_patterns = {
            'medication_advice': [
                r'take.*(?:aspirin|acetaminophen|ibuprofen)',
                r'administer.*medication',
                r'give.*pills'
            ],
            'diagnosis_language': [
                r'you have.*(?:heart attack|stroke|fracture)',
                r'this is definitely',
                r'diagnosed with'
            ],
            'absolute_statements': [
                r'always do',
                r'never happens',
                r'guaranteed to work'
            ],
            'time_specific_advice': [
                r'within.*(?:minutes|hours).*will',
                r'after.*(?:time).*should'
            ]
        }
        
        # Emergency service contact requirements
        self.ems_requirements = {
            'immediate': [
                'unconscious', 'not breathing', 'no pulse', 'severe bleeding',
                'chest pain', 'stroke', 'seizure', 'choking'
            ],
            'urgent': [
                'difficulty breathing', 'severe pain', 'head injury',
                'suspected fracture', 'burns'
            ]
        }
        
        # Medical disclaimers and safety language
        self.safety_language = {
            'emergency_disclaimer': "This guidance does not replace professional medical care. Call emergency services immediately if the situation is life-threatening.",
            'medication_warning': "Do not give medications unless specifically trained and authorized.",
            'assessment_limitation': "This assessment is based on limited information. Professional medical evaluation is recommended.",
            'general_disclaimer': "If you are unsure about any step or if the condition worsens, seek immediate medical attention."
        }
    
    def validate_response(self, response: str, query: str = "", 
                         emergency_context: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive validation of medical emergency response
        
        Args:
            response: LLM generated response
            query: Original user query
            emergency_context: Detected emergency context
            
        Returns:
            ValidationResult with safety assessment
        """
        
        issues = []
        suggestions = []
        safety_score = 1.0
        
        # Check for dangerous patterns
        dangerous_issues = self._check_dangerous_patterns(response)
        if dangerous_issues:
            return ValidationResult(
                safety_level=SafetyLevel.BLOCKED,
                score=0.0,
                issues=dangerous_issues,
                suggestions=["Response contains dangerous medical advice and has been blocked."]
            )
        
        # Check for warning patterns
        warning_issues = self._check_warning_patterns(response)
        if warning_issues:
            issues.extend(warning_issues)
            safety_score -= 0.2
        
        # Check for required safety elements
        if emergency_context:
            missing_elements = self._check_required_elements(response, emergency_context)
            if missing_elements:
                issues.extend([f"Missing required element: {elem}" for elem in missing_elements])
                safety_score -= 0.1 * len(missing_elements)
        
        # Check for emergency service guidance
        ems_issues = self._check_emergency_service_guidance(response, query)
        if ems_issues:
            issues.extend(ems_issues)
            safety_score -= 0.15
        
        # Check for appropriate disclaimers
        disclaimer_issues = self._check_disclaimers(response)
        if disclaimer_issues:
            issues.extend(disclaimer_issues)
            safety_score -= 0.1
        
        # Check response structure and clarity
        structure_issues = self._check_response_structure(response)
        if structure_issues:
            issues.extend(structure_issues)
            safety_score -= 0.05
        
        # Generate suggestions for improvement
        suggestions = self._generate_suggestions(issues, emergency_context)
        
        # Determine safety level
        if safety_score >= 0.8:
            safety_level = SafetyLevel.SAFE
        elif safety_score >= 0.6:
            safety_level = SafetyLevel.WARNING
        else:
            safety_level = SafetyLevel.DANGEROUS
        
        # Generate modified response if needed
        modified_response = None
        if safety_level in [SafetyLevel.WARNING, SafetyLevel.DANGEROUS]:
            modified_response = self._enhance_response_safety(response, issues, emergency_context)
        
        return ValidationResult(
            safety_level=safety_level,
            score=max(0.0, safety_score),
            issues=issues,
            suggestions=suggestions,
            modified_response=modified_response
        )
    
    def _check_dangerous_patterns(self, response: str) -> List[str]:
        """Check for dangerous medical advice patterns"""
        
        issues = []
        response_lower = response.lower()
        
        for category, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    issues.append(f"Dangerous advice detected: {category}")
                    break
        
        return issues
    
    def _check_warning_patterns(self, response: str) -> List[str]:
        """Check for patterns that need warning or clarification"""
        
        issues = []
        response_lower = response.lower()
        
        for category, patterns in self.warning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    issues.append(f"Warning pattern: {category}")
                    break
        
        return issues
    
    def _check_required_elements(self, response: str, emergency_context: Dict) -> List[str]:
        """Check for required safety elements based on emergency type"""
        
        emergency_type = emergency_context.get('emergency_type', 'general')
        required_elements = self.required_safety_elements.get(emergency_type, [])
        
        missing = []
        response_lower = response.lower()
        
        for element in required_elements:
            if element.lower() not in response_lower:
                missing.append(element)
        
        return missing
    
    def _check_emergency_service_guidance(self, response: str, query: str) -> List[str]:
        """Check if emergency service guidance is appropriate"""
        
        issues = []
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Check if immediate EMS keywords are in query but not addressed in response
        immediate_keywords = self.ems_requirements['immediate']
        urgent_keywords = self.ems_requirements['urgent']
        
        has_immediate = any(keyword in query_lower for keyword in immediate_keywords)
        has_urgent = any(keyword in query_lower for keyword in urgent_keywords)
        
        ems_mentioned = any(phrase in response_lower for phrase in [
            'call 911', 'emergency services', 'call emergency', 'ambulance'
        ])
        
        if (has_immediate or has_urgent) and not ems_mentioned:
            issues.append("Missing emergency services guidance for serious condition")
        
        return issues
    
    def _check_disclaimers(self, response: str) -> List[str]:
        """Check for appropriate medical disclaimers"""
        
        issues = []
        response_lower = response.lower()
        
        # Check for general disclaimer elements
        disclaimer_elements = [
            'professional medical', 'emergency services', 'seek medical',
            'call 911', 'not a substitute', 'medical attention'
        ]
        
        has_disclaimer = any(element in response_lower for element in disclaimer_elements)
        
        if not has_disclaimer and len(response) > 200:  # Only for substantial responses
            issues.append("Missing medical disclaimer")
        
        return issues
    
    def _check_response_structure(self, response: str) -> List[str]:
        """Check response structure and clarity"""
        
        issues = []
        
        # Check for numbered steps
        if len(response) > 100 and not re.search(r'\d+\.', response):
            issues.append("Response lacks clear numbered steps")
        
        # Check for excessive length
        if len(response) > 2000:
            issues.append("Response may be too long for emergency situation")
        
        # Check for multiple questions at end
        question_marks = response.count('?')
        if question_marks > 2:
            issues.append("Too many questions - should focus on one clarification")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str], 
                            emergency_context: Optional[Dict]) -> List[str]:
        """Generate specific suggestions for improvement"""
        
        suggestions = []
        
        for issue in issues:
            if "Missing required element" in issue:
                suggestions.append("Add essential safety steps for this emergency type")
            elif "Missing emergency services" in issue:
                suggestions.append("Include clear guidance on when to call 911/emergency services")
            elif "Missing medical disclaimer" in issue:
                suggestions.append("Add disclaimer about seeking professional medical care")
            elif "Warning pattern" in issue:
                suggestions.append("Clarify medical advice with appropriate qualifications")
            elif "lacks clear numbered steps" in issue:
                suggestions.append("Structure response with clear, numbered action steps")
        
        # Emergency-specific suggestions
        if emergency_context:
            emergency_type = emergency_context.get('emergency_type')
            if emergency_type == 'bleeding':
                suggestions.append("Emphasize direct pressure and avoiding removal of embedded objects")
            elif emergency_type == 'cardiac':
                suggestions.append("Prioritize calling emergency services and CPR readiness")
            elif emergency_type == 'respiratory':
                suggestions.append("Focus on airway management and breathing assessment")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _enhance_response_safety(self, response: str, issues: List[str], 
                               emergency_context: Optional[Dict]) -> str:
        """Enhance response safety by adding missing elements"""
        
        enhanced = response
        
        # Add emergency services guidance if missing
        if any("emergency services" in issue for issue in issues):
            ems_guidance = "\n\n⚠️ WHEN TO CALL EMERGENCY SERVICES: Call 911 immediately if the person is unconscious, not breathing normally, has severe bleeding, or if you are unsure about the severity."
            enhanced += ems_guidance
        
        # Add disclaimer if missing
        if any("disclaimer" in issue for issue in issues):
            disclaimer = f"\n\n{self.safety_language['emergency_disclaimer']}"
            enhanced += disclaimer
        
        # Add emergency-specific safety reminders
        if emergency_context:
            emergency_type = emergency_context.get('emergency_type')
            if emergency_type == 'bleeding' and 'embedded' not in enhanced.lower():
                enhanced += "\n\n⚠️ IMPORTANT: Never remove embedded objects like knives, glass, or metal from wounds."
            elif emergency_type == 'neurological' and 'spine' not in enhanced.lower():
                enhanced += "\n\n⚠️ SPINE SAFETY: If head/neck injury is suspected, avoid moving the person unless absolutely necessary."
        
        return enhanced
    
    def get_safety_score_explanation(self, validation_result: ValidationResult) -> str:
        """Get human-readable explanation of safety score"""
        
        score = validation_result.score
        level = validation_result.safety_level
        
        if level == SafetyLevel.SAFE:
            return f"Response is medically safe (score: {score:.2f}). No significant safety concerns identified."
        elif level == SafetyLevel.WARNING:
            return f"Response has minor safety concerns (score: {score:.2f}). Review suggested improvements."
        elif level == SafetyLevel.DANGEROUS:
            return f"Response has significant safety issues (score: {score:.2f}). Major revision needed."
        else:  # BLOCKED
            return "Response contains dangerous medical advice and has been blocked for safety."

def validate_medical_response(response: str, query: str = "", 
                            emergency_context: Optional[Dict] = None) -> ValidationResult:
    """
    Convenience function for validating medical responses
    
    Args:
        response: LLM generated response
        query: Original user query
        emergency_context: Emergency context from prompt manager
        
    Returns:
        ValidationResult with safety assessment
    """
    validator = MedicalResponseValidator()
    return validator.validate_response(response, query, emergency_context)
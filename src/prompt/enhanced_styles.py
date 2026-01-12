"""Enhanced prompt engineering with context-aware formatting and emergency-specific templates"""

from typing import Optional, Dict, Any, List
import re

class EmergencyPromptManager:
    """Manages emergency-specific prompt templates and formatting"""
    
    def __init__(self):
        self.emergency_templates = {
            'bleeding': {
                'priority_actions': [
                    "Apply direct pressure immediately",
                    "Control bleeding before assessment",
                    "Check for embedded objects (do NOT remove)",
                    "Monitor for shock signs"
                ],
                'critical_warnings': [
                    "Do NOT remove embedded objects",
                    "If bleeding soaks through bandage, add more layers on top",
                    "Call emergency services if bleeding cannot be controlled"
                ],
                'assessment_focus': 'bleeding severity, location, type (arterial/venous)'
            },
            'cardiac': {
                'priority_actions': [
                    "Check responsiveness and breathing",
                    "Call emergency services immediately",
                    "Begin CPR if no pulse",
                    "Use AED if available"
                ],
                'critical_warnings': [
                    "Time is critical for heart attacks",
                    "Do NOT delay emergency services",
                    "Continue CPR until help arrives"
                ],
                'assessment_focus': 'consciousness, pulse, breathing, chest pain'
            },
            'respiratory': {
                'priority_actions': [
                    "Ensure airway is clear",
                    "Position for optimal breathing",
                    "Monitor breathing rate and quality",
                    "Prepare for airway obstruction"
                ],
                'critical_warnings': [
                    "Airway obstruction can be rapidly fatal",
                    "Do NOT leave patient alone if breathing difficulty",
                    "Encourage calm, controlled breathing"
                ],
                'assessment_focus': 'airway patency, breathing rate, oxygen saturation'
            },
            'neurological': {
                'priority_actions': [
                    "Assess consciousness level",
                    "Protect cervical spine if trauma",
                    "Check pupils and motor responses",
                    "Monitor for seizure activity"
                ],
                'critical_warnings': [
                    "Do NOT move patient if spinal injury suspected",
                    "Note time of onset for stroke symptoms",
                    "Protect from injury during seizures"
                ],
                'assessment_focus': 'consciousness, neurological deficits, spine stability'
            },
            'trauma': {
                'priority_actions': [
                    "Ensure scene safety first",
                    "Control major bleeding",
                    "Immobilize suspected fractures",
                    "Assess for multiple injuries"
                ],
                'critical_warnings': [
                    "Check for internal injuries",
                    "Do NOT move unless absolutely necessary",
                    "Assume spinal injury until proven otherwise"
                ],
                'assessment_focus': 'mechanism of injury, visible trauma, vital signs'
            }
        }
        
        self.severity_modifiers = {
            'critical': {
                'urgency_level': 'IMMEDIATE',
                'tone_adjustment': 'directive and urgent',
                'time_pressure': 'every second counts',
                'ems_emphasis': 'Call emergency services NOW'
            },
            'moderate': {
                'urgency_level': 'URGENT',
                'tone_adjustment': 'calm but decisive',
                'time_pressure': 'prompt action needed',
                'ems_emphasis': 'Call emergency services promptly'
            },
            'mild': {
                'urgency_level': 'STANDARD',
                'tone_adjustment': 'reassuring and methodical',
                'time_pressure': 'careful assessment time',
                'ems_emphasis': 'Consider emergency services if worsening'
            }
        }
    
    def get_emergency_context(self, query: str, rag_results: List[Dict]) -> Dict[str, Any]:
        """Extract emergency context from query and RAG results"""
        
        # Detect emergency type
        emergency_type = self._detect_emergency_type(query, rag_results)
        
        # Detect severity indicators
        severity = self._detect_severity_level(query, rag_results)
        
        # Extract specific details
        details = self._extract_emergency_details(query)
        
        return {
            'emergency_type': emergency_type,
            'severity': severity,
            'details': details,
            'template': self.emergency_templates.get(emergency_type, {}),
            'severity_modifier': self.severity_modifiers.get(severity, self.severity_modifiers['moderate'])
        }
    
    def _detect_emergency_type(self, query: str, rag_results: List[Dict]) -> str:
        """Detect primary emergency type from query and context"""
        
        query_lower = query.lower()
        
        # Check query for emergency keywords
        type_scores = {}
        
        patterns = {
            'bleeding': [
                'bleed', 'blood', 'hemorrhage', 'cut', 'wound', 'laceration',
                'arterial', 'venous', 'tourniquet', 'pressure'
            ],
            'cardiac': [
                'heart', 'cardiac', 'chest pain', 'heart attack', 'mi',
                'arrhythmia', 'pulse', 'cpr', 'defibrillator'
            ],
            'respiratory': [
                'breath', 'airway', 'chok', 'asthma', 'lung', 'oxygen',
                'respiratory', 'cough', 'wheez'
            ],
            'neurological': [
                'stroke', 'seizure', 'unconscious', 'head', 'brain',
                'spinal', 'neuro', 'concussion', 'consciousness'
            ],
            'trauma': [
                'injury', 'accident', 'fracture', 'break', 'burn',
                'trauma', 'wound', 'fall', 'crash'
            ]
        }
        
        for emergency_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            type_scores[emergency_type] = score
        
        # Also check RAG results for emergency type indicators
        if rag_results:
            for result in rag_results:
                metadata = result.get('metadata', {})
                result_type = metadata.get('emergency_type', '')
                if result_type in type_scores:
                    type_scores[result_type] += 2  # Weight RAG results higher
        
        # Return highest scoring type or 'general' if no clear match
        if type_scores and max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return 'general'
    
    def _detect_severity_level(self, query: str, rag_results: List[Dict]) -> str:
        """Detect severity level from query and context"""
        
        query_lower = query.lower()
        
        severity_keywords = {
            'critical': [
                'severe', 'massive', 'major', 'life-threatening', 'critical',
                'emergency', 'urgent', 'profuse', 'uncontrolled', 'heavy'
            ],
            'moderate': [
                'moderate', 'significant', 'concerning', 'substantial',
                'noticeable', 'persistent'
            ],
            'mild': [
                'mild', 'minor', 'slight', 'small', 'limited', 'light'
            ]
        }
        
        severity_scores = {}
        for severity, keywords in severity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            severity_scores[severity] = score
        
        # Check RAG results for severity indicators
        if rag_results:
            for result in rag_results:
                metadata = result.get('metadata', {})
                result_severity = metadata.get('severity_level', '')
                if result_severity in severity_scores:
                    severity_scores[result_severity] += 1
        
        # Default to moderate if no clear indicators
        if severity_scores and max(severity_scores.values()) > 0:
            return max(severity_scores, key=severity_scores.get)
        
        return 'moderate'
    
    def _extract_emergency_details(self, query: str) -> Dict[str, Any]:
        """Extract specific emergency details from query"""
        
        details = {
            'body_part': None,
            'age_group': None,
            'consciousness': None,
            'time_factor': None
        }
        
        query_lower = query.lower()
        
        # Body parts
        body_parts = {
            'head': ['head', 'skull', 'brain'],
            'neck': ['neck', 'throat'],
            'chest': ['chest', 'ribs', 'sternum'],
            'abdomen': ['abdomen', 'stomach', 'belly'],
            'arm': ['arm', 'shoulder', 'elbow', 'wrist', 'hand'],
            'leg': ['leg', 'thigh', 'knee', 'ankle', 'foot'],
            'back': ['back', 'spine', 'spinal']
        }
        
        for part, keywords in body_parts.items():
            if any(keyword in query_lower for keyword in keywords):
                details['body_part'] = part
                break
        
        # Age groups
        if any(word in query_lower for word in ['child', 'kid', 'baby', 'infant', 'pediatric']):
            details['age_group'] = 'pediatric'
        elif any(word in query_lower for word in ['elderly', 'senior', 'old', 'geriatric']):
            details['age_group'] = 'geriatric'
        
        # Consciousness level
        if any(word in query_lower for word in ['unconscious', 'unresponsive', 'passed out']):
            details['consciousness'] = 'unconscious'
        elif any(word in query_lower for word in ['confused', 'disoriented', 'altered']):
            details['consciousness'] = 'altered'
        
        # Time factors
        if any(word in query_lower for word in ['sudden', 'suddenly', 'immediate']):
            details['time_factor'] = 'acute'
        elif any(word in query_lower for word in ['gradual', 'slowly', 'over time']):
            details['time_factor'] = 'gradual'
        
        return details

def _style_guidelines(style: str) -> str:
    """Get style guidelines for different communication styles"""
    style = (style or "warm").lower()
    if style == "coach":
        return (
            "Tone: supportive coach; confident, motivating, plain language.\n"
            "Style: short sentences; direct commands; 2nd person ('you'). Avoid jargon."
        )
    if style in ("pro", "professional"):
        return (
            "Tone: professional first-aid instructor; precise and calm.\n"
            "Style: concise, technical terms when needed with brief explanations."
        )
    # default warm
    return (
        "Tone: warm, calm, and reassuring.\n"
        "Style: friendly plain language; short, direct steps; avoid jargon unless explained."
    )

def build_enhanced_prompt(
    user_query: str,
    rag_context: str = "",
    decision_text: str = "",
    style: str = "warm",
    rag_results: Optional[List[Dict]] = None
) -> str:
    """
    Build an enhanced prompt with emergency-specific formatting and context awareness.
    """
    
    # Initialize prompt manager
    prompt_manager = EmergencyPromptManager()
    
    # Extract emergency context
    emergency_context = prompt_manager.get_emergency_context(user_query, rag_results or [])
    
    # Get style guidelines
    style_text = _style_guidelines(style)
    
    # Build emergency-specific instructions
    emergency_instructions = _build_emergency_instructions(emergency_context)
    
    # Build references block
    refs_block = ""
    if rag_context and rag_context.strip():
        refs_block = f"MEDICAL REFERENCES (authoritative excerpts to base your answer on):\n{rag_context.strip()}\n"
    
    # Build protocol block
    protocol_block = ""
    if decision_text and decision_text.strip():
        protocol_block = (
            "PROTOCOL (follow as primary instructions; use exact steps when applicable):\n"
            f"{decision_text.strip()}\n"
        )
    
    # Emergency-specific rules
    emergency_rules = _build_emergency_rules(emergency_context)
    
    # Main prompt structure
    prompt = f"""You are an advanced emergency medical assistant specializing in {emergency_context['emergency_type']} emergencies.
Current situation assessment: {emergency_context['severity'].upper()} priority - {emergency_context['severity_modifier']['time_pressure']}.

{style_text}

{emergency_instructions}

{emergency_rules}

USER EMERGENCY:
{user_query.strip()}

{protocol_block}{refs_block}
Provide immediate, actionable guidance following the emergency-specific requirements above."""
    
    return prompt

def _build_emergency_instructions(emergency_context: Dict[str, Any]) -> str:
    """Build emergency-specific instructions"""
    
    template = emergency_context.get('template', {})
    severity_modifier = emergency_context.get('severity_modifier', {})
    
    instructions = [
        f"EMERGENCY TYPE: {emergency_context['emergency_type'].upper()}",
        f"URGENCY LEVEL: {severity_modifier.get('urgency_level', 'STANDARD')}",
        f"COMMUNICATION TONE: {severity_modifier.get('tone_adjustment', 'calm and reassuring')}"
    ]
    
    if template.get('priority_actions'):
        instructions.append("\nPRIORITY ACTIONS FOR THIS EMERGENCY:")
        for action in template['priority_actions']:
            instructions.append(f"• {action}")
    
    if template.get('critical_warnings'):
        instructions.append("\nCRITICAL WARNINGS:")
        for warning in template['critical_warnings']:
            instructions.append(f"⚠️  {warning}")
    
    if template.get('assessment_focus'):
        instructions.append(f"\nASSESSMENT FOCUS: {template['assessment_focus']}")
    
    return "\n".join(instructions)

def _build_emergency_rules(emergency_context: Dict[str, Any]) -> str:
    """Build emergency-specific response rules"""
    
    severity = emergency_context['severity']
    emergency_type = emergency_context['emergency_type']
    severity_modifier = emergency_context['severity_modifier']
    
    base_rules = [
        "RESPONSE REQUIREMENTS:",
        "- Start with ONE brief reassurance sentence appropriate to the urgency level",
        "- Provide 4-8 numbered action steps in order of priority",
        "- Use present tense, imperative mood for immediate actions",
        "- Include timing guidance where critical (e.g., 'within 5 minutes')",
    ]
    
    # Severity-specific rules
    if severity == 'critical':
        base_rules.extend([
            "- EMPHASIZE immediate life-saving actions first",
            "- Include clear 'CALL EMERGENCY SERVICES NOW' instruction",
            "- Mention time-critical nature when relevant",
            "- Prioritize airway, breathing, circulation"
        ])
    elif severity == 'moderate':
        base_rules.extend([
            "- Balance immediate actions with assessment steps",
            "- Include when to call emergency services",
            "- Provide clear monitoring instructions"
        ])
    else:  # mild
        base_rules.extend([
            "- Focus on proper assessment and gradual intervention",
            "- Include when to seek medical attention",
            "- Emphasize prevention of escalation"
        ])
    
    # Emergency-type specific rules
    if emergency_type == 'bleeding':
        base_rules.extend([
            "- NEVER suggest removing embedded objects",
            "- Emphasize direct pressure technique",
            "- Include shock prevention measures"
        ])
    elif emergency_type == 'cardiac':
        base_rules.extend([
            "- Prioritize CPR readiness",
            "- Mention AED use if available",
            "- Emphasize continuous monitoring"
        ])
    elif emergency_type == 'respiratory':
        base_rules.extend([
            "- Focus on airway management",
            "- Include positioning instructions",
            "- Monitor breathing continuously"
        ])
    
    base_rules.extend([
        "- End with exactly ONE question for clarification or status check",
        "- Include 'When to call emergency services' section if not already covered",
        "- Do NOT speculate beyond provided references/protocols"
    ])
    
    return "\n".join(base_rules)

# Legacy function for backward compatibility
def build_prompt(
    user_query: str,
    rag_context: str = "",
    decision_text: str = "",
    style: str = "warm",
) -> str:
    """Legacy function - redirects to enhanced prompt builder"""
    return build_enhanced_prompt(user_query, rag_context, decision_text, style)
"""
Prompt Generation Module

This module handles creating comprehensive prompts for LLM analysis
by combining issue descriptions, filtered logs, and relevant context.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts that can be generated."""
    ANALYSIS = "analysis"
    DEBUGGING = "debugging"
    ROOT_CAUSE = "root_cause"
    SOLUTION = "solution"


@dataclass
class AnalysisData:
    """Container for all analysis data."""
    issue_description: str
    extracted_keywords: List[str]
    filtered_logs: str
    context_description: str
    log_file_path: str
    analysis_date_range: Optional[str] = None


class PromptGenerator:
    """Generates comprehensive prompts for LLM analysis."""
    
    def __init__(self):
        """Initialize the prompt generator."""
        self.templates = {
            PromptType.ANALYSIS: self._get_analysis_template(),
            PromptType.DEBUGGING: self._get_debugging_template(),
            PromptType.ROOT_CAUSE: self._get_root_cause_template(),
            PromptType.SOLUTION: self._get_solution_template()
        }
    
    def generate_prompt(self, analysis_data: AnalysisData, 
                      prompt_type: PromptType = PromptType.ANALYSIS) -> str:
        """
        Generate a comprehensive prompt for LLM analysis.
        
        Args:
            analysis_data: Container with all analysis data
            prompt_type: Type of prompt to generate
            
        Returns:
            Formatted prompt string
        """
        template = self.templates[prompt_type]
        
        # Format the prompt with the analysis data
        prompt = template.format(
            issue_description=analysis_data.issue_description,
            keywords=", ".join(analysis_data.extracted_keywords),
            filtered_logs=analysis_data.filtered_logs,
            context_description=analysis_data.context_description,
            log_file_path=analysis_data.log_file_path,
            date_range=analysis_data.analysis_date_range or "Not specified"
        )
        
        return prompt
    
    def _get_analysis_template(self) -> str:
        """Get the analysis prompt template."""
        return """
# Log Analysis Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## Analysis Context
{context_description}

## Log File Information
- **File Path:** {log_file_path}
- **Date Range:** {date_range}

## Filtered Log Entries
{filtered_logs}

## Analysis Instructions

Please analyze the provided log entries in the context of the described issue. Focus on:

1. **Pattern Recognition**: Identify recurring patterns, errors, or anomalies in the logs
2. **Timeline Analysis**: Look for temporal patterns and sequence of events
3. **Error Correlation**: Connect log entries to the described issue
4. **Context Integration**: Use the provided context to understand technical implications
5. **Root Cause Indicators**: Identify potential root causes based on log evidence

## Expected Output Format

Please provide your analysis in the following structure:

### Summary
Brief overview of findings

### Key Findings
- List of significant observations
- Error patterns identified
- Timeline of events

### Potential Root Causes
- Primary suspected causes
- Supporting evidence from logs
- Technical explanations

### Recommendations
- Immediate actions to take
- Further investigation needed
- Preventive measures

### Confidence Level
Rate your confidence in the analysis (1-10) and explain any limitations.
"""
    
    def _get_debugging_template(self) -> str:
        """Get the debugging prompt template."""
        return """
# Debugging Analysis Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## Technical Context
{context_description}

## Log File Information
- **File Path:** {log_file_path}
- **Date Range:** {date_range}

## Relevant Log Entries
{filtered_logs}

## Debugging Instructions

Please help debug the described issue using the provided log entries. Focus on:

1. **Error Identification**: Identify specific errors, exceptions, or failures
2. **Stack Trace Analysis**: Analyze any stack traces or error details
3. **State Analysis**: Determine the application state when issues occurred
4. **Dependency Issues**: Look for external service or dependency problems
5. **Configuration Problems**: Identify potential configuration issues

## Expected Output Format

### Error Summary
- List of all errors found
- Severity levels
- Frequency of occurrence

### Debugging Steps
1. **Immediate Actions**
   - Steps to reproduce the issue
   - Data to collect
   - Logs to monitor

2. **Investigation Path**
   - Areas to investigate
   - Tools to use
   - Metrics to check

3. **Resolution Strategy**
   - Potential fixes
   - Testing approach
   - Rollback plan

### Debugging Checklist
- [ ] Error reproduction steps
- [ ] Environment verification
- [ ] Configuration validation
- [ ] Dependency health check
- [ ] Performance metrics review
"""
    
    def _get_root_cause_template(self) -> str:
        """Get the root cause analysis template."""
        return """
# Root Cause Analysis Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## System Context
{context_description}

## Log File Information
- **File Path:** {log_file_path}
- **Date Range:** {date_range}

## Log Evidence
{filtered_logs}

## Root Cause Analysis Instructions

Perform a systematic root cause analysis using the provided evidence. Follow this approach:

1. **Problem Definition**: Clearly define what went wrong
2. **Timeline Construction**: Build a chronological sequence of events
3. **Causal Chain Analysis**: Identify cause-and-effect relationships
4. **Evidence Evaluation**: Assess the strength of evidence for each potential cause
5. **Root Cause Identification**: Determine the fundamental cause

## Analysis Framework

### Problem Statement
- What exactly happened?
- When did it occur?
- What was the impact?

### Timeline of Events
Create a chronological timeline of relevant events from the logs.

### Potential Causes Analysis
For each potential cause, evaluate:
- **Evidence Strength**: How strong is the supporting evidence?
- **Likelihood**: How likely is this cause?
- **Impact**: What would be the impact if this were the cause?

### Root Cause Determination
- **Primary Root Cause**: The fundamental cause
- **Contributing Factors**: Secondary causes that enabled the primary cause
- **Evidence Summary**: Key evidence supporting the conclusion

### Prevention Strategy
- **Immediate Fixes**: Actions to prevent immediate recurrence
- **System Improvements**: Long-term improvements to prevent similar issues
- **Monitoring Enhancements**: Better detection and alerting
"""
    
    def _get_solution_template(self) -> str:
        """Get the solution-focused template."""
        return """
# Solution Development Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## Technical Context
{context_description}

## Log File Information
- **File Path:** {log_file_path}
- **Date Range:** {date_range}

## Relevant Log Data
{filtered_logs}

## Solution Development Instructions

Develop a comprehensive solution for the described issue. Consider:

1. **Problem Understanding**: Ensure complete understanding of the issue
2. **Solution Options**: Generate multiple solution approaches
3. **Implementation Planning**: Create detailed implementation steps
4. **Risk Assessment**: Evaluate risks and mitigation strategies
5. **Testing Strategy**: Define testing and validation approaches

## Solution Framework

### Problem Analysis
- **Core Issue**: What needs to be fixed?
- **Impact Assessment**: What are the consequences of not fixing?
- **Constraints**: What limitations exist?

### Solution Options
Present 2-3 different solution approaches:

#### Option 1: [Name]
- **Description**: Brief description
- **Pros**: Advantages
- **Cons**: Disadvantages
- **Effort**: Implementation effort estimate
- **Risk**: Risk level

#### Option 2: [Name]
- **Description**: Brief description
- **Pros**: Advantages
- **Cons**: Disadvantages
- **Effort**: Implementation effort estimate
- **Risk**: Risk level

### Recommended Solution
- **Selected Option**: Which option is recommended and why
- **Implementation Plan**: Step-by-step implementation
- **Timeline**: Estimated timeline
- **Resources**: Required resources and skills

### Implementation Details
- **Code Changes**: Specific code modifications needed
- **Configuration Changes**: Configuration updates required
- **Deployment Steps**: Deployment and rollout plan
- **Testing Plan**: How to test the solution

### Success Criteria
- **Definition of Done**: How to know the solution is complete
- **Metrics**: How to measure success
- **Monitoring**: What to monitor post-implementation
"""
    
    def generate_custom_prompt(self, analysis_data: AnalysisData, 
                              custom_instructions: str) -> str:
        """
        Generate a custom prompt with user-defined instructions.
        
        Args:
            analysis_data: Container with all analysis data
            custom_instructions: Custom instructions to include
            
        Returns:
            Formatted prompt string
        """
        base_template = """
# Custom Log Analysis Request

## Issue Description
{issue_description}

## Extracted Keywords
{keywords}

## Context Information
{context_description}

## Log File Information
- **File Path:** {log_file_path}
- **Date Range:** {date_range}

## Filtered Log Entries
{filtered_logs}

## Custom Instructions
{custom_instructions}

Please provide your analysis following the custom instructions above.
"""
        
        return base_template.format(
            issue_description=analysis_data.issue_description,
            keywords=", ".join(analysis_data.extracted_keywords),
            context_description=analysis_data.context_description,
            log_file_path=analysis_data.log_file_path,
            date_range=analysis_data.analysis_date_range or "Not specified",
            filtered_logs=analysis_data.filtered_logs,
            custom_instructions=custom_instructions
        )
    
    def get_prompt_statistics(self, analysis_data: AnalysisData) -> Dict[str, int]:
        """
        Get statistics about the generated prompt.
        
        Args:
            analysis_data: Container with all analysis data
            
        Returns:
            Dictionary with prompt statistics
        """
        prompt = self.generate_prompt(analysis_data)
        
        return {
            "total_characters": len(prompt),
            "total_words": len(prompt.split()),
            "total_lines": len(prompt.split('\n')),
            "issue_description_length": len(analysis_data.issue_description),
            "keywords_count": len(analysis_data.extracted_keywords),
            "filtered_logs_length": len(analysis_data.filtered_logs),
            "context_length": len(analysis_data.context_description)
        }


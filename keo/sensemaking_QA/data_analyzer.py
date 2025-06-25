"""
Aviation Maintenance Data Analyzer - Updated for Data Separation
Analyzes OMIn datasets to extract patterns and themes for sensemaking question generation
UPDATED: Excludes FAA_sample_100.csv and sampled maintenance files (used for KG construction)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter, defaultdict
import json
import os

class AviationDataAnalyzer:
    def __init__(self, data_paths: Dict[str, str]):
        """
        Initialize the analyzer with paths to datasets for sensemaking question generation
        
        Args:
            data_paths: Dictionary with keys for datasets to use for question generation
                       (excludes files used for KG construction)
        """
        self.data_paths = data_paths
        self.datasets = {}
        self.analysis_results = {}
        
    def load_datasets(self) -> None:
        """Load datasets for sensemaking question generation (excluding KG construction files)"""
        try:
            # Load maintenance data remaining after sampling (for question generation)
            if 'maintenance_remaining' in self.data_paths:
                self.datasets['maintenance_remaining'] = pd.read_csv(self.data_paths['maintenance_remaining'])
                print(f"Loaded maintenance remaining: {len(self.datasets['maintenance_remaining'])} records")
            
            # Load aircraft annotation data (always used for questions)
            if 'aircraft_annotation' in self.data_paths:
                self.datasets['aircraft_annotation'] = pd.read_csv(self.data_paths['aircraft_annotation'])
                print(f"Loaded aircraft annotation: {len(self.datasets['aircraft_annotation'])} records")
                
            print(f"Total datasets loaded for question generation: {len(self.datasets)}")
            print("Note: FAA_sample_100.csv and 5 maintenance sample files excluded (used for KG construction)")
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise
    
    def analyze_failure_patterns(self) -> Dict:
        """Analyze failure patterns across available datasets"""
        patterns = {
            'common_failures': {},
            'failure_categories': {},
            'component_issues': {},
            'maintenance_types': {},
            'severity_patterns': {}
        }
        
        # Analyze maintenance remaining data (not used for KG)
        if 'maintenance_remaining' in self.datasets:
            maint_data = self.datasets['maintenance_remaining']
            
            # Analyze maintenance narratives
            if 'c119' in maint_data.columns:
                maint_narratives = maint_data['c119'].dropna()
                patterns['maintenance_keywords'] = self._extract_keywords_from_narratives(maint_narratives)
                patterns['maintenance_categories'] = self._categorize_maintenance_issues(maint_narratives)
        
        # Analyze aircraft annotation data
        if 'aircraft_annotation' in self.datasets:
            annotation_data = self.datasets['aircraft_annotation']
            
            if 'PROBLEM' in annotation_data.columns and 'ACTION' in annotation_data.columns:
                problems = annotation_data['PROBLEM'].dropna()
                actions = annotation_data['ACTION'].dropna()
                
                patterns['problem_types'] = self._categorize_problems(problems)
                patterns['action_types'] = self._categorize_actions(actions)
                patterns['problem_action_pairs'] = self._analyze_problem_action_relationships(
                    annotation_data
                )
        
        self.analysis_results['failure_patterns'] = patterns
        return patterns
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in maintenance and incidents"""
        temporal_patterns = {
            'seasonal_trends': {},
            'time_of_day_patterns': {},
            'maintenance_intervals': {}
        }
        
        # Analyze temporal patterns from remaining maintenance data
        if 'maintenance_remaining' in self.datasets:
            maint_data = self.datasets['maintenance_remaining']
            
            # Extract date information if available
            date_columns = [col for col in maint_data.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                temporal_patterns['maintenance_timing'] = self._analyze_maintenance_timing(
                    maint_data, date_columns
                )
        
        self.analysis_results['temporal_patterns'] = temporal_patterns
        return temporal_patterns
    
    def analyze_aircraft_types(self) -> Dict:
        """Analyze patterns by aircraft type and configuration"""
        aircraft_patterns = {
            'aircraft_models': {},
            'engine_types': {},
            'weight_categories': {},
            'operation_types': {}
        }
        
        # Analyze aircraft type patterns from remaining maintenance data
        if 'maintenance_remaining' in self.datasets:
            maint_data = self.datasets['maintenance_remaining']
            
            # Look for aircraft type indicators in column names
            aircraft_cols = [col for col in maint_data.columns if any(
                keyword in col.lower() for keyword in ['aircraft', 'engine', 'weight', 'wing']
            )]
            
            for col in aircraft_cols:
                if col in maint_data.columns:
                    aircraft_patterns[f'{col}_distribution'] = maint_data[col].value_counts().to_dict()
        
        self.analysis_results['aircraft_patterns'] = aircraft_patterns
        return aircraft_patterns
    
    def identify_sensemaking_themes(self) -> List[Dict]:
        """Identify key themes for sensemaking questions"""
        themes = []
        
        # Theme 1: Root Cause Analysis
        themes.append({
            'category': 'root_cause_analysis',
            'description': 'Understanding underlying causes of failures',
            'focus_areas': [
                'maintenance oversight patterns',
                'component failure cascades',
                'human factors in maintenance',
                'environmental contributors'
            ],
            'data_sources': ['maintenance_remaining', 'aircraft_annotation']
        })
        
        # Theme 2: Predictive Maintenance
        themes.append({
            'category': 'predictive_maintenance',
            'description': 'Identifying early warning signs and prevention strategies',
            'focus_areas': [
                'component degradation patterns',
                'maintenance interval optimization',
                'failure precursor identification',
                'risk assessment models'
            ],
            'data_sources': ['maintenance_remaining', 'aircraft_annotation']
        })
        
        # Theme 3: Safety Recommendations
        themes.append({
            'category': 'safety_recommendations',
            'description': 'Developing actionable safety improvements',
            'focus_areas': [
                'procedural improvements',
                'training gap identification',
                'inspection protocol optimization',
                'communication enhancement'
            ],
            'data_sources': ['maintenance_remaining', 'aircraft_annotation']
        })
        
        # Theme 4: System-Level Understanding
        themes.append({
            'category': 'system_level_understanding',
            'description': 'Holistic view of aviation maintenance ecosystem',
            'focus_areas': [
                'multi-factor incident scenarios',
                'operational context effects',
                'fleet-wide pattern analysis',
                'regulatory compliance trends'
            ],
            'data_sources': ['maintenance_remaining', 'aircraft_annotation']
        })
        
        self.analysis_results['sensemaking_themes'] = themes
        return themes
    
    def _extract_keywords_from_narratives(self, narratives: pd.Series) -> Dict:
        """Extract key technical terms and failure indicators from narratives"""
        keywords = defaultdict(int)
        
        # Common aviation maintenance keywords
        aviation_terms = [
            'engine', 'fuel', 'brake', 'gear', 'hydraulic', 'electrical', 'control',
            'leak', 'crack', 'fail', 'malfunction', 'overheat', 'vibration', 'noise',
            'pressure', 'temperature', 'wear', 'corrosion', 'fatigue', 'fracture'
        ]
        
        for narrative in narratives:
            if pd.isna(narrative):
                continue
            
            narrative_lower = str(narrative).lower()
            for term in aviation_terms:
                if term in narrative_lower:
                    keywords[term] += 1
        
        return dict(keywords)
    
    def _categorize_maintenance_issues(self, narratives: pd.Series) -> Dict:
        """Categorize maintenance issues"""
        categories = {
            'preventive_maintenance': 0,
            'corrective_maintenance': 0,
            'inspection_issues': 0,
            'component_replacement': 0,
            'system_checks': 0
        }
        
        # Simple categorization based on keywords
        for narrative in narratives:
            if pd.isna(narrative):
                continue
            
            narrative_lower = str(narrative).lower()
            
            if any(word in narrative_lower for word in ['inspect', 'check', 'exam']):
                categories['inspection_issues'] += 1
            elif any(word in narrative_lower for word in ['replace', 'install']):
                categories['component_replacement'] += 1
            elif any(word in narrative_lower for word in ['repair', 'fix']):
                categories['corrective_maintenance'] += 1
            elif any(word in narrative_lower for word in ['prevent', 'schedule']):
                categories['preventive_maintenance'] += 1
            else:
                categories['system_checks'] += 1
        
        return categories
    
    def _categorize_problems(self, problems: pd.Series) -> Dict:
        """Categorize problems from aircraft annotation data"""
        problem_categories = defaultdict(int)
        
        for problem in problems:
            if pd.isna(problem):
                continue
            
            problem_lower = str(problem).lower()
            
            # Categorize based on keywords
            if any(word in problem_lower for word in ['leak', 'leaking']):
                problem_categories['leakage_issues'] += 1
            elif any(word in problem_lower for word in ['crack', 'cracked']):
                problem_categories['structural_damage'] += 1
            elif any(word in problem_lower for word in ['loose', 'tight']):
                problem_categories['fastener_issues'] += 1
            elif any(word in problem_lower for word in ['engine', 'power']):
                problem_categories['engine_problems'] += 1
            else:
                problem_categories['other_problems'] += 1
        
        return dict(problem_categories)
    
    def _categorize_actions(self, actions: pd.Series) -> Dict:
        """Categorize maintenance actions"""
        action_categories = defaultdict(int)
        
        for action in actions:
            if pd.isna(action):
                continue
            
            action_lower = str(action).lower()
            
            if any(word in action_lower for word in ['replace', 'replaced']):
                action_categories['replacement'] += 1
            elif any(word in action_lower for word in ['tighten', 'tightened']):
                action_categories['tightening'] += 1
            elif any(word in action_lower for word in ['inspect', 'inspected']):
                action_categories['inspection'] += 1
            elif any(word in action_lower for word in ['repair', 'repaired']):
                action_categories['repair'] += 1
            else:
                action_categories['other_actions'] += 1
        
        return dict(action_categories)
    
    def _analyze_problem_action_relationships(self, data: pd.DataFrame) -> Dict:
        """Analyze relationships between problems and actions"""
        relationships = defaultdict(list)
        
        if 'PROBLEM' in data.columns and 'ACTION' in data.columns:
            for _, row in data.iterrows():
                problem = str(row['PROBLEM']).lower() if pd.notna(row['PROBLEM']) else ''
                action = str(row['ACTION']).lower() if pd.notna(row['ACTION']) else ''
                
                if problem and action:
                    # Simplified relationship mapping
                    if 'leak' in problem and 'replace' in action:
                        relationships['leak_replacement'].append((problem, action))
                    elif 'crack' in problem and ('repair' in action or 'replace' in action):
                        relationships['crack_repair'].append((problem, action))
        
        return dict(relationships)
    
    def _analyze_maintenance_timing(self, data: pd.DataFrame, date_columns: List[str]) -> Dict:
        """Analyze timing patterns in maintenance data"""
        timing_patterns = {}
        
        # This would analyze temporal patterns if date columns are properly formatted
        # For now, return placeholder structure
        timing_patterns['hourly_distribution'] = {}
        timing_patterns['daily_distribution'] = {}
        timing_patterns['monthly_distribution'] = {}
        
        return timing_patterns
    
    def save_analysis_results(self, output_path: str) -> None:
        """Save analysis results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            print(f"Analysis results saved to {output_path}")
        except Exception as e:
            print(f"Error saving analysis results: {e}")
    
    def get_analysis_summary(self) -> Dict:
        """Get a summary of all analysis results"""
        summary = {
            'datasets_loaded': len(self.datasets),
            'total_records': sum(len(df) for df in self.datasets.values()),
            'analysis_categories': list(self.analysis_results.keys()),
            'themes_identified': len(self.analysis_results.get('sensemaking_themes', []))
        }
        
        return summary

    def analyze_components(self) -> Dict:
        """Analyze aircraft components and their failure patterns"""
        component_patterns = {
            'top_components': {},
            'failure_modes': {},
            'maintenance_frequency': {},
            'criticality_levels': {}
        }
        
        # Analyze component data from remaining maintenance data
        if 'maintenance_remaining' in self.datasets:
            maint_data = self.datasets['maintenance_remaining']
            
            # Extract component information from narrative fields
            if 'c119' in maint_data.columns:
                narratives = maint_data['c119'].dropna()
                component_patterns['component_mentions'] = self._extract_component_mentions(narratives)
                component_patterns['component_failure_modes'] = self._analyze_component_failures(narratives)
        
        # Analyze component data from aircraft annotation
        if 'aircraft_annotation' in self.datasets:
            annotation_data = self.datasets['aircraft_annotation']
            
            if 'PROBLEM' in annotation_data.columns:
                problems = annotation_data['PROBLEM'].dropna()
                component_patterns['annotated_components'] = self._extract_components_from_problems(problems)
        
        self.analysis_results['component_patterns'] = component_patterns
        return component_patterns
    
    def analyze_text_patterns(self) -> Dict:
        """Analyze text patterns and common terminology"""
        text_patterns = {
            'common_keywords': {},
            'technical_terms': {},
            'maintenance_actions': {},
            'severity_indicators': {}
        }
        
        # Analyze text patterns from remaining maintenance data
        if 'maintenance_remaining' in self.datasets:
            maint_data = self.datasets['maintenance_remaining']
            
            # Analyze narrative text columns
            text_columns = [col for col in maint_data.columns if maint_data[col].dtype == 'object']
            for col in text_columns:
                if col in maint_data.columns and not maint_data[col].dropna().empty:
                    text_data = maint_data[col].dropna()
                    text_patterns[f'{col}_keywords'] = self._extract_keywords_from_text(text_data)
        
        # Analyze text patterns from aircraft annotation
        if 'aircraft_annotation' in self.datasets:
            annotation_data = self.datasets['aircraft_annotation']
            
            if 'PROBLEM' in annotation_data.columns:
                problems = annotation_data['PROBLEM'].dropna()
                text_patterns['problem_keywords'] = self._extract_keywords_from_text(problems)
            
            if 'ACTION' in annotation_data.columns:
                actions = annotation_data['ACTION'].dropna()
                text_patterns['action_keywords'] = self._extract_keywords_from_text(actions)
        
        self.analysis_results['text_patterns'] = text_patterns
        return text_patterns
    
    def _extract_component_mentions(self, narratives: pd.Series) -> Dict:
        """Extract component mentions from maintenance narratives"""
        # Common aircraft components
        components = ['engine', 'wing', 'fuselage', 'landing gear', 'propeller', 'flap', 'aileron', 
                     'rudder', 'elevator', 'brake', 'tire', 'battery', 'alternator', 'fuel pump',
                     'oil', 'hydraulic', 'avionics', 'radio', 'transponder', 'compass']
        
        component_counts = {}
        for narrative in narratives:
            if pd.notna(narrative):
                text = str(narrative).lower()
                for component in components:
                    if component in text:
                        component_counts[component] = component_counts.get(component, 0) + 1
        
        return dict(sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_component_failures(self, narratives: pd.Series) -> Dict:
        """Analyze component failure modes from narratives"""
        failure_modes = {}
        
        # Common failure indicators
        failure_terms = ['failed', 'broken', 'cracked', 'leak', 'worn', 'damaged', 'malfunction', 'inoperative']
        
        for narrative in narratives:
            if pd.notna(narrative):
                text = str(narrative).lower()
                for term in failure_terms:
                    if term in text:
                        failure_modes[term] = failure_modes.get(term, 0) + 1
        
        return dict(sorted(failure_modes.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_components_from_problems(self, problems: pd.Series) -> Dict:
        """Extract components mentioned in problem descriptions"""
        component_mentions = {}
        
        for problem in problems:
            if pd.notna(problem):
                text = str(problem).lower()
                # Simple keyword extraction - could be enhanced with NLP
                words = text.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        component_mentions[word] = component_mentions.get(word, 0) + 1
        
        # Return top 10 most mentioned terms
        return dict(sorted(component_mentions.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _extract_keywords_from_text(self, text_data: pd.Series) -> Dict:
        """Extract common keywords from text data"""
        keyword_counts = {}
        
        # Common stop words to filter out
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        for text in text_data:
            if pd.notna(text):
                # Simple tokenization and counting
                words = str(text).lower().split()
                for word in words:
                    # Clean word and filter
                    clean_word = ''.join(char for char in word if char.isalnum())
                    if len(clean_word) > 3 and clean_word not in stop_words:
                        keyword_counts[clean_word] = keyword_counts.get(clean_word, 0) + 1
        
        # Return top 15 keywords
        return dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:15])

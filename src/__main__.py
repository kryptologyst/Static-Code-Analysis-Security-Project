"""
Main entry point for the static code analysis application.

This module provides command-line interface and main application logic.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.analysis.scanner import CodeScanner
from src.models.vulnerability_detector import VulnerabilityDetector
from src.data.features import CodeFeatureExtractor
from src.utils.helpers import setup_logging, set_random_seed, load_yaml_config
from src.utils.explainability import ExplainabilityManager
from src.evaluation.metrics import VulnerabilityEvaluator

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


def analyze_file(file_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze a single file for vulnerabilities."""
    logger.info(f"Analyzing file: {file_path}")
    
    # Initialize scanner
    scanner = CodeScanner(config)
    
    # Scan file
    vulnerabilities = scanner.scan_file(file_path)
    
    # Convert to dictionaries
    results = [v.to_dict() for v in vulnerabilities]
    
    logger.info(f"Found {len(results)} vulnerabilities in {file_path}")
    return results


def analyze_directory(directory_path: str, config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze all Python files in a directory."""
    logger.info(f"Analyzing directory: {directory_path}")
    
    # Initialize scanner
    scanner = CodeScanner(config)
    
    # Scan directory
    results = scanner.scan_directory(directory_path)
    
    # Convert to dictionaries
    converted_results = {}
    for file_path, vulnerabilities in results.items():
        converted_results[file_path] = [v.to_dict() for v in vulnerabilities]
    
    total_vulnerabilities = sum(len(vulns) for vulns in converted_results.values())
    logger.info(f"Found {total_vulnerabilities} vulnerabilities across {len(converted_results)} files")
    
    return converted_results


def analyze_code_string(code: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyze code from string input."""
    logger.info("Analyzing code string")
    
    # Initialize scanner
    scanner = CodeScanner(config)
    
    # Scan code
    vulnerabilities = scanner.scan_code(code)
    
    # Convert to dictionaries
    results = [v.to_dict() for v in vulnerabilities]
    
    logger.info(f"Found {len(results)} vulnerabilities in code string")
    return results


def print_results(results: List[Dict[str, Any]], show_details: bool = True) -> None:
    """Print analysis results to console."""
    if not results:
        print("✅ No vulnerabilities detected!")
        return
    
    print(f"\n🚨 Found {len(results)} vulnerabilities:\n")
    
    for i, vuln in enumerate(results, 1):
        severity_colors = {
            'critical': '\033[91m',  # Red
            'high': '\033[93m',       # Yellow
            'medium': '\033[94m',     # Blue
            'low': '\033[92m',        # Green
            'info': '\033[96m'        # Cyan
        }
        
        color = severity_colors.get(vuln['severity'], '\033[0m')
        reset_color = '\033[0m'
        
        print(f"{i}. {color}{vuln['severity'].upper()}{reset_color}: {vuln['vulnerability_type']}")
        print(f"   Line {vuln['line_number']}: {vuln['description']}")
        
        if show_details:
            print(f"   Code: {vuln['code_snippet']}")
            print(f"   Confidence: {vuln['confidence']:.2f}")
            print(f"   Rule ID: {vuln['rule_id']}")
        
        print()


def save_results_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSON file."""
    import json
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def save_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to CSV file."""
    import pandas as pd
    
    if not results:
        logger.warning("No results to save")
        return
    
    df = pd.DataFrame(results)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Static Code Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --file example.py
  python -m src.main --directory src/
  python -m src.main --code "def test(): eval('1+1')"
  python -m src.main --file example.py --output results.json
  python -m src.main --file example.py --config custom_config.yaml
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', '-f', type=str, help='Python file to analyze')
    input_group.add_argument('--directory', '-d', type=str, help='Directory to analyze')
    input_group.add_argument('--code', '-c', type=str, help='Code string to analyze')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress console output')
    
    # Configuration options
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--severity', nargs='+', choices=['critical', 'high', 'medium', 'low', 'info'],
                       help='Filter by severity levels')
    parser.add_argument('--show-details', action='store_true', help='Show detailed vulnerability information')
    
    # Analysis options
    parser.add_argument('--ml-analysis', action='store_true', help='Enable ML-based analysis')
    parser.add_argument('--explain', action='store_true', help='Generate explanations for findings')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_yaml_config(args.config)
        
        # Set random seed
        set_random_seed(config.get('random_seed', 42))
        
        # Apply severity filter if specified
        if args.severity:
            config['severity_filter'] = args.severity
        
        # Analyze based on input type
        if args.file:
            results = analyze_file(args.file, config)
        elif args.directory:
            directory_results = analyze_directory(args.directory, config)
            # Flatten results
            results = []
            for file_path, vulns in directory_results.items():
                for vuln in vulns:
                    vuln['file_path'] = file_path
                    results.append(vuln)
        elif args.code:
            results = analyze_code_string(args.code, config)
        else:
            parser.error("Must specify --file, --directory, or --code")
        
        # Apply ML analysis if requested
        if args.ml_analysis and results:
            logger.info("Performing ML-based analysis")
            
            # Initialize ML components
            detector = VulnerabilityDetector(config.get('models', {}).get('vulnerability_detector', {}))
            extractor = CodeFeatureExtractor(config.get('features', {}))
            
            # Extract features from code snippets
            code_samples = [vuln.get('code_snippet', '') for vuln in results]
            if code_samples:
                try:
                    features = extractor.fit_transform(code_samples)
                    
                    # Make predictions (mock for demo)
                    predictions = [1] * len(features)  # Placeholder
                    
                    # Add ML predictions to results
                    for i, vuln in enumerate(results):
                        vuln['ml_prediction'] = predictions[i] if i < len(predictions) else 0
                
                except Exception as e:
                    logger.warning(f"ML analysis failed: {e}")
        
        # Generate explanations if requested
        if args.explain and results:
            logger.info("Generating explanations")
            
            explainer = ExplainabilityManager(config)
            
            for vuln in results:
                try:
                    explanation = explainer.explain_vulnerability(
                        vuln, 
                        vuln.get('code_snippet', '')
                    )
                    vuln['explanation'] = explanation.explanation_text
                    vuln['rule_evidence'] = explanation.rule_evidence
                except Exception as e:
                    logger.warning(f"Explanation generation failed: {e}")
        
        # Output results
        if not args.quiet:
            print_results(results, args.show_details)
        
        # Save results if output specified
        if args.output:
            if args.format == 'json':
                save_results_json(results, args.output)
            elif args.format == 'csv':
                save_results_csv(results, args.output)
        
        # Print summary
        if not args.quiet:
            severity_counts = {}
            for vuln in results:
                severity = vuln['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            print(f"\n📊 Summary:")
            for severity, count in sorted(severity_counts.items()):
                print(f"  {severity}: {count}")
        
        logger.info("Analysis completed successfully")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

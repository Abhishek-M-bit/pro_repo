from pipeline import FleetInvestigationPipeline
import json

# Global singleton to hold the loaded pipeline models in memory
_pipeline_instance = None

def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FleetInvestigationPipeline()
    return _pipeline_instance

def analyze_incident(incident: dict) -> dict:
    """
    Main entry point for backend integration.
    Analyzes a current vehicle incident and returns a structured Evidence Package.
    
    Args:
        incident (dict): Dictionary containing the 11 sensor readings, Mode, and active_dtcs.
        
    Returns:
        dict: The final evidence package containing similarity, anomaly, and RAG results.
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.analyze(incident)
        return result
    except ValueError as ve:
        # Input validation errors
        return {
            "error": "Input Validation Error",
            "message": str(ve)
        }
    except Exception as e:
        # Unexpected errors
        return {
            "error": "Internal Processing Error",
            "message": str(e)
        }

if __name__ == "__main__":
    # Demo execution
    demo_incident = {
        "LOAD_PCT": 26.3,
        "ECT": 169.0,
        "MAP": 14.4,
        "RPM": 790.0,
        "VSS": 0.0,
        "IAT": 106.0,
        "MAF": 0.02,
        "FRP": 4507.5,
        "BARO": 14.2,
        "VPWR": 13.64,
        "AAT": 126.0,
        "Mode": 0,
        "active_dtcs": [
            "P0403",
            "P0404",
            "P2009",
            "P2015"
        ]
    }
    
    print("Running inference demo...\n")
    evidence_package = analyze_incident(demo_incident)
    print(json.dumps(evidence_package, indent=2))

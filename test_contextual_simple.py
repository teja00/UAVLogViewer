#!/usr/bin/env python3
"""
Simple test for the contextual improvements made to the MultiRoleAgent.
This tests specifically the issue described where "consistently high" was misinterpreted.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_code_generation_improvements():
    """Test the improved code generation methods."""
    print("🧪 Testing Code Generation Improvements")
    print("="*60)
    
    # Test the specific problematic code patterns
    test_patterns = [
        "average_GPS_quality = 98.7",
        "consistently high",
        "altitude_consistency", 
        "flight_performance = 'high'"
    ]
    
    # Import the tool directly to test
    try:
        from backend.agent.multi_role_agent import PythonCodeExecutorTool
        from models import V2ConversationSession
        import pandas as pd
        
        # Create a mock session with sample data
        session = V2ConversationSession(session_id="test")
        session.dataframes = {
            'GPS': pd.DataFrame({
                'Alt': [1420, 1425, 1430, 1425, 1420],
                'Status': [3, 3, 3, 3, 3]
            }),
            'CTUN': pd.DataFrame({
                'Alt': [1422, 1425, 1428, 1425, 1422],
                'DAlt': [1425, 1425, 1425, 1425, 1425]
            })
        }
        
        # Create the tool
        from tools.analysis_tools import AnalysisTools
        tool = PythonCodeExecutorTool(AnalysisTools(), session)
        
        print("🔍 Testing improved interpretation of ambiguous code...")
        
        for i, pattern in enumerate(test_patterns, 1):
            print(f"\nTest {i}: '{pattern}'")
            try:
                result = tool._run(pattern)
                print(f"✅ Result: {result[:200]}...")
                
                # Check if result contains meaningful analysis
                result_lower = result.lower()
                meaningful_indicators = ['altitude', 'analysis', 'meters', 'consistency', 'stable']
                meaningful_count = sum(1 for indicator in meaningful_indicators if indicator in result_lower)
                
                if "Cannot execute incomplete code" in result:
                    print("  ❌ FAIL: Still returning generic error")
                elif meaningful_count >= 2:
                    print("  ✅ PASS: Generated meaningful analysis")
                else:
                    print("  ⚠️  PARTIAL: Some analysis but could be better")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
        # Test new methods directly
        print(f"\n🔬 Testing new altitude consistency code generation...")
        try:
            consistency_code = tool._generate_altitude_consistency_code()
            print(f"✅ Generated altitude consistency code ({len(consistency_code)} chars)")
            
            # Check if it contains key analysis elements
            if all(keyword in consistency_code for keyword in ['altitude', 'consistency', 'std_dev', 'analysis']):
                print("  ✅ PASS: Contains comprehensive altitude analysis")
            else:
                print("  ⚠️  PARTIAL: Missing some analysis elements")
                
        except Exception as e:
            print(f"  ❌ Error generating altitude consistency code: {str(e)}")
        
        print(f"\n🔬 Testing new flight stability code generation...")
        try:
            stability_code = tool._generate_flight_stability_code()
            print(f"✅ Generated flight stability code ({len(stability_code)} chars)")
            
            # Check if it contains key analysis elements
            if all(keyword in stability_code for keyword in ['stability', 'attitude', 'vibration']):
                print("  ✅ PASS: Contains comprehensive stability analysis")
            else:
                print("  ⚠️  PARTIAL: Missing some stability elements")
                
        except Exception as e:
            print(f"  ❌ Error generating flight stability code: {str(e)}")
            
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   This may be due to missing dependencies or configuration issues.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_query_interpretation():
    """Test the improved query interpretation patterns."""
    print("\n🔍 Testing Query Interpretation Patterns")
    print("="*50)
    
    try:
        from backend.agent.multi_role_agent import PythonCodeExecutorTool
        from models import V2ConversationSession
        
        session = V2ConversationSession(session_id="test_interpretation")
        
        # Create a mock tool to test interpretation methods
        from tools.analysis_tools import AnalysisTools
        tool = PythonCodeExecutorTool(AnalysisTools(), session)
        
        # Test the enhanced query generation
        test_queries = [
            "consistently high altitude",
            "stable flight performance", 
            "flight consistency analysis",
            "altitude stability metrics"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: '{query}'")
            try:
                # Test the query code generation
                generated_code = tool._generate_code_for_query(query)
                
                if generated_code:
                    print(f"  ✅ Generated code for query ({len(generated_code)} chars)")
                    
                    # Check if it's the right type of analysis
                    if 'consistency' in query and 'altitude_consistency_code' in str(generated_code):
                        print("  ✅ PASS: Correctly identified as altitude consistency query")
                    elif 'stability' in query and ('stability_code' in str(generated_code) or 'attitude' in generated_code):
                        print("  ✅ PASS: Correctly identified as stability query")
                    else:
                        print("  ⚠️  PARTIAL: Generated code but may not be optimal match")
                else:
                    print("  ❌ FAIL: No code generated for query")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error in query interpretation test: {str(e)}")

def main():
    """Run the simplified contextual improvement tests."""
    print("🧪 Simplified MultiRoleAgent Contextual Improvements Test")
    print("="*70)
    
    try:
        # Test 1: Code generation improvements
        test_code_generation_improvements()
        
        # Test 2: Query interpretation
        test_query_interpretation()
        
        print("\n" + "="*70)
        print("🎉 Test completed!")
        print("\n💡 Key Improvements Made:")
        print("   1. Enhanced interpretation of incomplete code assignments")
        print("   2. Better contextual understanding of 'consistently high' queries")
        print("   3. New altitude consistency and flight stability analysis methods")
        print("   4. Improved fallback strategies for ambiguous queries")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
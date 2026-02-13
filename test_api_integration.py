"""
Integration tests for the Flask API
"""
import unittest
import tempfile
import os
import sys
from unittest.mock import patch
import json

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
import pandas as pd


class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a temporary CSV file with sample data for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        
        sample_data = """Support,Amine_1_or_Additive_1,Amine_2_or_Additive_2,Amine_3_or_Additive_3,MW_Mn_g_mol,Organic_Content_pct,Relative_Humidity_pct,CO2_Concentration_vol_pct,Flow_Rate_mL_min,Adsorption_Temperature_C,CO2_Test_Method,CO2_Capacity_mmol_g
SBA-15,BPEI,No,No,1000,10,0,0.04,100,25,TGA,2.5
NS,TEPA,No,No,2000,15,0,0.04,100,25,TGA,2.8
MCM-41,DEA,No,No,1500,12,0,0.04,100,25,TGA,2.6
SBA-15,TEPA,No,No,1200,11,0,0.04,100,25,TGA,2.7
"""
        self.temp_file.write(sample_data)
        self.temp_file.close()
        
        # Create data directory and copy the sample data
        os.makedirs('data', exist_ok=True)
        with open(self.temp_file.name, 'r') as src, open('data/historical_experiments.csv', 'w') as dst:
            dst.write(src.read())

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        if os.path.exists('data/historical_experiments.csv'):
            os.unlink('data/historical_experiments.csv')
        # Remove the data directory if it exists and is empty
        if os.path.exists('data'):
            try:
                os.rmdir('data')
            except OSError:
                # Directory not empty, list contents for debugging
                import shutil
                shutil.rmtree('data', ignore_errors=True)

    def test_get_root(self):
        """Test the root endpoint"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_get_test_endpoint(self):
        """Test the test endpoint"""
        response = self.app.get('/api/test')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)

    def test_initialize_system(self):
        """Test system initialization"""
        init_data = {
            "searchBounds": {
                "supports": ["SBA-15", "NS", "MCM-41"],
                "amine1": ["BPEI", "TEPA", "DEA"],
                "amine2": ["No", "DEA"],
                "additive3": ["No"],
                "mwRange": [0, 10000],
                "ocRange": [0, 100]
            },
            "conditions": {
                "co2Concentration": 0.04,
                "temperature": 25.0,
                "humidity": 0,
                "flowRate": 100.0,
                "testMethod": "TGA"
            }
        }
        
        response = self.app.post('/api/init', json=init_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('data_points', data)
        self.assertIn('best_capacity', data)

    def test_get_status(self):
        """Test getting system status"""
        response = self.app.get('/api/get-status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        # Status should be returned even if system is not initialized

    def test_generate_candidates_uninitialized(self):
        """Test generating candidates when system is not initialized"""
        response = self.app.post('/api/generate-candidates', json={"n_candidates": 1})
        # This should return an error since the system is not initialized
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_initialize_and_generate_candidates(self):
        """Test initializing and then generating candidates"""
        # First, initialize the system
        init_data = {
            "searchBounds": {
                "supports": ["SBA-15", "NS", "MCM-41"],
                "amine1": ["BPEI", "TEPA", "DEA"],
                "amine2": ["No", "DEA"],
                "additive3": ["No"],
                "mwRange": [0, 10000],
                "ocRange": [0, 100]
            },
            "conditions": {
                "co2Concentration": 0.04,
                "temperature": 25.0,
                "humidity": 0,
                "flowRate": 100.0,
                "testMethod": "TGA"
            }
        }
        
        init_response = self.app.post('/api/init', json=init_data)
        self.assertEqual(init_response.status_code, 200)
        
        # Now try to generate candidates
        response = self.app.post('/api/generate-candidates', json={"n_candidates": 1})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        # Even with limited data, it should return successful response

    def test_record_experiment_without_candidates(self):
        """Test recording an experiment without generating candidates first"""
        experiment_data = {
            "candidate_id": 0,
            "actual_capacity": 2.8,
            "notes": "Test experiment"
        }
        response = self.app.post('/api/record-experiment', json=experiment_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_export_data(self):
        """Test data export functionality"""
        response = self.app.get('/api/export-data')
        # Export might fail if system is not initialized, but should return appropriate status
        self.assertIn(response.status_code, [200, 400])

    def test_toggle_data_mode_uninitialized(self):
        """Test toggling data mode when system is not initialized"""
        response = self.app.post('/api/toggle-data-mode', json={"mode": "real-only"})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_update_config_uninitialized(self):
        """Test updating config when system is not initialized"""
        config_data = {
            "searchBounds": {
                "supports": ["SBA-15"],
                "amine1": ["BPEI"],
                "amine2": ["No"],
                "additive3": ["No"],
                "mwRange": [0, 1000],
                "ocRange": [0, 50]
            },
            "conditions": {
                "co2Concentration": 0.05,
                "temperature": 30.0,
                "humidity": 5,
                "flowRate": 120.0,
                "testMethod": "BET"
            }
        }
        response = self.app.post('/api/update-config', json=config_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_archive_experiments_uninitialized(self):
        """Test archiving experiments when system is not initialized"""
        response = self.app.post('/api/archive-experiments')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

    def test_get_optimization_history_uninitialized(self):
        """Test getting optimization history when system is not initialized"""
        response = self.app.get('/api/get-optimization-history')
        self.assertIn(response.status_code, [200, 400])  # Could be either depending on implementation
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('success', data)

    def test_reset_system(self):
        """Test resetting the system"""
        response = self.app.post('/api/reset-system', json={})
        # Should return success or error appropriately
        self.assertIn(response.status_code, [200, 400])
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertTrue(data['success'])


class TestFlaskAPIInitialized(unittest.TestCase):
    """Test API endpoints that require initialization first"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create a temporary CSV file with sample data for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        
        sample_data = """Support,Amine_1_or_Additive_1,Amine_2_or_Additive_2,Amine_3_or_Additive_3,MW_Mn_g_mol,Organic_Content_pct,Relative_Humidity_pct,CO2_Concentration_vol_pct,Flow_Rate_mL_min,Adsorption_Temperature_C,CO2_Test_Method,CO2_Capacity_mmol_g
SBA-15,BPEI,No,No,1000,10,0,0.04,100,25,TGA,2.5
NS,TEPA,No,No,2000,15,0,0.04,100,25,TGA,2.8
MCM-41,DEA,No,No,1500,12,0,0.04,100,25,TGA,2.6
SBA-15,TEPA,No,No,1200,11,0,0.04,100,25,TGA,2.7
"""
        self.temp_file.write(sample_data)
        self.temp_file.close()
        
        # Create data directory and copy the sample data
        os.makedirs('data', exist_ok=True)
        with open(self.temp_file.name, 'r') as src, open('data/historical_experiments.csv', 'w') as dst:
            dst.write(src.read())
        
        # Initialize the system for these tests
        init_data = {
            "searchBounds": {
                "supports": ["SBA-15", "NS", "MCM-41"],
                "amine1": ["BPEI", "TEPA", "DEA"],
                "amine2": ["No", "DEA"],
                "additive3": ["No"],
                "mwRange": [0, 10000],
                "ocRange": [0, 100]
            },
            "conditions": {
                "co2Concentration": 0.04,
                "temperature": 25.0,
                "humidity": 0,
                "flowRate": 100.0,
                "testMethod": "TGA"
            }
        }
        
        response = self.app.post('/api/init', json=init_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        if os.path.exists('data/historical_experiments.csv'):
            os.unlink('data/historical_experiments.csv')
        # Remove the data directory if it exists and is empty
        if os.path.exists('data'):
            try:
                os.rmdir('data')
            except OSError:
                # Directory not empty, list contents for debugging
                import shutil
                shutil.rmtree('data', ignore_errors=True)

    def test_get_status_after_initialization(self):
        """Test getting status after initialization"""
        response = self.app.get('/api/get-status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertTrue(data['initialized'])

    def test_generate_candidates_after_initialization(self):
        """Test generating candidates after initialization"""
        response = self.app.post('/api/generate-candidates', json={"n_candidates": 1})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        # May return empty candidates if optimization hasn't started properly

    def test_get_optimization_history_after_initialization(self):
        """Test getting optimization history after initialization"""
        response = self.app.get('/api/get-optimization-history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('history', data)

    def test_update_config_after_initialization(self):
        """Test updating config after initialization"""
        config_data = {
            "searchBounds": {
                "supports": ["SBA-15"],
                "amine1": ["BPEI"],
                "amine2": ["No"],
                "additive3": ["No"],
                "mwRange": [0, 1000],
                "ocRange": [0, 50]
            },
            "conditions": {
                "co2Concentration": 0.05,
                "temperature": 30.0,
                "humidity": 5,
                "flowRate": 120.0,
                "testMethod": "BET"
            }
        }
        response = self.app.post('/api/update-config', json=config_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])


if __name__ == '__main__':
    unittest.main()
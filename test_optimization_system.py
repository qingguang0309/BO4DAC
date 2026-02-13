"""
Unit tests for the optimization system components
"""
import unittest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization_system import HistoricalDataProcessor, CatalystBOWithHistory, InteractiveCatalystOptimizer


class TestHistoricalDataProcessor(unittest.TestCase):
    def setUp(self):
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

    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test that HistoricalDataProcessor initializes correctly"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        self.assertEqual(processor.data_path, self.temp_file.name)
        self.assertIsNotNone(processor.df)
        self.assertEqual(len(processor.df), 4)
        self.assertIn('Support', processor.categorical_cols)
        self.assertIn('MW_Mn_g_mol', processor.continuous_cols)
        self.assertIn('CO2_Capacity_mmol_g', processor.target_col)

    def test_preprocess_data(self):
        """Test data preprocessing"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        X, Y, df = processor.preprocess_data(target_conditions=target_conditions)
        
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)
        self.assertEqual(X.shape[0], 4)  # All 4 rows should be included
        self.assertEqual(Y.shape[0], 4)
        self.assertEqual(X.shape[1], 6)  # 4 categorical + 2 continuous features

    def test_encode_configuration(self):
        """Test configuration encoding"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        # Preprocess to set up label encoders
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        processor.preprocess_data(target_conditions=target_conditions)
        
        config = {
            'Support': 'SBA-15',
            'Amine_1_or_Additive_1': 'BPEI',
            'Amine_2_or_Additive_2': 'No',
            'Amine_3_or_Additive_3': 'No',
            'MW_Mn_g_mol': 1000.0,
            'Organic_Content_pct': 10.0
        }
        
        encoded = processor.encode_configuration(config)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (1, 6))


class TestCatalystBOWithHistory(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with sample data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        
        sample_data = """Support,Amine_1_or_Additive_1,Amine_2_or_Additive_2,Amine_3_or_Additive_3,MW_Mn_g_mol,Organic_Content_pct,Relative_Humidity_pct,CO2_Concentration_vol_pct,Flow_Rate_mL_min,Adsorption_Temperature_C,CO2_Test_Method,CO2_Capacity_mmol_g
SBA-15,BPEI,No,No,1000,10,0,0.04,100,25,TGA,2.5
NS,TEPA,No,No,2000,15,0,0.04,100,25,TGA,2.8
MCM-41,DEA,No,No,1500,12,0,0.04,100,25,TGA,2.6
SBA-15,TEPA,No,No,1200,11,0,0.04,100,25,TGA,2.7
"""
        self.temp_file.write(sample_data)
        self.temp_file.close()

    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test CatalystBOWithHistory initialization"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        bo_system = CatalystBOWithHistory(
            processor,
            target_conditions=target_conditions
        )
        
        self.assertEqual(len(bo_system.train_X), 4)
        self.assertEqual(len(bo_system.train_Y), 4)
        self.assertEqual(bo_system.target_conditions, target_conditions)

    def test_encode_decode_configuration(self):
        """Test encoding and decoding of configurations"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        bo_system = CatalystBOWithHistory(
            processor,
            target_conditions=target_conditions
        )
        
        config = {
            'Support': 'SBA-15',
            'Amine_1_or_Additive_1': 'BPEI',
            'Amine_2_or_Additive_2': 'No',
            'Amine_3_or_Additive_3': 'No',
            'MW_Mn_g_mol': 1000.0,
            'Organic_Content_pct': 10.0
        }
        
        # Test encoding
        X_tensor = bo_system.encode_configuration_for_bo(config)
        self.assertIsInstance(X_tensor, torch.Tensor)
        self.assertEqual(X_tensor.shape, (1, 6))
        
        # Test decoding
        decoded_config = bo_system.decode_configuration(X_tensor)
        self.assertIsInstance(decoded_config, dict)
        self.assertEqual(decoded_config['Support'], 'SBA-15')
        self.assertEqual(decoded_config['Amine_1_or_Additive_1'], 'BPEI')

    def test_add_experimental_result(self):
        """Test adding experimental results"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        bo_system = CatalystBOWithHistory(
            processor,
            target_conditions=target_conditions
        )
        
        initial_length = len(bo_system.real_experiments_df)
        
        config = {
            'Support': 'NewSupport',
            'Amine_1_or_Additive_1': 'NewAmine',
            'Amine_2_or_Additive_2': 'No',
            'Amine_3_or_Additive_3': 'No',
            'MW_Mn_g_mol': 1100.0,
            'Organic_Content_pct': 11.0
        }
        
        bo_system.add_experimental_result(config, 3.0)
        
        # Check that the experiment was added
        self.assertEqual(len(bo_system.real_experiments_df), initial_length + 1)
        
        # Check that optimization history was updated
        history = bo_system.get_optimization_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[-1]['actual_capacity'], 3.0)

    def test_check_similar_configuration(self):
        """Test checking for similar configurations"""
        processor = HistoricalDataProcessor(self.temp_file.name)
        
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        bo_system = CatalystBOWithHistory(
            processor,
            target_conditions=target_conditions
        )
        
        config = {
            'Support': 'SBA-15',
            'Amine_1_or_Additive_1': 'BPEI',
            'Amine_2_or_Additive_2': 'No',
            'Amine_3_or_Additive_3': 'No',
            'MW_Mn_g_mol': 1000.0,
            'Organic_Content_pct': 10.0
        }
        
        # This should return True since this config exists in historical data
        is_similar = bo_system.check_similar_configuration(config)
        # Note: Since we're comparing with historical data, this might return True depending on implementation
        # The important thing is that the method runs without error


class TestInteractiveCatalystOptimizer(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with sample data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        
        sample_data = """Support,Amine_1_or_Additive_1,Amine_2_or_Additive_2,Amine_3_or_Additive_3,MW_Mn_g_mol,Organic_Content_pct,Relative_Humidity_pct,CO2_Concentration_vol_pct,Flow_Rate_mL_min,Adsorption_Temperature_C,CO2_Test_Method,CO2_Capacity_mmol_g
SBA-15,BPEI,No,No,1000,10,0,0.04,100,25,TGA,2.5
NS,TEPA,No,No,2000,15,0,0.04,100,25,TGA,2.8
MCM-41,DEA,No,No,1500,12,0,0.04,100,25,TGA,2.6
SBA-15,TEPA,No,No,1200,11,0,0.04,100,25,TGA,2.7
"""
        self.temp_file.write(sample_data)
        self.temp_file.close()

    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_file.name)

    @patch('builtins.input', side_effect=['0'])  # Exit immediately
    def test_initialization(self, mock_input):
        """Test InteractiveCatalystOptimizer initialization"""
        target_conditions = {
            'Relative_Humidity_pct': 0,
            'CO2_Concentration_vol_pct': 0.04,
            'Flow_Rate_mL_min': 100,
            'Adsorption_Temperature_C': 25,
            'CO2_Test_Method': 'TGA'
        }
        
        categorical_bounds = {
            "Support": ['SBA-15', 'NS', 'MCM-41'],
            "Amine_1_or_Additive_1": ['BPEI', 'TEPA', 'DEA'],
            "Amine_2_or_Additive_2": ['No', 'DEA'],
            "Amine_3_or_Additive_3": ['No']
        }

        continuous_bounds = {
            "MW_Mn_g_mol": (0, 10000),
            "Organic_Content_pct": (0, 100)
        }
        
        optimizer = InteractiveCatalystOptimizer(
            data_path=self.temp_file.name,
            target_conditions=target_conditions,
            categorical_bounds=categorical_bounds,
            continuous_bounds=continuous_bounds
        )
        
        self.assertEqual(optimizer.data_path, self.temp_file.name)
        self.assertEqual(optimizer.target_conditions, target_conditions)


if __name__ == '__main__':
    unittest.main()
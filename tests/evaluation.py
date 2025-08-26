import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
import pytest
from omegaconf import OmegaConf
from evaluation import visualize_generated_samples, plot_fom_hist, eval_static, evaluation

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_images_chw():
    """Generate sample images in (N, C, H, W) format."""
    return np.random.rand(20, 3, 32, 32)


@pytest.fixture
def sample_images_hw():
    """Generate sample images in (N, H, W) format."""
    return np.random.rand(20, 32, 32)


@pytest.fixture
def sample_images_single_channel():
    """Generate sample images in (N, 1, H, W) format."""
    return np.random.rand(20, 1, 32, 32)


@pytest.fixture
def sample_fom():
    """Generate sample figure of merit values."""
    return np.random.normal(0.5, 0.2, 100)


@pytest.fixture
def mock_cfg():
    """Create a mock configuration object."""
    cfg = OmegaConf.create({
        'savepath': '/tmp/test',
        'model': {'_target_': 'test.model'},
        'train_set_size': 1000,
        'debug': False,
        'data_path': '/path/to/data',
        'evaluation': {
            'functions': [
                {'_target_': 'evaluation.eval_static'},
                {'_target_': 'nanophoto.evaluation.compute_entropy'},
                {'_target_': 'custom.visualization_fn'}
            ]
        }
    })
    return cfg


class TestVisualizeGeneratedSamples:
    """Test the visualize_generated_samples function."""

    def test_visualize_chw_format(self, sample_images_chw, temp_dir):
        """Test visualization with (N, C, H, W) format images."""
        visualize_generated_samples(sample_images_chw, temp_dir, "TestModel", n_samples=16)
        
        expected_file = os.path.join(temp_dir, "testmodel_samples_grid.png")
        assert os.path.exists(expected_file)

    def test_visualize_hw_format(self, sample_images_hw, temp_dir):
        """Test visualization with (N, H, W) format images."""
        visualize_generated_samples(sample_images_hw, temp_dir, "TestModel", n_samples=9)
        
        expected_file = os.path.join(temp_dir, "testmodel_samples_grid.png")
        assert os.path.exists(expected_file)

    def test_visualize_single_channel(self, sample_images_single_channel, temp_dir):
        """Test visualization with single channel images."""
        visualize_generated_samples(sample_images_single_channel, temp_dir, "GrayModel", n_samples=4)
        
        expected_file = os.path.join(temp_dir, "graymodel_samples_grid.png")
        assert os.path.exists(expected_file)

    def test_visualize_fewer_samples_than_requested(self, temp_dir):
        """Test when requesting more samples than available."""
        small_images = np.random.rand(5, 32, 32)
        visualize_generated_samples(small_images, temp_dir, "SmallModel", n_samples=16)
        
        expected_file = os.path.join(temp_dir, "smallmodel_samples_grid.png")
        assert os.path.exists(expected_file)

    def test_visualize_single_sample(self, temp_dir):
        """Test visualization with single sample."""
        single_image = np.random.rand(1, 32, 32)
        visualize_generated_samples(single_image, temp_dir, "SingleModel", n_samples=1)
        
        expected_file = os.path.join(temp_dir, "singlemodel_samples_grid.png")
        assert os.path.exists(expected_file)

    def test_visualize_default_samples(self, sample_images_chw, temp_dir):
        """Test visualization with default number of samples."""
        visualize_generated_samples(sample_images_chw, temp_dir, "DefaultModel")
        
        expected_file = os.path.join(temp_dir, "defaultmodel_samples_grid.png")
        assert os.path.exists(expected_file)


class TestPlotFomHist:
    """Test the plot_fom_hist function."""

    def test_plot_fom_histogram(self, sample_fom, temp_dir):
        """Test FOM histogram generation and saving."""
        plot_fom_hist(sample_fom, "TestModel", temp_dir)
        
        expected_file = os.path.join(temp_dir, "fom_histogram.png")
        assert os.path.exists(expected_file)

    def test_plot_fom_with_extreme_values(self, temp_dir):
        """Test FOM histogram with extreme values."""
        extreme_fom = np.array([0, 0, 0, 1000, 1000, 1000])
        plot_fom_hist(extreme_fom, "ExtremeModel", temp_dir)
        
        expected_file = os.path.join(temp_dir, "fom_histogram.png")
        assert os.path.exists(expected_file)

    def test_plot_fom_empty_array(self, temp_dir):
        """Test FOM histogram with empty array."""
        empty_fom = np.array([])
        plot_fom_hist(empty_fom, "EmptyModel", temp_dir)
        
        expected_file = os.path.join(temp_dir, "fom_histogram.png")
        assert os.path.exists(expected_file)


class TestEvalStatic:
    """Test the eval_static function."""

    @patch('evaluation.eval_metrics')
    @patch.dict(os.environ, {'SLURM_JOB_ID': '12345'})
    def test_eval_static_complete(self, mock_eval_metrics, sample_images_chw, sample_fom, temp_dir, mock_cfg):
        """Test complete eval_static functionality."""
        mock_cfg.data_path = '/path/to/data/file.txt'
        
        eval_static(sample_images_chw, sample_fom, temp_dir, "TestModel", mock_cfg)
        
        # Check that visualization was created
        viz_file = os.path.join(temp_dir, "testmodel_samples_grid.png")
        assert os.path.exists(viz_file)
        
        # Check that histogram was created
        hist_file = os.path.join(temp_dir, "fom_histogram.png")
        assert os.path.exists(hist_file)
        
        # Check that results JSON was created
        results_file = os.path.join(temp_dir, "experiment_results.json")
        assert os.path.exists(results_file)
        
        # Verify JSON content
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        assert results['model_type'] == 'test.model'
        assert results['train_set_size'] == 1000
        assert results['debug'] is False
        assert results['experiment_path'] == temp_dir
        
        # Check that eval_metrics was called
        mock_eval_metrics.assert_called_once()

    @patch('evaluation.eval_metrics')
    @patch.dict(os.environ, {}, clear=True)
    def test_eval_static_no_slurm_job(self, mock_eval_metrics, sample_images_chw, sample_fom, temp_dir, mock_cfg):
        """Test eval_static without SLURM_JOB_ID environment variable."""
        mock_cfg.data_path = '/path/to/data/file.txt'
        
        eval_static(sample_images_chw, sample_fom, temp_dir, "TestModel", mock_cfg)
        
        # Verify that eval_metrics was called with 'local' fallback
        call_args = mock_eval_metrics.call_args[0]
        dataset_cfg = call_args[0]
        assert 'TestModel_local' in dataset_cfg[0]['name']


class TestEvaluation:
    """Test the evaluation function."""

    @patch('evaluation.ic')
    @patch('hydra.utils.instantiate')
    def test_evaluation_with_multiple_functions(self, mock_instantiate, mock_ic, sample_images_chw, sample_fom, mock_cfg):
        """Test evaluation with multiple evaluation functions."""
        # Mock evaluation functions
        mock_eval_static = Mock(return_value={'static_result': 'success'})
        mock_eval_static.__name__ = 'eval_static'
        
        mock_compute_entropy = Mock(return_value={'entropy': 0.85})
        mock_compute_entropy.__name__ = 'compute_entropy'
        
        mock_custom_viz = Mock(return_value={'viz_result': 'done'})
        mock_custom_viz.__name__ = 'custom_visualization_fn'
        
        mock_instantiate.side_effect = [mock_eval_static, mock_compute_entropy, mock_custom_viz]
        
        results = evaluation(sample_images_chw, sample_fom, "TestModel", mock_cfg)
        
        # Check that all functions were called
        assert len(results) == 3
        assert 'eval_static' in results
        assert 'compute_entropy' in results
        assert 'custom_visualization_fn' in results
        
        # Verify function calls with correct signatures
        mock_eval_static.assert_called_once_with(sample_images_chw, sample_fom, mock_cfg.savepath, "TestModel", mock_cfg)
        mock_compute_entropy.assert_called_once_with(sample_images_chw)
        mock_custom_viz.assert_called_once_with(sample_images_chw, sample_fom, mock_cfg.savepath, "TestModel")

    @patch('evaluation.ic')
    @patch('hydra.utils.instantiate')
    def test_evaluation_with_function_errors(self, mock_instantiate, mock_ic, sample_images_chw, sample_fom, mock_cfg, capsys):
        """Test evaluation handles function errors gracefully."""
        # Mock a function that raises an exception
        mock_failing_fn = Mock(side_effect=Exception("Test error"))
        mock_failing_fn.__name__ = 'failing_function'
        
        mock_successful_fn = Mock(return_value={'success': True})
        mock_successful_fn.__name__ = 'successful_function'
        
        mock_instantiate.side_effect = [mock_failing_fn, mock_successful_fn]
        
        # Modify config to have only 2 functions
        mock_cfg.evaluation.functions = [
            {'_target_': 'test.failing_function'},
            {'_target_': 'test.successful_function'}
        ]
        
        results = evaluation(sample_images_chw, sample_fom, "TestModel", mock_cfg)
        
        # Check that successful function result is included
        assert 'successful_function' in results
        assert results['successful_function'] == {'success': True}
        
        # Check that error was printed
        captured = capsys.readouterr()
        assert "Error running evaluation function" in captured.out
        assert "Test error" in captured.out

    @patch('evaluation.ic')
    @patch('hydra.utils.instantiate')
    def test_evaluation_with_nameless_function(self, mock_instantiate, mock_ic, sample_images_chw, sample_fom, mock_cfg):
        """Test evaluation with function that doesn't have __name__ attribute."""
        # Mock a function without __name__
        mock_fn = Mock(return_value={'result': 'success'})
        del mock_fn.__name__  # Remove __name__ attribute
        
        mock_instantiate.return_value = mock_fn
        
        # Modify config to have only 1 function
        mock_cfg.evaluation.functions = [{'_target_': 'test.nameless_function'}]
        
        results = evaluation(sample_images_chw, sample_fom, "TestModel", mock_cfg)
        
        # Check that function was executed and result stored with target name
        assert len(results) == 1
        assert 'test.nameless_function' in results
        assert results['test.nameless_function'] == {'result': 'success'}

    @patch('evaluation.ic')
    @patch('hydra.utils.instantiate')
    def test_evaluation_with_nn_distance_function(self, mock_instantiate, mock_ic, sample_images_chw, sample_fom, mock_cfg):
        """Test evaluation with nn_distance function (special signature)."""
        mock_nn_distance = Mock(return_value={'distance': 0.42})
        mock_nn_distance.__name__ = 'nn_distance_function'
        
        mock_instantiate.return_value = mock_nn_distance
        
        # Modify config to have nn_distance function
        mock_cfg.evaluation.functions = [{'_target_': 'nanophoto.evaluation.nn_distance'}]
        
        results = evaluation(sample_images_chw, sample_fom, "TestModel", mock_cfg)
        
        # Check that function was called with only images argument
        mock_nn_distance.assert_called_once_with(sample_images_chw)
        assert 'nn_distance_function' in results
        assert results['nn_distance_function'] == {'distance': 0.42}
"""
Unit tests for configuration module.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open

from src.config import Config, get_config, get_email_password


class TestConfig:
    """Test the Config class"""
    
    def test_config_loading_success(self, temp_config_file):
        """Test successful configuration loading"""
        config = Config(temp_config_file)
        
        assert config.get('exchange.name') == 'coinbase'
        assert config.get('trading.initial_capital') == 1000
        assert len(config.get_trading_pairs()) == 3
        assert 'BTC-USD' in config.get_trading_pairs()
    
    def test_config_file_not_found(self):
        """Test handling of missing config file"""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Config('nonexistent_config.yaml')
    
    def test_get_with_default(self, temp_config_file):
        """Test getting config values with defaults"""
        config = Config(temp_config_file)
        
        # Existing value
        assert config.get('trading.initial_capital', 500) == 1000
        
        # Non-existing value with default
        assert config.get('nonexistent.key', 'default_value') == 'default_value'
        
        # Non-existing value without default
        assert config.get('nonexistent.key') is None
    
    def test_get_trading_pairs(self, temp_config_file):
        """Test trading pairs retrieval"""
        config = Config(temp_config_file)
        pairs = config.get_trading_pairs()
        
        assert isinstance(pairs, list)
        assert len(pairs) == 3
        assert all(isinstance(pair, str) for pair in pairs)
    
    def test_get_exchange_config(self, temp_config_file):
        """Test exchange configuration retrieval"""
        config = Config(temp_config_file)
        exchange_config = config.get_exchange_config()
        
        assert isinstance(exchange_config, dict)
        assert exchange_config['name'] == 'coinbase'
        assert 'base_url' in exchange_config
    
    def test_get_trading_config(self, temp_config_file):
        """Test trading configuration retrieval"""
        config = Config(temp_config_file)
        trading_config = config.get_trading_config()
        
        assert isinstance(trading_config, dict)
        assert trading_config['initial_capital'] == 1000
        assert 'position_sizing' in trading_config
    
    def test_get_risk_config(self, temp_config_file):
        """Test risk management configuration"""
        config = Config(temp_config_file)
        risk_config = config.get_risk_config()
        
        assert isinstance(risk_config, dict)
        assert risk_config['stop_loss'] == 0.05
        assert risk_config['max_daily_loss'] == 0.10
    
    def test_get_backtesting_config(self, temp_config_file):
        """Test backtesting configuration"""
        config = Config(temp_config_file)
        backtest_config = config.get_backtesting_config()
        
        assert isinstance(backtest_config, dict)
        assert backtest_config['initial_capital'] == 1000
        assert backtest_config['commission'] == 0.005
    
    def test_get_strategy_weights(self, temp_config_file):
        """Test strategy weights retrieval"""
        config = Config(temp_config_file)
        weights = config.get_strategy_weights()
        
        assert isinstance(weights, dict)
        assert weights['swing'] == 1.0
        assert weights['rsi'] == 0.8
        assert weights['macd'] == 0.7
    
    def test_get_notification_config(self, temp_config_file):
        """Test notification configuration"""
        config = Config(temp_config_file)
        notification_config = config.get_notification_config()
        
        assert isinstance(notification_config, dict)
        assert 'email' in notification_config
        assert notification_config['email']['enabled'] is False
    
    def test_environment_variable_replacement(self):
        """Test environment variable replacement in config"""
        config_content = """
database:
  url: ${DATABASE_URL}
api:
  key: ${API_KEY}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch.dict(os.environ, {'DATABASE_URL': 'test_db_url', 'API_KEY': 'test_api_key'}):
                config = Config(temp_path)
                
                assert config.get('database.url') == 'test_db_url'
                assert config.get('api.key') == 'test_api_key'
        finally:
            os.unlink(temp_path)
    
    def test_environment_variable_not_found(self):
        """Test handling of missing environment variables"""
        config_content = """
api:
  key: ${NONEXISTENT_ENV_VAR}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            # Should return the original placeholder if env var doesn't exist
            assert config.get('api.key') == '${NONEXISTENT_ENV_VAR}'
        finally:
            os.unlink(temp_path)
    
    def test_nested_environment_variables(self):
        """Test environment variable replacement in nested structures"""
        config_content = """
database:
  connections:
    primary: ${PRIMARY_DB}
    secondary: ${SECONDARY_DB}
  settings:
    - name: timeout
      value: ${DB_TIMEOUT}
    - name: pool_size
      value: ${DB_POOL_SIZE}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch.dict(os.environ, {
                'PRIMARY_DB': 'primary_connection',
                'SECONDARY_DB': 'secondary_connection',
                'DB_TIMEOUT': '30',
                'DB_POOL_SIZE': '10'
            }):
                config = Config(temp_path)
                
                assert config.get('database.connections.primary') == 'primary_connection'
                assert config.get('database.connections.secondary') == 'secondary_connection'
                settings = config.get('database.settings')
                assert settings[0]['value'] == '30'
                assert settings[1]['value'] == '10'
        finally:
            os.unlink(temp_path)


class TestGlobalConfig:
    """Test global config functions"""
    
    def test_get_config_singleton(self, temp_config_file):
        """Test that get_config returns the same instance"""
        config1 = get_config(temp_config_file)
        config2 = get_config(temp_config_file)
        
        # Should be the same instance
        assert config1 is config2
    
    def test_get_config_different_paths(self, temp_config_file):
        """Test get_config with different config paths"""
        # Create a second temp config
        config_content = """
exchange:
  name: "test_exchange"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path2 = f.name
        
        try:
            config1 = get_config(temp_config_file)
            config2 = get_config(temp_path2)
            
            # Should be different instances
            assert config1 is not config2
            assert config1.get('exchange.name') == 'coinbase'
            assert config2.get('exchange.name') == 'test_exchange'
        finally:
            os.unlink(temp_path2)


class TestEnvironmentVariables:
    """Test environment variable getters"""
    
    def test_get_email_password(self):
        """Test email password retrieval from environment"""
        with patch.dict(os.environ, {'EMAIL_PASSWORD': 'test_password'}):
            assert get_email_password() == 'test_password'
        
        # Test default when not set
        with patch.dict(os.environ, {}, clear=True):
            assert get_email_password() == ''
    
    def test_coinbase_credentials(self):
        """Test Coinbase API credential getters"""
        from src.config import get_coinbase_api_key, get_coinbase_secret, get_coinbase_passphrase
        
        test_creds = {
            'COINBASE_API_KEY': 'test_api_key',
            'COINBASE_SECRET': 'test_secret',
            'COINBASE_PASSPHRASE': 'test_passphrase'
        }
        
        with patch.dict(os.environ, test_creds):
            assert get_coinbase_api_key() == 'test_api_key'
            assert get_coinbase_secret() == 'test_secret'
            assert get_coinbase_passphrase() == 'test_passphrase'
        
        # Test defaults when not set
        with patch.dict(os.environ, {}, clear=True):
            assert get_coinbase_api_key() == ''
            assert get_coinbase_secret() == ''
            assert get_coinbase_passphrase() == ''


class TestConfigEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_yaml_content(self):
        """Test handling of invalid YAML content"""
        invalid_yaml = """
        invalid: yaml: content: [
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(Exception):  # Should raise a YAML parsing error
                Config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_empty_config_file(self):
        """Test handling of empty configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')  # Empty file
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            # Should handle empty config gracefully
            assert config.get('nonexistent.key') is None
            assert config.get_trading_pairs() == []
        finally:
            os.unlink(temp_path)
    
    def test_config_with_none_values(self):
        """Test configuration with None values"""
        config_content = """
        some_key: null
        another_key: ~
        nested:
          value: null
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            assert config.get('some_key') is None
            assert config.get('another_key') is None
            assert config.get('nested.value') is None
        finally:
            os.unlink(temp_path)
    
    def test_deep_nested_path(self, temp_config_file):
        """Test accessing deeply nested configuration paths"""
        config = Config(temp_config_file)
        
        # This should not raise an error even if the path doesn't exist
        assert config.get('very.deeply.nested.path.that.does.not.exist') is None
        
        # Test with default
        assert config.get('very.deeply.nested.path', 'default') == 'default'
    
    def test_config_type_preservation(self):
        """Test that configuration values maintain their types"""
        config_content = """
        integer_value: 42
        float_value: 3.14
        boolean_value: true
        string_value: "hello"
        list_value:
          - item1
          - item2
        dict_value:
          key1: value1
          key2: value2
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = Config(temp_path)
            
            assert isinstance(config.get('integer_value'), int)
            assert config.get('integer_value') == 42
            
            assert isinstance(config.get('float_value'), float)
            assert config.get('float_value') == 3.14
            
            assert isinstance(config.get('boolean_value'), bool)
            assert config.get('boolean_value') is True
            
            assert isinstance(config.get('string_value'), str)
            assert config.get('string_value') == 'hello'
            
            assert isinstance(config.get('list_value'), list)
            assert len(config.get('list_value')) == 2
            
            assert isinstance(config.get('dict_value'), dict)
            assert config.get('dict_value.key1') == 'value1'
        finally:
            os.unlink(temp_path)